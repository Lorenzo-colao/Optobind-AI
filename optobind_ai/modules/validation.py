"""
optobind_ai.modules.validation
================================
Module 3 — Validation structurale via AlphaFold 3 / Chai-1 / ESMFold.

Ce module prédit la structure repliée du complexe ternaire
(Protéine + Zn²⁺ + DFHBI-1T) à partir de la séquence optimisée
par ProteinMPNN, puis évalue :

  1. **pLDDT** (global + poche de liaison) : confiance par résidu
     d'AlphaFold. Seuil requis : pLDDT > 90.
  2. **RMSD** (squelette RFdiffusion ↔ prédiction AlphaFold) :
     alignement Cα via Biopython. Seuil : RMSD < 2.0 Å.
  3. **Clashes stériques** : détection des paires d'atomes non liés
     à distance < 0.6 × somme des rayons de Van der Waals.

Backend supporté via la clé ``alphafold.backend`` du fichier YAML :
  - ``"chai-1"``    (défaut, mode local, open-source)
  - ``"esmfold"``   (ESMFold via la bibliothèque ``esm``)
  - ``"alphafold3"`` (AlphaFold 3 via interface officielle)
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

from optobind_ai.modules.biophysics import (
    analyze_cysteines,
    analyze_interface_interactions,
    analyze_metal_coordination,
    analyze_secondary_structure_and_ramachandran,
    analyze_sequence_liabilities,
    ensure_metal_coordination_conect,
)
from optobind_ai.utils.io_helpers import DesignedSequence, ValidationResult


# ──────────────────────────────────────────────────────────────────────────────
# Point d'entrée public
# ──────────────────────────────────────────────────────────────────────────────

def run_alphafold(
    sequences   : list[DesignedSequence],
    ligand_file : Path,
    target_pdb  : Path,
    output_dir  : Path,
    dry_run     : bool,
    cfg         : dict[str, Any],
    logger      : logging.Logger,
) -> list[ValidationResult]:
    """
    Prédit la structure des complexes ternaires et calcule les métriques
    de validation.

    Parameters
    ----------
    sequences   : Séquences issues de ProteinMPNN.
    ligand_file : Fichier SDF/PDB du fluorophore.
    target_pdb  : PDB de la cible (Zn²⁺).
    output_dir  : Répertoire de sortie pour les PDB prédits.
    dry_run     : Si True, génère des métriques fictives.
    cfg         : Sous-configuration ``alphafold`` du YAML.
    logger      : Logger partagé du pipeline.

    Returns
    -------
    list[ValidationResult]
    """
    backend   = cfg.get("backend",   "chai-1")
    model_dir = cfg.get("model_dir", "")
    use_cuda  = cfg.get("use_cuda",  True)

    logger.info("  ▸ Backend     : %s", backend)
    logger.info("  ▸ Model dir   : %s", model_dir or "(défaut)")
    logger.info("  ▸ CUDA        : %s", use_cuda)
    logger.info("  ▸ Sequences   : %d", len(sequences))

    results: list[ValidationResult] = []

    # ── Optimisation VRAM Chai-1 : Exécution en session persistante (Batch) ────
    if backend == "chai-1" and not dry_run and sequences:
        batch_results = _run_chai1_batch(
            sequences   = sequences,
            ligand_file = ligand_file,
            target_pdb  = target_pdb,
            output_dir  = output_dir,
            model_dir   = model_dir,
            use_cuda    = use_cuda,
            cfg         = cfg,
            logger      = logger,
        )
        for seq in sequences:
            if seq.name in batch_results:
                results.append(batch_results[seq.name])
            else:
                res = _predict_and_validate(
                    seq         = seq,
                    ligand_file = ligand_file,
                    target_pdb  = target_pdb,
                    output_dir  = output_dir,
                    backend     = backend,
                    model_dir   = model_dir,
                    use_cuda    = use_cuda,
                    cfg         = cfg,
                    logger      = logger,
                )
                results.append(res)
        return results

    try:
        from tqdm import tqdm
        seq_iter = tqdm(sequences, desc=f"  [3/3] Validation {backend.upper()}", unit="seq", dynamic_ncols=True)
    except ImportError:
        seq_iter = sequences

    for seq in seq_iter:
        logger.info("  ── %s  (len=%d AA)", seq.name, len(seq.sequence))

        if dry_run:
            res = _dry_run_validation(seq, output_dir)
        else:
            res = _predict_and_validate(
                seq         = seq,
                ligand_file = ligand_file,
                target_pdb  = target_pdb,
                output_dir  = output_dir,
                backend     = backend,
                model_dir   = model_dir,
                use_cuda    = use_cuda,
                cfg         = cfg,
                logger      = logger,
            )

        results.append(res)
        status = "✓ PASS" if res.passed else "✗ FAIL"
        if hasattr(seq_iter, "set_postfix_str"):
            seq_iter.set_postfix_str(f"{seq.name.split('_')[-2]}_{seq.name.split('_')[-1]} {status} (pLDDT={res.plddt_mean:.1f})")
        m_rmsd_str = f"mRMSD={res.motif_rmsd:.2f} Å" if res.motif_rmsd != float("inf") else "mRMSD=-"
        logger.info(
            "       %s  pLDDT=%.1f  pTM=%.3f  ipTM=%.3f  PAE=%.1f Å  ΔSASA=%.0f Å²  hydro=%.1f%%  hbonds=%d (%.2f/100Å²)  liab=%d  SS(H/E/C)=%.0f/%.0f/%.0f%%  rama=%.1f%%  RMSD=%.2f Å (%s)  clashes=%d  cys_free=%d",
            status, res.plddt_mean, res.ptm, res.iptm, res.pae_inter, res.delta_sasa, res.hydrophobic_sasa_pct, res.n_interface_hbonds, res.hbond_density, res.n_liabilities, res.helix_pct, res.beta_pct, res.coil_pct, res.ramachandran_outliers_pct, res.rmsd, m_rmsd_str, res.n_clashes, res.n_unpaired_cys,
        )

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Dispatcher de backend
# ──────────────────────────────────────────────────────────────────────────────

def _predict_and_validate(
    seq         : DesignedSequence,
    ligand_file : Path,
    target_pdb  : Path,
    output_dir  : Path,
    backend     : str,
    model_dir   : str,
    use_cuda    : bool,
    cfg         : dict[str, Any],
    logger      : logging.Logger,
) -> ValidationResult:
    """Dispatche vers le backend de prédiction approprié."""

    if backend == "chai-1":
        predicted_pdb, plddt_scores = _run_chai1(
            seq, ligand_file, target_pdb, output_dir, model_dir, use_cuda, cfg, logger
        )
    elif backend == "esmfold":
        predicted_pdb, plddt_scores = _run_esmfold(
            seq, output_dir, use_cuda, logger
        )
    elif backend == "alphafold3":
        predicted_pdb, plddt_scores = _run_alphafold3(
            seq, ligand_file, target_pdb, output_dir, model_dir, use_cuda, cfg, logger
        )
    else:
        raise ValueError(
            f"Backend inconnu : '{backend}'. "
            "Valeurs acceptées : 'chai-1', 'esmfold', 'alphafold3'."
        )

    # ── Calcul des métriques ──────────────────────────────────────────────────
    plddt_mean = sum(plddt_scores) / len(plddt_scores) if plddt_scores else 0.0

    rmsd = _compute_rmsd(
        reference_pdb  = seq.scaffold.pdb_path,
        predicted_pdb  = predicted_pdb,
        logger         = logger,
    ) if predicted_pdb else float("inf")

    n_clashes = _count_clashes(
        pdb_path       = predicted_pdb,
        vdw_cutoff     = cfg.get("clash_vdw_cutoff", 0.6),
        logger         = logger,
    ) if predicted_pdb else 0

    from optobind_ai.utils.config import DEFAULT_CONFIG
    rmsd_thresh  = cfg.get("rmsd_threshold", DEFAULT_CONFIG["validation"]["rmsd_threshold"])
    motif_thresh = cfg.get("motif_rmsd_threshold", DEFAULT_CONFIG["validation"].get("motif_rmsd_threshold", 1.5))
    plddt_thresh = cfg.get("plddt_threshold", DEFAULT_CONFIG["validation"]["plddt_threshold"])
    max_cys      = cfg.get("max_unpaired_cys", DEFAULT_CONFIG["validation"].get("max_unpaired_cys", 0))
    max_hydro    = cfg.get("max_hydrophobic_sasa_pct", DEFAULT_CONFIG["validation"].get("max_hydrophobic_sasa_pct", 35.0))
    max_coil     = cfg.get("max_coil_pct", DEFAULT_CONFIG["validation"].get("max_coil_pct", 45.0))
    max_rama     = cfg.get("max_ramachandran_outliers_pct", DEFAULT_CONFIG["validation"].get("max_ramachandran_outliers_pct", 2.0))
    min_hbd      = cfg.get("min_hbond_density", DEFAULT_CONFIG["validation"].get("min_hbond_density", 0.15))
    max_liab     = cfg.get("max_liabilities", DEFAULT_CONFIG["validation"].get("max_liabilities", 0))

    from optobind_ai.modules.sequencing import motif_positions_from_contigs
    contigs = cfg.get("contigs", DEFAULT_CONFIG["rfdiffusion"].get("contigs", ""))
    motif_positions = motif_positions_from_contigs(contigs, seq.scaffold.n_residues, logger)

    # pLDDT sur la poche de liaison (orienté motif si défini, sinon région cœur)
    if motif_positions and plddt_scores:
        m_scores = [plddt_scores[p - 1] for p in motif_positions if 0 <= p - 1 < len(plddt_scores)]
        plddt_pocket = sum(m_scores) / len(m_scores) if m_scores else (sum(plddt_scores) / len(plddt_scores))
    elif len(plddt_scores) > 50:
        pocket_slice = plddt_scores[30:51]
        plddt_pocket = sum(pocket_slice) / len(pocket_slice) if pocket_slice else 0.0
    else:
        plddt_pocket = sum(plddt_scores) / len(plddt_scores) if plddt_scores else 0.0
    if predicted_pdb and motif_positions:
        motif_rmsd = _compute_rmsd(
            reference_pdb   = seq.scaffold.pdb_path,
            predicted_pdb   = predicted_pdb,
            logger          = logger,
            residue_indices = motif_positions,
        )
    else:
        motif_rmsd = rmsd

    ptm, iptm = _parse_chai_tm_scores(output_dir / seq.name, logger)
    pae_mean, pae_inter = _parse_chai_pae(output_dir / seq.name, len(seq.sequence), logger)
    struct = _load_structure(predicted_pdb)
    sasa, delta_sasa, hydro_pct = _compute_sasa(predicted_pdb, logger, structure=struct)
    cys_info = analyze_cysteines(predicted_pdb, logger=logger, structure=struct)
    inter_info = analyze_interface_interactions(predicted_pdb, logger=logger, structure=struct)
    ss_info = analyze_secondary_structure_and_ramachandran(predicted_pdb, logger=logger, structure=struct)
    liab_info = analyze_sequence_liabilities(seq.sequence)
    hbond_density = round(100.0 * inter_info["n_hbonds"] / delta_sasa, 2) if delta_sasa > 0 else 0.0

    result = ValidationResult(
        sequence                  = seq,
        pdb_path                  = predicted_pdb,
        plddt_mean                = plddt_mean,
        plddt_pocket              = plddt_pocket,
        rmsd                      = rmsd,
        motif_rmsd                = motif_rmsd,
        n_clashes                 = n_clashes,
        ptm                       = ptm,
        iptm                      = iptm,
        pae_mean                  = pae_mean,
        pae_inter                 = pae_inter,
        sasa                      = sasa,
        delta_sasa                = delta_sasa,
        hydrophobic_sasa_pct      = hydro_pct,
        n_interface_hbonds        = inter_info["n_hbonds"],
        n_interface_salt_bridges  = inter_info["n_salt_bridges"],
        hbond_density             = hbond_density,
        n_unpaired_cys            = cys_info["n_unpaired"],
        unpaired_details          = cys_info["details"],
        helix_pct                 = ss_info["helix_pct"],
        beta_pct                  = ss_info["beta_pct"],
        coil_pct                  = ss_info["coil_pct"],
        ramachandran_outliers_pct = ss_info["ramachandran_outliers_pct"],
        n_ramachandran_outliers   = ss_info["n_ramachandran_outliers"],
        ramachandran_details      = ss_info["ramachandran_details"],
        n_liabilities             = liab_info["n_liabilities"],
        liabilities_details       = liab_info["details"],
    )
    result.validate(
        rmsd_thresh,
        plddt_thresh,
        motif_rmsd_threshold=motif_thresh,
        max_unpaired_cys=max_cys,
        max_hydrophobic_sasa_pct=max_hydro,
        max_coil_pct=max_coil,
        max_ramachandran_outliers_pct=max_rama,
        min_hbond_density=min_hbd,
        max_liabilities=max_liab,
    )
    return result


def _evaluate_single_prediction(
    seq         : DesignedSequence,
    out_dir     : Path,
    target_pdb  : Path,
    ligand_file : Path,
    cfg         : dict[str, Any],
    logger      : logging.Logger,
) -> ValidationResult:
    """Évalue immédiatement une structure prédite par Chai-1 dès sa sortie."""
    cif_file = _find_prediction_file(out_dir, ["*.cif"])
    if cif_file:
        try:
            import gemmi
            doc = gemmi.cif.read_file(str(cif_file))
            st = gemmi.make_structure_from_block(doc.sole_block())
            pdb_out = out_dir / "pred.pdb"
            st.write_minimal_pdb(str(pdb_out))
            ensure_metal_coordination_conect(pdb_out, target_pdb=target_pdb)
        except Exception as exc:
            logger.debug("[GEMMI] Conversion CIF vers PDB : %s", exc)

    pred_pdb = _find_prediction_file(out_dir, ["pred.pdb", "*.pdb", "pred.model_idx_0.cif", "*.cif"])
    if pred_pdb and pred_pdb.suffix.lower() == ".pdb":
        ensure_metal_coordination_conect(pred_pdb, target_pdb=target_pdb)
    plddt_scores = _parse_chai_plddt(out_dir, logger)
    plddt_mean = sum(plddt_scores) / len(plddt_scores) if plddt_scores else 0.0

    rmsd = _compute_rmsd(
        reference_pdb = seq.scaffold.pdb_path,
        predicted_pdb = pred_pdb,
        logger        = logger,
    ) if pred_pdb else float("inf")

    n_clashes = _count_clashes(
        pdb_path   = pred_pdb,
        vdw_cutoff = cfg.get("clash_vdw_cutoff", 0.6),
        logger     = logger,
    ) if pred_pdb else 0

    from optobind_ai.utils.config import DEFAULT_CONFIG
    rmsd_thresh  = cfg.get("rmsd_threshold", DEFAULT_CONFIG["validation"]["rmsd_threshold"])
    motif_thresh = cfg.get("motif_rmsd_threshold", DEFAULT_CONFIG["validation"].get("motif_rmsd_threshold", 1.5))
    plddt_thresh = cfg.get("plddt_threshold", DEFAULT_CONFIG["validation"]["plddt_threshold"])
    max_cys      = cfg.get("max_unpaired_cys", DEFAULT_CONFIG["validation"].get("max_unpaired_cys", 0))
    max_hydro    = cfg.get("max_hydrophobic_sasa_pct", DEFAULT_CONFIG["validation"].get("max_hydrophobic_sasa_pct", 35.0))
    max_coil     = cfg.get("max_coil_pct", DEFAULT_CONFIG["validation"].get("max_coil_pct", 45.0))
    max_rama     = cfg.get("max_ramachandran_outliers_pct", DEFAULT_CONFIG["validation"].get("max_ramachandran_outliers_pct", 2.0))
    min_hbd      = cfg.get("min_hbond_density", DEFAULT_CONFIG["validation"].get("min_hbond_density", 0.15))
    max_liab     = cfg.get("max_liabilities", DEFAULT_CONFIG["validation"].get("max_liabilities", 0))

    from optobind_ai.modules.sequencing import motif_positions_from_contigs
    contigs = cfg.get("contigs", DEFAULT_CONFIG["rfdiffusion"].get("contigs", ""))
    motif_positions = motif_positions_from_contigs(contigs, seq.scaffold.n_residues, logger)

    # pLDDT sur la poche de liaison (orienté motif si défini, sinon région cœur)
    if motif_positions and plddt_scores:
        m_scores = [plddt_scores[p - 1] for p in motif_positions if 0 <= p - 1 < len(plddt_scores)]
        plddt_pocket = sum(m_scores) / len(m_scores) if m_scores else (sum(plddt_scores) / len(plddt_scores))
    elif len(plddt_scores) > 50:
        pocket_slice = plddt_scores[30:51]
        plddt_pocket = sum(pocket_slice) / len(pocket_slice) if pocket_slice else 0.0
    else:
        plddt_pocket = sum(plddt_scores) / len(plddt_scores) if plddt_scores else 0.0
    if pred_pdb and motif_positions:
        motif_rmsd = _compute_rmsd(
            reference_pdb   = seq.scaffold.pdb_path,
            predicted_pdb   = pred_pdb,
            logger          = logger,
            residue_indices = motif_positions,
        )
    else:
        motif_rmsd = rmsd

    ptm, iptm = _parse_chai_tm_scores(out_dir, logger)
    pae_mean, pae_inter = _parse_chai_pae(out_dir, len(seq.sequence), logger)
    struct = _load_structure(pred_pdb)
    sasa, delta_sasa, hydro_pct = _compute_sasa(pred_pdb, logger, structure=struct)
    cys_info = analyze_cysteines(pred_pdb, logger=logger, structure=struct)
    inter_info = analyze_interface_interactions(pred_pdb, logger=logger, structure=struct)
    ss_info = analyze_secondary_structure_and_ramachandran(pred_pdb, logger=logger, structure=struct)
    liab_info = analyze_sequence_liabilities(seq.sequence)
    hbond_density = round(100.0 * inter_info["n_hbonds"] / delta_sasa, 2) if delta_sasa > 0 else 0.0

    res = ValidationResult(
        sequence                  = seq,
        pdb_path                  = pred_pdb,
        plddt_mean                = plddt_mean,
        plddt_pocket              = plddt_pocket,
        rmsd                      = rmsd,
        motif_rmsd                = motif_rmsd,
        n_clashes                 = n_clashes,
        ptm                       = ptm,
        iptm                      = iptm,
        pae_mean                  = pae_mean,
        pae_inter                 = pae_inter,
        sasa                      = sasa,
        delta_sasa                = delta_sasa,
        hydrophobic_sasa_pct      = hydro_pct,
        n_interface_hbonds        = inter_info["n_hbonds"],
        n_interface_salt_bridges  = inter_info["n_salt_bridges"],
        hbond_density             = hbond_density,
        n_unpaired_cys            = cys_info["n_unpaired"],
        unpaired_details          = cys_info["details"],
        helix_pct                 = ss_info["helix_pct"],
        beta_pct                  = ss_info["beta_pct"],
        coil_pct                  = ss_info["coil_pct"],
        ramachandran_outliers_pct = ss_info["ramachandran_outliers_pct"],
        n_ramachandran_outliers   = ss_info["n_ramachandran_outliers"],
        ramachandran_details      = ss_info["ramachandran_details"],
        n_liabilities             = liab_info["n_liabilities"],
        liabilities_details       = liab_info["details"],
    )
    res.validate(
        rmsd_thresh,
        plddt_thresh,
        motif_rmsd_threshold=motif_thresh,
        max_unpaired_cys=max_cys,
        max_hydrophobic_sasa_pct=max_hydro,
        max_coil_pct=max_coil,
        max_ramachandran_outliers_pct=max_rama,
        min_hbond_density=min_hbd,
        max_liabilities=max_liab,
    )
    return res


# ──────────────────────────────────────────────────────────────────────────────
# Backend Chai-1
# ──────────────────────────────────────────────────────────────────────────────

def _run_chai1_batch(
    sequences   : list[DesignedSequence],
    ligand_file : Path,
    target_pdb  : Path,
    output_dir  : Path,
    model_dir   : str,
    use_cuda    : bool,
    cfg         : dict[str, Any],
    logger      : logging.Logger,
) -> dict[str, ValidationResult]:
    """
    Exécute l'inférence Chai-1 en une seule session persistante pour toutes
    les séquences. Le modèle (~5.6 Go) est chargé UNE SEULE FOIS en VRAM
    au lieu d'être rechargé pour chaque séquence.
    """
    fasta_dir = output_dir / "_chai_fastas"
    fasta_dir.mkdir(parents=True, exist_ok=True)
    ligand_smiles = _sdf_to_smiles(ligand_file, logger)

    restraints_path = fasta_dir / "restraints_metal.csv"
    has_restraints = _write_chai_restraints(restraints_path, target_pdb)
    if has_restraints:
        logger.info("  🔒 Contraintes de coordination métallique Chai-1 activées (restraints_metal.csv)")

    tasks = []
    for seq in sequences:
        fpath = fasta_dir / f"{seq.name}.fasta"
        _write_chai_fasta(fpath, seq.sequence, ligand_smiles)
        out_p = output_dir / seq.name
        tasks.append({
            "name": seq.name,
            "fasta": str(fpath.resolve()),
            "out": str(out_p.resolve()),
            "constraint": str(restraints_path.resolve()) if has_restraints else None,
        })

    import json as _json
    tasks_file = fasta_dir / "batch_tasks.json"
    tasks_file.write_text(_json.dumps(tasks), encoding="utf-8")

    python_env_path = sys.executable
    recycles = int(cfg.get("num_trunk_recycles", 1))
    samples  = int(cfg.get("num_diffn_samples", 1))
    use_esm  = bool(cfg.get("use_esm_embeddings", True))

    batch_runner_py = f"""
import os
import sys
import json
import time
import pathlib
import shutil

os.environ["MPLBACKEND"] = "Agg"
os.environ["DISABLE_PANDERA_IMPORT_WARNING"] = "True"
try:
    import matplotlib
    matplotlib.use("Agg")
except Exception:
    pass

try:
    import torch
    # Activer TensorFloat-32 et semi-précision pour une inférence VRAM ultra-rapide
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    # Sur A100 (40/80 Go) ou G4 (96 Go), désactiver low_memory pour garder toutes les activations
    # en mémoire HBM2e ultra-rapide (2000 Go/s) au lieu de streamer via le bus CPU lent !
    is_high_vram = False
    if torch.cuda.is_available():
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if total_vram_gb >= 30:
            is_high_vram = True
    from chai_lab.chai1 import run_inference
except Exception as exc:
    print(f"CHAI1_INIT_FAIL: {{exc}}", file=sys.stderr)
    sys.exit(1)

with open('{tasks_file.as_posix()}', 'r', encoding='utf-8') as f:
    task_list = json.load(f)

vram_mode = "Direct HBM2e (haute vitesse)" if is_high_vram else "Low-Memory (optimisé 24GB)"
print(f"CHAI1_SESSION_START: {{len(task_list)}} séquences à replier avec poids chauds en VRAM [Mode : {{vram_mode}}].", flush=True)

for i, t in enumerate(task_list, 1):
    out_p = pathlib.Path(t['out'])
    fasta_p = pathlib.Path(t['fasta'])
    name = t['name']

    # Nettoyer le répertoire de sortie pour Chai-1
    if out_p.exists():
        for item in out_p.iterdir():
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink(missing_ok=True)
    out_p.mkdir(parents=True, exist_ok=True)

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.time()
            print(f"CHAI1_ITEM_START: {{i}}/{{len(task_list)}}:{{name}}", flush=True)
            constraint_p = pathlib.Path(t['constraint']) if t.get('constraint') else None
            cands = run_inference(
                fasta_file=fasta_p,
                output_dir=out_p,
                device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
                constraint_path=constraint_p if (constraint_p and constraint_p.exists()) else None,
                num_trunk_recycles={recycles},
                num_diffn_samples={samples},
                use_esm_embeddings={use_esm},
                low_memory=not is_high_vram,
            )
            # Sauvegarder la matrice PAE (Predicted Aligned Error) pour analyse inter-domaines
            try:
                if hasattr(cands, "pae") and cands.pae is not None:
                    import numpy as _np
                    pae_tensor = cands.pae.detach().cpu().numpy()
                    pae_mat = pae_tensor[0] if pae_tensor.ndim == 3 else pae_tensor
                    _np.save(out_p / "pae.npy", pae_mat.astype(_np.float32))
            except Exception:
                pass
            try:
                del cands
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            elapsed = time.time() - t0
            print(f"CHAI1_ITEM_DONE: {{i}}/{{len(task_list)}}:{{name}}:{{elapsed:.1f}}", flush=True)
            break
        except Exception as exc:
            err_str = str(exc)
            is_net_err = any(k in err_str.lower() for k in ("incompleteread", "connection", "timeout", "broken", "download", "http"))
            if attempt < max_retries and is_net_err:
                print(f"Tentative {{attempt}} interrompue (réseau). Reprise dans 5s...", file=sys.stderr, flush=True)
                time.sleep(5)
                continue
            print(f"CHAI1_ITEM_FAIL: {{name}} - {{exc}}", file=sys.stderr, flush=True)
            break

print("CHAI1_SESSION_DONE", flush=True)
sys.exit(0)
"""
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["DISABLE_PANDERA_IMPORT_WARNING"] = "True"
    for cand_dir in [
        os.environ.get("CHAI_DOWNLOADS_DIR", ""),
        "/content/chai_ssd",
        "/content/drive/MyDrive/optobind_weights_cache/chai",
        str(Path.home() / ".cache" / "chai"),
    ]:
        if cand_dir and Path(cand_dir).exists():
            env["CHAI_DOWNLOADS_DIR"] = str(cand_dir)
            break

    cmd = [python_env_path, "-u", "-c", batch_runner_py]
    logger.info("  [CHAI-1 BATCH] Lancement de la session d'inférence persistante (modèle conservé en VRAM)...")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    stdout_lines: list[str] = []
    evaluated_results: dict[str, ValidationResult] = {}
    seq_map = {s.name: s for s in sequences}

    import time as _time
    if proc.stdout:
        for line in iter(proc.stdout.readline, ""):
            stdout_lines.append(line)
            line_str = line.strip()

            if line_str.startswith("CHAI1_SESSION_START:"):
                total_s = line_str.split(":", 1)[1].strip()
                logger.info("  ⏳ Modèle Chai-1 prêt en VRAM — Début du repliement (%s)...", total_s)

            elif line_str.startswith("CHAI1_ITEM_START:"):
                parts = line_str.split(":", 2)
                progress = parts[1].strip()
                item_name = parts[2].strip() if len(parts) > 2 else ""
                logger.info("     ▸ [Chai-1 %s] Repliement en cours de %s...", progress, item_name)

            elif line_str.startswith("CHAI1_ITEM_DONE:"):
                parts = line_str.split(":", 3)
                progress = parts[1].strip()
                item_name = parts[2].strip() if len(parts) > 2 else ""
                dur = parts[3].strip() if len(parts) > 3 else ""
                dur_str = f" ({dur}s)" if dur else ""

                seq_obj = seq_map.get(item_name)
                if seq_obj:
                    res = _evaluate_single_prediction(
                        seq         = seq_obj,
                        out_dir     = output_dir / item_name,
                        target_pdb  = target_pdb,
                        ligand_file = ligand_file,
                        cfg         = cfg,
                        logger      = logger,
                    )
                    evaluated_results[item_name] = res
                    status = "✓ PASS" if res.passed else "✗ FAIL"
                    logger.info(
                        "     ✓ [Chai-1 %s] %s replié%s → %s  |  pLDDT=%.1f  |  pTM=%.3f  |  ipTM=%.3f  |  PAE=%.1f Å  |  ΔSASA=%.0f Å²  |  RMSD=%.2f Å  |  clashes=%d",
                        progress, item_name, dur_str, status, res.plddt_mean, res.ptm, res.iptm, res.pae_inter, res.delta_sasa, res.rmsd, res.n_clashes,
                    )
                else:
                    logger.info("     ✓ [Chai-1 %s] %s replié%s !", progress, item_name, dur_str)

            elif "CHAI1_ITEM_FAIL:" in line_str:
                logger.warning("     ✗ %s", line_str)

            elif "Tentative" in line_str and "interrompue" in line_str:
                logger.warning("     ⚠️ %s", line_str)

        proc.stdout.close()

    ret = proc.wait()
    if ret != 0:
        err_msg = "".join(stdout_lines[-30:])
        logger.warning("  [CHAI-1 BATCH] La session globale a rencontré une anomalie :\n%s", textwrap.indent(err_msg, "    "))
    else:
        logger.info("  ✓ [CHAI-1 BATCH] Repliement de toutes les structures terminé avec succès.")

    return evaluated_results


def _run_chai1(
    seq         : DesignedSequence,
    ligand_file : Path,
    target_pdb  : Path,
    output_dir  : Path,
    model_dir   : str,
    use_cuda    : bool,
    cfg         : dict[str, Any],
    logger      : logging.Logger,
) -> tuple[Path | None, list[float]]:
    """
    Lance Chai-1 via son API Python native run_inference() pour prédire
    le complexe ternaire (Protéine + DFHBI-1T + Zn²⁺).
    """
    # Chai-1 exige un répertoire de sortie STRICTEMENT vide.
    # On isole les fichiers FASTA d'entrée dans un sous-répertoire séparé.
    fasta_dir = output_dir / "_chai_fastas"
    fasta_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = fasta_dir / f"{seq.name}.fasta"

    ligand_smiles = _sdf_to_smiles(ligand_file, logger)
    _write_chai_fasta(fasta_path, seq.sequence, ligand_smiles)

    restraints_path = fasta_dir / "restraints_metal.csv"
    has_restraints = _write_chai_restraints(restraints_path, target_pdb)
    constraint_path_str = restraints_path.as_posix() if has_restraints else ""

    # Répertoire de sortie dédié et garanti vide pour Chai-1
    out_dir = output_dir / seq.name
    if out_dir.exists():
        import shutil
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    python_env_path = sys.executable
    recycles = int(cfg.get("num_trunk_recycles", 1))
    samples  = int(cfg.get("num_diffn_samples", 1))
    use_esm  = bool(cfg.get("use_esm_embeddings", True))

    runner_py = f"""
import os
import sys
import time
import pathlib
import shutil

# Forcer le backend non interactif pour éviter le conflit matplotlib_inline sous Jupyter/Colab
os.environ["MPLBACKEND"] = "Agg"
os.environ["DISABLE_PANDERA_IMPORT_WARNING"] = "True"
try:
    import matplotlib
    matplotlib.use("Agg")
except Exception:
    pass

try:
    import torch
    # Activer TensorFloat-32 et semi-précision pour une inférence VRAM ultra-rapide
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    is_high_vram = False
    if torch.cuda.is_available():
        if torch.cuda.get_device_properties(0).total_memory / (1024**3) >= 30:
            is_high_vram = True
    from chai_lab.chai1 import run_inference
except Exception as exc:
    print(f"CHAI1_FAIL: {{exc}}", file=sys.stderr)
    sys.exit(1)

out_p = pathlib.Path('{out_dir.as_posix()}')
fasta_p = pathlib.Path('{fasta_path.as_posix()}')
constraint_p = pathlib.Path('{constraint_path_str}') if '{constraint_path_str}' else None

max_retries = 3
for attempt in range(1, max_retries + 1):
    try:
        # Chai-1 exige un répertoire strictement vide à chaque tentative
        if out_p.exists():
            for item in out_p.iterdir():
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    item.unlink(missing_ok=True)

        cands = run_inference(
            fasta_file=fasta_p,
            output_dir=out_p,
            device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
            constraint_path=constraint_p if (constraint_p and constraint_p.exists()) else None,
            num_trunk_recycles={recycles},
            num_diffn_samples={samples},
            use_esm_embeddings={use_esm},
            low_memory=not is_high_vram,
        )
        try:
            if hasattr(cands, "pae") and cands.pae is not None:
                import numpy as _np
                pae_tensor = cands.pae.detach().cpu().numpy()
                pae_mat = pae_tensor[0] if pae_tensor.ndim == 3 else pae_tensor
                _np.save(out_p / "pae.npy", pae_mat.astype(_np.float32))
        except Exception:
            pass
        print("CHAI1_DONE")
        sys.exit(0)
    except Exception as exc:
        err_str = str(exc)
        is_net_err = any(k in err_str.lower() for k in ("incompleteread", "connection", "timeout", "broken", "download", "http"))
        if attempt < max_retries and is_net_err:
            print(f"Tentative {{attempt}} interrompue (réseau : {{exc}}). Reprise dans 5s...", file=sys.stderr)
            time.sleep(5)
            continue
        print(f"CHAI1_FAIL: {{exc}}", file=sys.stderr)
        sys.exit(1)
"""

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["DISABLE_PANDERA_IMPORT_WARNING"] = "True"
    for cand_dir in [
        os.environ.get("CHAI_DOWNLOADS_DIR", ""),
        "/content/chai_ssd",
        "/content/drive/MyDrive/optobind_weights_cache/chai",
        str(Path.home() / ".cache" / "chai"),
    ]:
        if cand_dir and Path(cand_dir).exists():
            env["CHAI_DOWNLOADS_DIR"] = str(cand_dir)
            break

    cmd = [python_env_path, "-u", "-c", runner_py]

    # Si la prédiction existe déjà (générée par _run_chai1_batch), on évite de réexécuter
    cif_file = _find_prediction_file(out_dir, ["*.cif"])
    pdb_file = _find_prediction_file(out_dir, ["pred.pdb", "*.pdb"])
    if not cif_file and not pdb_file:
        logger.debug("  [CHAI-1] Inférence unitaire pour %s...", seq.name)
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
        if result.returncode != 0:
            logger.warning("[CHAI-1] Non disponible ou a échoué :\n%s", textwrap.indent(result.stderr, "    "))
            return None, []
    else:
        logger.debug("  [CHAI-1] Prédiction déjà disponible via la session batch pour %s", seq.name)

    # ── Conversion du mmCIF en PDB standard via gemmi ─────────────────────────
    cif_file = _find_prediction_file(out_dir, ["*.cif"])
    if cif_file:
        try:
            import gemmi
            doc = gemmi.cif.read_file(str(cif_file))
            st = gemmi.make_structure_from_block(doc.sole_block())
            pdb_out = out_dir / "pred.pdb"
            st.write_minimal_pdb(str(pdb_out))
            ensure_metal_coordination_conect(pdb_out, target_pdb=target_pdb)
        except Exception as exc:
            logger.debug("[GEMMI] Conversion CIF vers PDB : %s", exc)

    pred_pdb = _find_prediction_file(out_dir, ["pred.pdb", "*.pdb", "pred.model_idx_0.cif", "*.cif"])
    if pred_pdb and pred_pdb.suffix.lower() == ".pdb":
        ensure_metal_coordination_conect(pred_pdb, target_pdb=target_pdb)
    plddt    = _parse_chai_plddt(out_dir, logger)
    return pred_pdb, plddt


# ──────────────────────────────────────────────────────────────────────────────
# Backend ESMFold
# ──────────────────────────────────────────────────────────────────────────────

def _run_esmfold(
    seq      : DesignedSequence,
    output_dir: Path,
    use_cuda  : bool,
    logger    : logging.Logger,
) -> tuple[Path | None, list[float]]:
    """Replie la séquence avec ESMFold (Meta, 2022). Ligand non inclus."""
    out_dir = output_dir / seq.name
    out_dir.mkdir(parents=True, exist_ok=True)
    predicted_pdb = out_dir / "esmfold_pred.pdb"

    cmd = [
        sys.executable, "-c",
        f"""
import esm, torch
model = esm.pretrained.esmfold_v1()
model = model.eval()
{'model = model.cuda()' if use_cuda else ''}
with torch.no_grad():
    output = model.infer_pdb('{seq.sequence}')
with open('{predicted_pdb}', 'w') as f:
    f.write(output)
print('ESMFold done.')
""",
    ]
    logger.debug("  [ESMFold] inférence pour %s", seq.name)
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        logger.error("[ESMFold] Erreur :\n%s", textwrap.indent(result.stderr, "    "))
        return None, []

    plddt = _parse_plddt_from_pdb(predicted_pdb, logger)
    return predicted_pdb, plddt


# ──────────────────────────────────────────────────────────────────────────────
# Backend AlphaFold 3
# ──────────────────────────────────────────────────────────────────────────────

def _run_alphafold3(
    seq         : DesignedSequence,
    ligand_file : Path,
    target_pdb  : Path,
    output_dir  : Path,
    model_dir   : str,
    use_cuda    : bool,
    cfg         : dict[str, Any],
    logger      : logging.Logger,
) -> tuple[Path | None, list[float]]:
    """
    Interface AlphaFold 3 via son script d'inférence officiel.

    AlphaFold 3 accepte un JSON de description du complexe (chaînes protéiques,
    ligands SMILES/CCD, ions métalliques).

    Ref : Abramson et al., Nature 2024, https://github.com/google-deepmind/alphafold3
    """
    import json as _json

    out_dir = output_dir / seq.name
    out_dir.mkdir(parents=True, exist_ok=True)

    ligand_smiles = _sdf_to_smiles(ligand_file, logger)

    # JSON d'entrée AlphaFold 3
    af3_input = {
        "name"     : seq.name,
        "sequences": [
            {"protein": {"id": "A", "sequence": seq.sequence}},
            {"ligand":  {"id": "B", "smiles": ligand_smiles}},
            {"ion":     {"id": "C", "ion": "ZN"}},
        ],
        "modelSeeds": [42],
    }
    input_json = out_dir / "af3_input.json"
    input_json.write_text(_json.dumps(af3_input, indent=2), encoding="utf-8")

    cmd = [
        sys.executable, "alphafold3/run_alphafold.py",
        "--input_json",  str(input_json),
        "--output_dir",  str(out_dir),
        "--model_dir",   model_dir or "weights/alphafold3/",
    ]
    logger.debug("  [AF3] %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        logger.error("[AF3] Erreur :\n%s", textwrap.indent(result.stderr, "    "))
        return None, []

    pred_pdb = _find_prediction_file(out_dir, ["*_model.cif", "*_model.pdb"])
    plddt    = _parse_plddt_from_pdb(pred_pdb, logger) if pred_pdb else []
    return pred_pdb, plddt


# ──────────────────────────────────────────────────────────────────────────────
# Métriques de validation
# ──────────────────────────────────────────────────────────────────────────────

def _compute_rmsd(
    reference_pdb  : Path,
    predicted_pdb  : Path | None,
    logger         : logging.Logger,
    residue_indices: list[int] | None = None,
) -> float:
    """
    Calcule le RMSD Cα entre le squelette de référence (RFdiffusion)
    et la structure prédite (AlphaFold / Chai-1).

    Si ``residue_indices`` est spécifié, calcule le RMSD exclusivement sur
    les résidus du motif fonctionnel (Motif RMSD).
    """
    if predicted_pdb is None or not predicted_pdb.exists():
        return float("inf")

    try:
        import numpy as np

        # Si le fichier prédit est un mmCIF (.cif), s'assurer d'avoir sa version PDB convertie
        if predicted_pdb.suffix.lower() in (".cif", ".mmcif"):
            pdb_conv = predicted_pdb.parent / "pred.pdb"
            if not pdb_conv.exists():
                try:
                    import gemmi
                    doc = gemmi.cif.read_file(str(predicted_pdb))
                    st = gemmi.make_structure_from_block(doc.sole_block())
                    st.write_minimal_pdb(str(pdb_conv))
                    predicted_pdb = pdb_conv
                except Exception:
                    pass
            else:
                predicted_pdb = pdb_conv

        target_res = set(residue_indices) if residue_indices else None

        def _get_ca(p: Path) -> np.ndarray:
            coords = []
            with p.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.startswith("ATOM") and line[12:16].strip() == "CA":
                        try:
                            if target_res is not None:
                                r_id = int(line[22:26].strip())
                                if r_id not in target_res:
                                    continue
                            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                        except (ValueError, IndexError):
                            pass
            return np.array(coords, dtype=np.float64)

        ref_coords = _get_ca(reference_pdb)
        pred_coords = _get_ca(predicted_pdb)

        n = min(len(ref_coords), len(pred_coords))
        if n == 0:
            return 0.0 if target_res is not None else float("inf")

        P = ref_coords[:n]
        Q = pred_coords[:n]

        # Centrage
        P_cent = P - P.mean(axis=0)
        Q_cent = Q - Q.mean(axis=0)

        # Matrice de covariance & SVD
        H = P_cent.T @ Q_cent
        U, S, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T

        # Superposition et RMSD
        diff = (P_cent @ R.T) - Q_cent
        rms = np.sqrt(np.mean(np.sum(diff**2, axis=-1)))
        return round(float(rms), 4)

    except Exception as exc:
        logger.warning("[RMSD] Erreur de calcul : %s", exc)
        return float("inf")


def _load_structure(pdb_path: Path | str | None) -> Any:
    """Charge et retourne l'objet Structure Biopython d'un fichier PDB/CIF en mémoire unique."""
    if pdb_path is None:
        return None
    p = Path(pdb_path)
    if not p.is_file():
        return None
    try:
        from Bio.PDB import PDBParser, MMCIFParser
        is_cif = p.suffix.lower() in [".cif", ".mmcif"]
        parser = MMCIFParser(QUIET=True) if is_cif else PDBParser(QUIET=True)
        return parser.get_structure("complex", str(p))
    except Exception:
        return None


def _count_clashes(
    pdb_path   : Path | None,
    vdw_cutoff : float,
    logger     : logging.Logger,
) -> int:
    """
    Détecte les clashes stériques entre atomes non liés via cKDTree O(N log N).
    Tolérant aux formats de colonnes PDB non standards (ligands, ions).
    """
    if pdb_path is None or not pdb_path.exists():
        return 0

    VDW_RADII: dict[str, float] = {
        "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52,
        "S": 1.80, "P": 1.80, "ZN": 1.39, "FE": 1.40, "F": 1.47,
    }
    DEFAULT_RADIUS = 1.70

    try:
        import numpy as np

        coords = []
        radii = []
        residues = []
        chains = []
        atom_names = []
        res_names = []
        elements = []

        with pdb_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        elem = line[76:78].strip().upper() if len(line) >= 78 and line[76:78].strip() else line[12:14].strip().upper()
                        elem = "".join(c for c in elem if c.isalpha())
                        ch = line[21]
                        r_seq = line[22:26].strip()
                        aname = line[12:16].strip().upper()
                        rname = line[17:20].strip().upper()

                        coords.append([x, y, z])
                        radii.append(VDW_RADII.get(elem, DEFAULT_RADIUS))
                        chains.append(ch)
                        residues.append(r_seq)
                        atom_names.append(aname)
                        res_names.append(rname)
                        elements.append(elem)
                    except (ValueError, IndexError):
                        pass

        if len(coords) < 2:
            return 0

        coords_arr = np.array(coords, dtype=np.float32)
        radii_arr  = np.array(radii, dtype=np.float32)

        metals = {"ZN", "FE", "CU", "MG", "MN", "NI", "CO", "CD", "CA"}
        clash_count = 0

        # Optimisation haute performance O(N log N) via SciPy cKDTree
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(coords_arr)
            max_search_r = float(vdw_cutoff * 2.0 * float(np.max(radii_arr)))
            candidate_pairs = tree.query_pairs(r=max_search_r, output_type="ndarray")

            for i, j in candidate_pairs:
                ch_i, ch_j = chains[i], chains[j]
                res_i, res_j = residues[i], residues[j]
                try:
                    if ch_i == ch_j and abs(int(res_i) - int(res_j)) <= 1:
                        continue
                except ValueError:
                    if ch_i == ch_j and res_i == res_j:
                        continue

                d = float(np.linalg.norm(coords_arr[i] - coords_arr[j]))
                thresh = float(vdw_cutoff * (radii_arr[i] + radii_arr[j]))
                if d >= thresh:
                    continue

                # Exclure les ponts disulfures légitimes (CYS SG - CYS SG entre 1.8 et 2.5 Å)
                if (atom_names[i] == "SG" and atom_names[j] == "SG"
                        and res_names[i] == "CYS" and res_names[j] == "CYS"
                        and 1.8 <= d <= 2.5):
                    continue

                # Exclure la coordination métallique légitime (métal - S/N/O entre 1.7 et 2.6 Å)
                elem_i, elem_j = elements[i], elements[j]
                if (elem_i in metals or elem_j in metals) and 1.7 <= d <= 2.6:
                    coord_elem = elem_j if elem_i in metals else elem_i
                    if coord_elem in {"S", "N", "O"}:
                        continue

                clash_count += 1

            return clash_count

        except Exception:
            # Fallback boucle vectorisée si cKDTree échoue
            n = len(coords_arr)
            for i in range(n):
                ch_i = chains[i]
                res_i = residues[i]
                dists = np.linalg.norm(coords_arr[i+1:] - coords_arr[i], axis=1)
                thresholds = vdw_cutoff * (radii_arr[i] + radii_arr[i+1:])
                clash_indices = np.where(dists < thresholds)[0]
                for idx in clash_indices:
                    j = i + 1 + idx
                    ch_j = chains[j]
                    res_j = residues[j]
                    try:
                        if ch_i == ch_j and abs(int(res_i) - int(res_j)) <= 1:
                            continue
                    except ValueError:
                        if ch_i == ch_j and res_i == res_j:
                            continue

                    d = float(dists[idx])
                    if (atom_names[i] == "SG" and atom_names[j] == "SG"
                            and res_names[i] == "CYS" and res_names[j] == "CYS"
                            and 1.8 <= d <= 2.5):
                        continue

                    elem_i, elem_j = elements[i], elements[j]
                    if (elem_i in metals or elem_j in metals) and 1.7 <= d <= 2.6:
                        coord_elem = elem_j if elem_i in metals else elem_i
                        if coord_elem in {"S", "N", "O"}:
                            continue

                    clash_count += 1

            return clash_count

    except Exception as exc:
        logger.warning("[CLASH] Erreur de calcul : %s", exc)
        return 0


def _compute_sasa(
    pdb_path: Path | None = None,
    logger: logging.Logger | None = None,
    structure: Any = None,
) -> tuple[float, float, float]:
    """
    Calcule :
      1. La surface accessible au solvant (SASA) totale.
      2. La surface enfouie à l'interface (ΔSASA) via l'algorithme de Shrake-Rupley.
      3. Le pourcentage de surface occupée par les résidus hydrophobes stricts (V, L, I, F, W, M).

    Returns
    -------
    tuple[float, float, float]
        (sasa_totale_Å², delta_sasa_interface_Å², hydrophobic_sasa_pct)
    """
    if structure is None:
        if pdb_path is None or not pdb_path.exists():
            return 0.0, 0.0, 0.0

        try:
            from Bio.PDB import PDBParser, MMCIFParser
            if pdb_path.suffix.lower() in (".cif", ".mmcif"):
                parser = MMCIFParser(QUIET=True)
            else:
                parser = PDBParser(QUIET=True)
            structure = parser.get_structure("complex", str(pdb_path))
        except Exception as exc:
            if logger:
                logger.debug("[SASA] Erreur parsing structure : %s", exc)
            return 0.0, 0.0, 0.0

    if structure is None or len(structure) == 0:
        return 0.0, 0.0, 0.0

    try:
        from Bio.PDB import ShrakeRupley, Structure, Model

        model = structure[0]
        has_sasa = any(hasattr(r, "sasa") for r in structure.get_residues())
        sr = ShrakeRupley()
        if not has_sasa:
            sr.compute(structure, level="R")
        total_sasa = round(float(sum(r.sasa for r in structure.get_residues() if hasattr(r, "sasa"))), 1)

        # Calcul du % hydrophobe sur la fraction protéique
        std_aa = {
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
            "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
        }
        hydro_aa = {"VAL", "LEU", "ILE", "PHE", "TRP", "MET"}
        prot_sasa = sum(r.sasa for r in structure.get_residues() if r.resname.upper() in std_aa and hasattr(r, "sasa"))
        hydro_sasa = sum(r.sasa for r in structure.get_residues() if r.resname.upper() in hydro_aa and hasattr(r, "sasa"))
        hydro_pct = round(float(hydro_sasa / prot_sasa * 100.0), 1) if prot_sasa > 0 else 0.0

        chains = list(model.get_chains())
        if len(chains) <= 1:
            return total_sasa, 0.0, hydro_pct

        # Calcul des surfaces des chaînes isolées : ΔSASA = Σ SASA(chaîne) - SASA(complexe)
        sum_isolated = 0.0
        for ch in chains:
            s_isolated = Structure.Structure(ch.id)
            m_isolated = Model.Model(0)
            s_isolated.add(m_isolated)
            m_isolated.add(ch.copy())
            sr.compute(s_isolated, level="R")
            sum_isolated += sum(r.sasa for r in s_isolated.get_residues() if hasattr(r, "sasa"))

        delta_sasa = max(0.0, round(float(sum_isolated - total_sasa), 1))
        return total_sasa, delta_sasa, hydro_pct

    except ImportError:
        if logger:
            logger.debug("[SASA] Biopython non disponible pour le calcul de SASA.")
        return 0.0, 0.0, 0.0
    except Exception as exc:
        if logger:
            logger.debug("[SASA] Erreur calcul SASA : %s", exc)
        return 0.0, 0.0, 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Parsers auxiliaires
# ──────────────────────────────────────────────────────────────────────────────

def _parse_plddt_from_pdb(pdb_path: Path | None, logger: logging.Logger) -> list[float]:
    """
    Lit les scores pLDDT depuis le champ B-factor d'un PDB AlphaFold.
    Un résidu = moyenne des B-factors de ses atomes.
    """
    if pdb_path is None or not pdb_path.exists():
        return []
    try:
        residues: dict[tuple, list[float]] = {}
        with pdb_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.startswith("ATOM"):
                    continue
                chain   = line[21]
                try:
                    res_seq = int(line[22:26].strip())
                except ValueError:
                    res_seq = line[22:26].strip()
                bfactor = float(line[60:66].strip())
                key = (chain, res_seq)
                residues.setdefault(key, []).append(bfactor)
        return [sum(v) / len(v) for v in residues.values()]
    except Exception as exc:
        logger.warning("[pLDDT] Erreur de parsing : %s", exc)
        return []


def _parse_chai_plddt(out_dir: Path, logger: logging.Logger) -> list[float]:
    """Lit les scores pLDDT depuis les fichiers JSON, NPZ ou PDB/CIF de Chai-1."""
    import json as _json

    # 1. Vérifier si scores.json existe (généré par notre runner Chai-1)
    for fname in ["scores.json", "scores.model_idx_0.json"]:
        score_file = out_dir / fname
        if score_file.exists():
            try:
                data = _json.loads(score_file.read_text(encoding="utf-8"))
                vals = [float(v) for v in data.get("per_residue_plddt", [])]
                if vals:
                    return vals
            except Exception as exc:
                logger.warning("[CHAI pLDDT JSON] %s", exc)

    # 2. Vérifier les fichiers NPZ de Chai-1 (ex: scores.model_idx_0.npz)
    for npz_path in sorted(out_dir.glob("scores*.npz")):
        try:
            import numpy as _np
            npz_data = _np.load(npz_path)
            for k in ("per_residue_plddt", "plddt", "lddt"):
                if k in npz_data:
                    arr = npz_data[k].flatten()
                    if arr.max() <= 1.0:
                        arr = arr * 100.0
                    return [round(float(x), 2) for x in arr]
        except Exception as exc:
            logger.warning("[CHAI pLDDT NPZ] %s", exc)

    # 3. Fallback : lire depuis le PDB
    pred_pdb = _find_prediction_file(out_dir, ["pred.pdb", "*.pdb"])
    if pred_pdb:
        vals = _parse_plddt_from_pdb(pred_pdb, logger)
        if vals:
            return vals

    # 4. Si seul le CIF est présent, le convertir via gemmi puis lire les B-factors
    cif_file = _find_prediction_file(out_dir, ["*.cif"])
    if cif_file:
        try:
            import gemmi
            doc = gemmi.cif.read_file(str(cif_file))
            st = gemmi.make_structure_from_block(doc.sole_block())
            pdb_out = out_dir / "pred.pdb"
            st.write_minimal_pdb(str(pdb_out))
            return _parse_plddt_from_pdb(pdb_out, logger)
        except Exception:
            pass

    return []


def _parse_chai_tm_scores(out_dir: Path, logger: logging.Logger) -> tuple[float, float]:
    """Lit les métriques pTM et ipTM générées par Chai-1 (fichiers NPZ ou JSON)."""
    ptm, iptm = 0.0, 0.0
    for npz_path in sorted(out_dir.glob("scores*.npz")):
        try:
            import numpy as _np
            npz_data = _np.load(npz_path)
            if "ptm" in npz_data:
                arr = npz_data["ptm"].flatten()
                if len(arr) > 0:
                    ptm = round(float(arr[0]), 4)
            if "iptm" in npz_data:
                arr = npz_data["iptm"].flatten()
                if len(arr) > 0:
                    iptm = round(float(arr[0]), 4)
            if ptm > 0 or iptm > 0:
                return ptm, iptm
        except Exception as exc:
            logger.debug("[CHAI TM NPZ] %s: %s", npz_path.name, exc)
    return ptm, iptm


def _parse_chai_pae(
    out_dir: Path,
    prot_len: int,
    logger: logging.Logger | None = None,
) -> tuple[float, float]:
    """
    Extrait les métriques PAE (Predicted Aligned Error) depuis pae.npy.

    Returns
    -------
    tuple[float, float]
        (pae_global_moyen_Å, pae_inter_domaines_ou_inter_chaines_Å)
    """
    pae_file = out_dir / "pae.npy"
    if not pae_file.exists():
        candidates = list(out_dir.glob("pae*.npy"))
        if candidates:
            pae_file = candidates[0]

    if pae_file.exists():
        try:
            import numpy as _np
            pae_mat = _np.load(pae_file)
            if pae_mat.ndim == 3:
                pae_mat = pae_mat[0]

            pae_mean = round(float(pae_mat.mean()), 2)
            n_tokens = pae_mat.shape[0]

            # Si le complexe comporte un ligand ou un partenaire (N > prot_len)
            if 0 < prot_len < n_tokens:
                block_a = pae_mat[:prot_len, prot_len:]
                block_b = pae_mat[prot_len:, :prot_len]
                pae_inter = round(float((block_a.mean() + block_b.mean()) / 2.0), 2)
            elif n_tokens > 10:
                # Monomère pur : évaluer l'orientation relative entre la moitié N-ter et C-ter
                half = n_tokens // 2
                block_a = pae_mat[:half, half:]
                block_b = pae_mat[half:, :half]
                pae_inter = round(float((block_a.mean() + block_b.mean()) / 2.0), 2)
            else:
                pae_inter = pae_mean

            return pae_mean, pae_inter
        except Exception as exc:
            if logger:
                logger.debug("[PAE] Erreur lecture matrice PAE : %s", exc)

    return 0.0, 0.0


def _sdf_to_smiles(ligand_file: Path, logger: logging.Logger) -> str:
    """
    Convertit un SDF en SMILES via RDKit (si disponible),
    ou retourne le SMILES hardcodé du DFHBI-1T comme fallback.
    """
    # SMILES exact du DFHBI-1T (PubChem CID 101889712)
    DFHBI1T_SMILES = "CC1=N/C(=C\\C2=CC(=C(C(=C2)F)O)F)/C(=O)N1CC(F)(F)F"

    if ligand_file.suffix.lower() == ".sdf":
        try:
            from rdkit import Chem
            mol = Chem.MolFromMolFile(str(ligand_file))
            if mol:
                return Chem.MolToSmiles(mol)
        except ImportError:
            logger.warning("[SDF→SMILES] RDKit non disponible. Utilisation du SMILES DFHBI-1T.")
        except Exception as exc:
            logger.warning("[SDF→SMILES] Erreur : %s", exc)

    return DFHBI1T_SMILES


def _write_chai_fasta(fasta_path: Path, protein_seq: str, ligand_smiles: str) -> None:
    """Écrit le fichier FASTA multi-entité au format Chai-1."""
    content = (
        f">protein|name=designed_protein\n{protein_seq}\n"
        f">ligand|name=DFHBI-1T\n{ligand_smiles}\n"
        f">ligand|name=Zn2+\n[Zn+2]\n"
    )
    fasta_path.write_text(content, encoding="utf-8")


def _write_chai_restraints(restraints_path: Path, target_pdb: Path) -> bool:
    """
    Génère un fichier de contraintes CSV pour Chai-1 afin de verrouiller
    et forcer la conservation stricte des liaisons de coordination métallique (Zn2+).
    """
    if not target_pdb or not target_pdb.exists():
        return False

    metal_info = analyze_metal_coordination(target_pdb)
    if metal_info.get("metals_count", 0) == 0 or not metal_info.get("coordination_residues"):
        return False

    import re as _re
    rows = [
        "restraint_id,chainA,res_idxA,chainB,res_idxB,connection_type,confidence,min_distance_angstrom,max_distance_angstrom,comment"
    ]
    aa_map = {"CYS": "C", "HIS": "H", "ASP": "D", "GLU": "E"}

    for idx, r_str in enumerate(metal_info["coordination_residues"], 1):
        m = _re.match(r"([A-Z]{3})(\d+)", r_str)
        if m:
            aa_3 = m.group(1)
            pos = m.group(2)
            aa_1 = aa_map.get(aa_3, "C")
            res_token = f"{aa_1}{pos}"
            max_d = 2.6 if aa_3 == "CYS" else 2.5
            rows.append(
                f"restraint_zn_{idx},C,,A,{res_token},pocket,1.0,0.0,{max_d},Coordination {r_str} - Zn2+"
            )

    if len(rows) > 1:
        restraints_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return True
    return False


def _find_prediction_file(directory: Path, patterns: list[str]) -> Path | None:
    """Cherche le premier fichier correspondant à l'un des patterns glob (tri déterministe)."""
    for pattern in patterns:
        matches = sorted(directory.glob(pattern))
        if matches:
            return matches[0]
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Mode dry_run
# ──────────────────────────────────────────────────────────────────────────────

def _dry_run_validation(
    seq        : DesignedSequence,
    output_dir : Path,
) -> ValidationResult:
    """Génère des métriques fictives pour les tests sans GPU."""
    import random

    plddt_mean   = round(random.uniform(75.0, 98.0), 2)
    plddt_pocket = round(random.uniform(70.0, 99.0), 2)
    ptm          = round(random.uniform(0.70, 0.95), 3)
    iptm         = round(random.uniform(0.60, 0.90), 3)
    pae_mean     = round(random.uniform(2.5, 6.0), 2)
    pae_inter    = round(random.uniform(3.0, 7.5), 2)
    sasa         = round(random.uniform(4500.0, 6500.0), 1)
    delta_sasa   = round(random.uniform(500.0, 950.0), 1)
    rmsd         = round(random.uniform(0.5, 3.5),   3)
    motif_rmsd   = round(random.uniform(0.3, 1.8),   3)
    n_clashes    = random.randint(0, 3)

    hydro_pct    = round(random.uniform(5.0, 22.0), 1)
    helix_pct    = round(random.uniform(50.0, 75.0), 1)
    beta_pct     = round(random.uniform(0.0, 15.0), 1)
    coil_pct     = round(100.0 - helix_pct - beta_pct, 1)
    rama_out     = round(random.uniform(0.0, 1.2), 1)

    n_hb = random.randint(1, 4)
    hb_dens = round(100.0 * n_hb / delta_sasa, 2) if delta_sasa > 0 else 0.0

    result = ValidationResult(
        sequence                  = seq,
        pdb_path                  = None,
        plddt_mean                = plddt_mean,
        plddt_pocket              = plddt_pocket,
        rmsd                      = rmsd,
        motif_rmsd                = motif_rmsd,
        n_clashes                 = n_clashes,
        ptm                       = ptm,
        iptm                      = iptm,
        pae_mean                  = pae_mean,
        pae_inter                 = pae_inter,
        sasa                      = sasa,
        delta_sasa                = delta_sasa,
        hydrophobic_sasa_pct      = hydro_pct,
        n_interface_hbonds        = n_hb,
        n_interface_salt_bridges  = random.randint(0, 1),
        hbond_density             = hb_dens,
        n_unpaired_cys            = 0,
        unpaired_details          = "-",
        helix_pct                 = helix_pct,
        beta_pct                  = beta_pct,
        coil_pct                  = coil_pct,
        ramachandran_outliers_pct = rama_out,
        n_ramachandran_outliers   = 1 if rama_out > 0 else 0,
        ramachandran_details      = "-" if rama_out == 0 else "GLY28",
        n_liabilities             = 0,
        liabilities_details       = "-",
        metadata                  = {"dry_run": True},
    )
    result.validate(
        rmsd_threshold=2.0,
        plddt_threshold=90.0,
        motif_rmsd_threshold=1.5,
        max_unpaired_cys=0,
        max_hydrophobic_sasa_pct=35.0,
        max_coil_pct=45.0,
        max_ramachandran_outliers_pct=2.0,
        min_hbond_density=0.15,
        max_liabilities=0,
    )
    return result
