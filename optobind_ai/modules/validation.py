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
        logger.info(
            "       %s  pLDDT=%.1f  RMSD=%.2f Å  clashes=%d",
            status, res.plddt_mean, res.rmsd, res.n_clashes,
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

    # pLDDT sur la poche de liaison (résidus 30–50, région aromatique)
    pocket_slice = plddt_scores[30:51] if len(plddt_scores) > 50 else plddt_scores
    plddt_pocket = sum(pocket_slice) / len(pocket_slice) if pocket_slice else 0.0

    from optobind_ai.utils.config import DEFAULT_CONFIG
    rmsd_thresh  = DEFAULT_CONFIG["validation"]["rmsd_threshold"]
    plddt_thresh = DEFAULT_CONFIG["validation"]["plddt_threshold"]

    result = ValidationResult(
        sequence     = seq,
        pdb_path     = predicted_pdb,
        plddt_mean   = plddt_mean,
        plddt_pocket = plddt_pocket,
        rmsd         = rmsd,
        n_clashes    = n_clashes,
    )
    result.validate(rmsd_thresh, plddt_thresh)
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
        except Exception as exc:
            logger.debug("[GEMMI] Conversion CIF vers PDB : %s", exc)

    pred_pdb = _find_prediction_file(out_dir, ["pred.pdb", "*.pdb", "pred.model_idx_0.cif", "*.cif"])
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

    pocket_slice = plddt_scores[30:51] if len(plddt_scores) > 50 else plddt_scores
    plddt_pocket = sum(pocket_slice) / len(pocket_slice) if pocket_slice else 0.0

    from optobind_ai.utils.config import DEFAULT_CONFIG
    rmsd_thresh  = cfg.get("rmsd_threshold", DEFAULT_CONFIG["validation"]["rmsd_threshold"])
    plddt_thresh = cfg.get("plddt_threshold", DEFAULT_CONFIG["validation"]["plddt_threshold"])

    res = ValidationResult(
        sequence     = seq,
        pdb_path     = pred_pdb,
        plddt_mean   = plddt_mean,
        plddt_pocket = plddt_pocket,
        rmsd         = rmsd,
        n_clashes    = n_clashes,
    )
    res.validate(rmsd_thresh, plddt_thresh)
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

    tasks = []
    for seq in sequences:
        fpath = fasta_dir / f"{seq.name}.fasta"
        _write_chai_fasta(fpath, seq.sequence, ligand_smiles)
        out_p = output_dir / seq.name
        tasks.append({"name": seq.name, "fasta": str(fpath.resolve()), "out": str(out_p.resolve())})

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
            cands = run_inference(
                fasta_file=fasta_p,
                output_dir=out_p,
                num_trunk_recycles={recycles},
                num_diffn_samples={samples},
                use_esm_embeddings={use_esm},
                low_memory=not is_high_vram,
            )
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
                        "     ✓ [Chai-1 %s] %s replié%s → %s  |  pLDDT=%.1f  |  RMSD=%.2f Å  |  clashes=%d",
                        progress, item_name, dur_str, status, res.plddt_mean, res.rmsd, res.n_clashes,
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
            num_trunk_recycles={recycles},
            num_diffn_samples={samples},
            use_esm_embeddings={use_esm},
            low_memory=not is_high_vram,
        )
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
        except Exception as exc:
            logger.debug("[GEMMI] Conversion CIF vers PDB : %s", exc)

    pred_pdb = _find_prediction_file(out_dir, ["pred.pdb", "*.pdb", "pred.model_idx_0.cif", "*.cif"])
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
    reference_pdb : Path,
    predicted_pdb : Path | None,
    logger        : logging.Logger,
) -> float:
    """
    Calcule le RMSD Cα entre le squelette de référence (RFdiffusion)
    et la structure prédite (AlphaFold / Chai-1).

    Algorithme de Kabsch vectorisé direct, tolérant aux PDB/mmCIF non standards
    (évite les plantages int('G2') du parser strict de Biopython).
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

        def _get_ca(p: Path) -> np.ndarray:
            coords = []
            with p.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.startswith("ATOM") and line[12:16].strip() == "CA":
                        try:
                            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                        except (ValueError, IndexError):
                            pass
            return np.array(coords, dtype=np.float64)

        ref_coords = _get_ca(reference_pdb)
        pred_coords = _get_ca(predicted_pdb)

        n = min(len(ref_coords), len(pred_coords))
        if n == 0:
            return float("inf")

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


def _count_clashes(
    pdb_path   : Path | None,
    vdw_cutoff : float,
    logger     : logging.Logger,
) -> int:
    """
    Détecte les clashes stériques entre atomes non liés.
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

                        coords.append([x, y, z])
                        radii.append(VDW_RADII.get(elem, DEFAULT_RADIUS))
                        chains.append(ch)
                        residues.append(r_seq)
                    except (ValueError, IndexError):
                        pass

        if len(coords) < 2:
            return 0

        coords_arr = np.array(coords, dtype=np.float32)
        radii_arr  = np.array(radii, dtype=np.float32)

        n = len(coords_arr)
        clash_count = 0
        for i in range(n):
            ch_i = chains[i]
            res_i = residues[i]
            dists = np.linalg.norm(coords_arr[i+1:] - coords_arr[i], axis=1)
            thresholds = vdw_cutoff * (radii_arr[i] + radii_arr[i+1:])

            # Ne tester que les atomes qui violent le seuil de Van der Waals
            clash_indices = np.where(dists < thresholds)[0]
            for idx in clash_indices:
                j = i + 1 + idx
                ch_j = chains[j]
                res_j = residues[j]
                # Exclure les liaisons chimiques intra-résidu ou le pont peptidique adjacent (|i - j| <= 1)
                try:
                    if ch_i == ch_j and abs(int(res_i) - int(res_j)) <= 1:
                        continue
                except ValueError:
                    if ch_i == ch_j and res_i == res_j:
                        continue
                clash_count += 1

        return clash_count

    except Exception as exc:
        logger.warning("[CLASH] Erreur de calcul : %s", exc)
        return 0


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


def _find_prediction_file(directory: Path, patterns: list[str]) -> Path | None:
    """Cherche le premier fichier correspondant à l'un des patterns glob."""
    for pattern in patterns:
        matches = list(directory.glob(pattern))
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
    rmsd         = round(random.uniform(0.5, 3.5),   3)
    n_clashes    = random.randint(0, 3)

    result = ValidationResult(
        sequence     = seq,
        pdb_path     = None,
        plddt_mean   = plddt_mean,
        plddt_pocket = plddt_pocket,
        rmsd         = rmsd,
        n_clashes    = n_clashes,
        metadata     = {"dry_run": True},
    )
    result.validate(rmsd_threshold=2.0, plddt_threshold=90.0)
    return result
