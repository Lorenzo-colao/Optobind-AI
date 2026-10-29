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


# ──────────────────────────────────────────────────────────────────────────────
# Backend Chai-1
# ──────────────────────────────────────────────────────────────────────────────

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
            low_memory=True,
        )

        # ── 1. Exporter les scores pLDDT par résidu directement ─────────────
        try:
            import json as _json
            plddt_arr = cands.plddt[0].cpu().numpy().flatten()
            # Mise à l'échelle 0-100 si nécessaire
            if plddt_arr.max() <= 1.0:
                plddt_arr = plddt_arr * 100.0
            plddt_list = [round(float(v), 2) for v in plddt_arr]
            with open(out_p / "scores.json", "w", encoding="utf-8") as _f:
                _json.dump({
                    "per_residue_plddt": plddt_list,
                    "mean_plddt": round(float(sum(plddt_list) / max(1, len(plddt_list))), 2),
                }, _f)
        except Exception as _e:
            pass

        # ── 2. Convertir pred.model_idx_0.cif en PDB standard via gemmi ─────
        try:
            import gemmi
            cif_file = out_p / "pred.model_idx_0.cif"
            if cif_file.exists():
                doc = gemmi.cif.read_file(str(cif_file))
                st = gemmi.make_structure_from_block(doc.sole_block())
                st.write_minimal_pdb(str(out_p / "pred.pdb"))
        except Exception as _e:
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

    cmd = [python_env_path, "-c", runner_py]
    logger.debug("  [CHAI-1] Inférence native via chai_lab.chai1.run_inference()...")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)

    if result.returncode != 0:
        logger.warning("[CHAI-1] Non disponible ou a échoué :\n%s", textwrap.indent(result.stderr, "    "))
        return None, []

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
        with pdb_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        elem = line[76:78].strip().upper() if len(line) >= 78 and line[76:78].strip() else line[12:14].strip().upper()
                        elem = "".join(c for c in elem if c.isalpha())
                        coords.append([x, y, z])
                        radii.append(VDW_RADII.get(elem, DEFAULT_RADIUS))
                    except (ValueError, IndexError):
                        pass

        if len(coords) < 2:
            return 0

        coords_arr = np.array(coords, dtype=np.float32)
        radii_arr  = np.array(radii, dtype=np.float32)

        n = len(coords_arr)
        clash_count = 0
        for i in range(n):
            dists = np.linalg.norm(coords_arr[i+1:] - coords_arr[i], axis=1)
            thresholds = vdw_cutoff * (radii_arr[i] + radii_arr[i+1:])
            clash_count += int(np.sum(dists < thresholds))

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
