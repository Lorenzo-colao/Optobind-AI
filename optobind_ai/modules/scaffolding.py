"""
optobind_ai.modules.scaffolding
================================
Module 1 — Génération du squelette 3D via RFdiffusion All-Atom.

RFdiffusion All-Atom (Watson et al., 2023) est un modèle de diffusion
génératif capable de produire des squelettes protéiques contraints par
la présence d'un ligand petit-molécule (ici DFHBI-1T) et d'un site de
coordination métallique (Zn²⁺) à pH 7.4.

Ce module :
  - Prépare le fichier de contraintes (hotspots, contigs) pour RFdiffusion.
  - Lance l'inférence via subprocess sur l'installation conda de RFdiffusion.
  - Parse les PDB générés et retourne des objets Scaffold.
  - En mode dry_run, génère des squelettes fictifs pour les tests CI.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

from optobind_ai.utils.io_helpers import Scaffold, count_pdb_residues


# ──────────────────────────────────────────────────────────────────────────────
# Point d'entrée public
# ──────────────────────────────────────────────────────────────────────────────

def run_rfdiffusion(
    target_pdb   : Path,
    ligand_file  : Path,
    output_dir   : Path,
    n_variants   : int,
    backbone_len : tuple[int, int],
    ph           : float,
    dry_run      : bool,
    cfg          : dict[str, Any],
    logger       : logging.Logger,
) -> list[Scaffold]:
    """
    Lance RFdiffusion All-Atom pour générer *n_variants* squelettes 3D.

    Parameters
    ----------
    target_pdb   : PDB de la cible (site Zn²⁺).
    ligand_file  : SDF/PDB du fluorophore DFHBI-1T.
    output_dir   : Répertoire de sortie pour les PDB générés.
    n_variants   : Nombre de squelettes à générer.
    backbone_len : (min_length, max_length) en résidus.
    ph           : pH cible (influence l'état de protonation des His).
    dry_run      : Si True, skip l'appel externe et crée des PDB factices.
    cfg          : Sous-configuration ``rfdiffusion`` du YAML.
    logger       : Logger partagé du pipeline.

    Returns
    -------
    list[Scaffold]
        Objets Scaffold avec les chemins PDB et métadonnées associées.
    """
    run_id = output_dir.parent.name
    logger.info("  ▸ Cible          : %s", target_pdb.name)
    logger.info("  ▸ Fluorophore    : %s", ligand_file.name)
    logger.info("  ▸ Variants       : %d", n_variants)
    logger.info("  ▸ Longueur chaîne: %d–%d AA", *backbone_len)
    logger.info("  ▸ pH cible       : %.1f", ph)

    # ── Préparation des paramètres ────────────────────────────────────────────
    contigs   = _build_contigs(cfg, backbone_len)
    hotspots  = cfg.get("hotspot_residues", [])
    num_steps = cfg.get("num_steps", 200)
    seed      = cfg.get("seed", 42)
    use_cuda  = cfg.get("use_cuda", True)

    logger.info("  ▸ Contigs        : %s", contigs)
    logger.info("  ▸ Hotspots       : %s", hotspots or "aucun")
    logger.info("  ▸ Steps          : %d  |  Seed : %d  |  CUDA : %s",
                num_steps, seed, use_cuda)

    scaffolds: list[Scaffold] = []

    if dry_run:
        logger.warning("  [DRY-RUN] Génération de squelettes fictifs.")
        scaffolds = _dry_run_scaffolds(output_dir, run_id, n_variants, backbone_len)
    else:
        scaffolds = _invoke_rfdiffusion(
            target_pdb  = target_pdb,
            ligand_file = ligand_file,
            output_dir  = output_dir,
            n_variants  = n_variants,
            contigs     = contigs,
            hotspots    = hotspots,
            num_steps   = num_steps,
            seed        = seed,
            use_cuda    = use_cuda,
            model_weights = cfg.get("model_weights", ""),
            run_id      = run_id,
            logger      = logger,
        )

    for s in scaffolds:
        logger.info("  ✓  Scaffold [%03d] : %s  (%d résidus)",
                    s.index, s.pdb_path.name, s.n_residues)
    return scaffolds


# ──────────────────────────────────────────────────────────────────────────────
# Construction de la commande RFdiffusion
# ──────────────────────────────────────────────────────────────────────────────

def _build_contigs(cfg: dict[str, Any], backbone_len: tuple[int, int]) -> str:
    """
    Construit la chaîne ``contigs`` passée à RFdiffusion.

    Le format est : ``A1-N/0 MIN-MAX`` où :
    - ``A1-N``  = chaîne fixe (site de liaison de la cible)
    - ``MIN-MAX`` = segment à halluciner (squelette de novo)
    """
    base = cfg.get("contigs", None)
    if base:
        return base
    lo, hi = backbone_len
    return f"A1-5/0 {lo}-{hi}"


def _find_or_clone_rfdiffusion(logger: logging.Logger) -> tuple[Path, Path]:
    """Trouve ou clone automatiquement RFdiffusion All-Atom et retourne (run_inference.py, rfdiff_root)."""
    candidates = [
        Path("RFdiffusion/run_inference.py"),
        Path("RFdiffusion/scripts/run_inference.py"),
        Path(__file__).resolve().parent.parent.parent / "RFdiffusion" / "run_inference.py",
        Path(__file__).resolve().parent.parent.parent / "RFdiffusion" / "scripts" / "run_inference.py",
        Path("/content/Optobind-AI/RFdiffusion/run_inference.py"),
        Path("/content/RFdiffusion/run_inference.py"),
    ]
    for p in candidates:
        if p.exists():
            rfdiff_root = p.parent if (p.parent / "config").exists() else (p.parent.parent if (p.parent.parent / "config").exists() else p.parent)
            return p.resolve(), rfdiff_root.resolve()

    logger.info("  [AUTO-SETUP] Clonage automatique de RFdiffusion All-Atom...")
    import subprocess as _sp
    _sp.run(["git", "clone", "--recurse-submodules", "https://github.com/baker-laboratory/rf_diffusion_all_atom.git", "RFdiffusion"], check=False)
    for p in candidates:
        if p.exists():
            rfdiff_root = p.parent if (p.parent / "config").exists() else (p.parent.parent if (p.parent.parent / "config").exists() else p.parent)
            return p.resolve(), rfdiff_root.resolve()

    return Path("RFdiffusion/run_inference.py"), Path("RFdiffusion")


def _patch_se3_pooling() -> None:
    """Corrige l'import AvgPooling dans SE3Transformer pour les versions récentes de DGL."""
    import site
    try:
        search_dirs = site.getsitepackages() + [site.getusersitepackages()]
    except Exception:
        search_dirs = []
    for sdir in search_dirs:
        p = Path(sdir) / "se3_transformer" / "model" / "layers" / "pooling.py"
        if p.exists():
            try:
                text = p.read_text(encoding="utf-8")
                if "from dgl.nn.pytorch import AvgPooling, MaxPooling" in text:
                    text = text.replace(
                        "from dgl.nn.pytorch import AvgPooling, MaxPooling",
                        "try:\n    from dgl.nn.pytorch import AvgPooling, MaxPooling\nexcept ImportError:\n    from dgl.nn import AvgPooling, MaxPooling",
                    )
                    p.write_text(text, encoding="utf-8")
            except Exception:
                pass


def _detect_ligand_name(pdb_path: Path, ligand_file: Path) -> str:
    """Détecte le code résidu du ligand (ex: ZN) présent dans le PDB cible ou son fichier."""
    if pdb_path.exists():
        try:
            with open(pdb_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.startswith("HETATM"):
                        res_name = line[17:20].strip()
                        if res_name:
                            return res_name
        except Exception:
            pass
    stem = ligand_file.stem.split("-")[0].split("_")[0]
    return stem[:3].upper() if len(stem) >= 3 else stem


def _invoke_rfdiffusion(
    target_pdb    : Path,
    ligand_file   : Path,
    output_dir    : Path,
    n_variants    : int,
    contigs       : str,
    hotspots      : list[str],
    num_steps     : int,
    seed          : int,
    use_cuda      : bool,
    model_weights : str,
    run_id        : str,
    logger        : logging.Logger,
) -> list[Scaffold]:
    """Lance RFdiffusion All-Atom via subprocess."""

    inference_script, rfdiff_root = _find_or_clone_rfdiffusion(logger)
    _patch_se3_pooling()

    # Préfixe de sortie
    output_prefix = str(output_dir / f"{run_id}_scaffold")

    # ── Résolution des chemins sans espaces pour Hydra ───────────────────────
    import os

    def _safe_path(p: Path) -> str:
        p = p.resolve()
        s = str(p)
        if " " not in s:
            return s
        link_name = "/tmp/optobind_" + p.name.replace(" ", "_")
        try:
            lp = Path(link_name)
            if lp.exists() or lp.is_symlink():
                lp.unlink()
            lp.symlink_to(p)
            return link_name
        except Exception:
            return s

    safe_target     = _safe_path(target_pdb)
    safe_prefix     = _safe_path(output_dir / f"{run_id}_scaffold")
    safe_weights    = _safe_path(Path(model_weights))

    # Commande native pour rf_diffusion_all_atom
    cmd = [
        sys.executable, str(inference_script),
        f"inference.input_pdb={safe_target}",
        f"inference.output_prefix={safe_prefix}",
        f"inference.num_designs={n_variants}",
        f"diffuser.T={num_steps}",
        f"contigmap.contigs=[{contigs}]",
        f"inference.ckpt_path={safe_weights}",
        "inference.deterministic=True",
    ]

    lig_code = _detect_ligand_name(target_pdb, ligand_file)
    if lig_code:
        cmd.append(f"inference.ligand={lig_code}")

    if hotspots:
        hs_str = "+".join(hotspots)
        cmd.append(f"ppi.hotspot_res=[{hs_str}]")

    logger.debug("  [CMD] %s", " ".join(cmd))

    # Injection du PYTHONPATH avec la racine du dépôt All-Atom et son sous-module rf2aa
    env = os.environ.copy()
    rf2aa_dir = rfdiff_root / "lib" / "rf2aa"
    current_pp = env.get("PYTHONPATH", "")
    paths = [str(rfdiff_root), str(rf2aa_dir)]
    if current_pp:
        paths.append(current_pp)
    env["PYTHONPATH"] = os.pathsep.join(paths)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        env=env,
        cwd=str(rfdiff_root),
    )

    if result.returncode != 0:
        logger.error("[RFdiffusion] STDERR:\n%s", textwrap.indent(result.stderr, "    "))
        raise RuntimeError(
            f"RFdiffusion a échoué (code={result.returncode}). "
            "Consultez les logs pour les détails."
        )

    if result.stdout:
        logger.debug("[RFdiffusion] STDOUT:\n%s", textwrap.indent(result.stdout, "    "))

    # Collecte des PDB générés
    return _collect_scaffolds(output_dir, run_id, n_variants)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _collect_scaffolds(output_dir: Path, run_id: str, n_variants: int) -> list[Scaffold]:
    """Scanne le répertoire et retourne les Scaffold correspondants aux PDB générés."""
    scaffolds: list[Scaffold] = []
    # Recherche robuste par nommage direct ou glob
    for i in range(n_variants):
        candidates = [
            output_dir / f"{run_id}_scaffold_{i}.pdb",
            output_dir / f"{run_id}_scaffold_{i:03d}.pdb",
            output_dir / f"{run_id}_scaffold_{i:04d}.pdb",
        ]
        found = None
        for c in candidates:
            if c.exists():
                found = c
                break
        if not found:
            # Recherche par motif
            matches = sorted(output_dir.glob(f"*{i}.pdb"))
            if matches:
                found = matches[0]

        if found:
            n_res = count_pdb_residues(found)
            scaffolds.append(Scaffold(index=i, pdb_path=found, run_id=run_id, n_residues=n_res))
        else:
            import logging as _logging
            _logging.getLogger("optobind").warning(
                "  [WARN] PDB attendu introuvable pour variant %d sous : %s", i, output_dir
            )
    return scaffolds


def _dry_run_scaffolds(
    output_dir   : Path,
    run_id       : str,
    n_variants   : int,
    backbone_len : tuple[int, int],
) -> list[Scaffold]:
    """
    Génère des fichiers PDB minimalistes pour les tests sans GPU.
    Produit une chaîne polyadénine (ALA) de longueur aléatoire dans [lo, hi].
    """
    import random

    lo, hi = backbone_len
    scaffolds: list[Scaffold] = []

    for i in range(n_variants):
        n_res  = random.randint(lo, hi)
        pdb_path = output_dir / f"{run_id}_scaffold_{i:03d}.pdb"
        _write_dummy_pdb(pdb_path, n_res)
        scaffolds.append(
            Scaffold(index=i, pdb_path=pdb_path, run_id=run_id, n_residues=n_res,
                     metadata={"dry_run": True})
        )
    return scaffolds


def _write_dummy_pdb(path: Path, n_res: int) -> None:
    """Écrit un PDB factice (squelette Cα linéaire) pour le mode dry_run."""
    lines = ["REMARK  OptoBind-AI DRY-RUN scaffold\n"]
    for i in range(1, n_res + 1):
        # Coordonnées Cα sur un axe z fictif
        x, y, z = 0.0, 0.0, float(i) * 3.8
        line = (
            f"ATOM  {i:5d}  CA  ALA A{i:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 99.00           C  \n"
        )
        lines.append(line)
    lines.append("END\n")
    path.write_text("".join(lines), encoding="utf-8")
