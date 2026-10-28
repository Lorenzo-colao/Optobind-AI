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


from optobind_ai.utils.patcher import patch_all_rfdiffusion

# ──────────────────────────────────────────────────────────────────────────────
# Point d'entrée public
# ──────────────────────────────────────────────────────────────────────────────
def _patch_rfdiffusion(rfdiff_root: Path | None = None) -> None:
    """
    Applique tous les patches requis (filter_het, collinearité SO(3), contigs)
    sur l'arborescence RFdiffusion All-Atom.
    """
    try:
        patch_all_rfdiffusion(rfdiff_root)
    except Exception as exc:
        logging.getLogger("optobind").warning(f"Erreur lors du patch de RFdiffusion : {exc}")



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
    Construit la chaîne ``contigs`` passée à RFdiffusion All-Atom.

    Dans RFdiffusion All-Atom, les segments sont séparés par des virgules sans espace :
    Ex : ``A1-1,70-100`` (conserver résidu A1 de la cible, halluciner 70-100 AA de novo).
    """
    base = cfg.get("contigs", None)
    if base:
        # Remplacer les slashes ou espaces par des virgules pour respecter la syntaxe All-Atom
        return base.replace("/", ",").replace(" ", ",")
    lo, hi = backbone_len
    return f"A1-1,{lo}-{hi}"


def _find_or_clone_rfdiffusion(logger: logging.Logger) -> tuple[Path, Path]:
    """Trouve ou clone automatiquement RFdiffusion All-Atom et retourne (run_inference.py, rfdiff_root)."""
    candidates = [
        Path("RFdiffusion/run_inference.py"),
        Path(__file__).resolve().parent.parent.parent / "RFdiffusion" / "run_inference.py",
        Path("/content/Optobind-AI/RFdiffusion/run_inference.py"),
        Path("/content/RFdiffusion/run_inference.py"),
    ]
    for p in candidates:
        if p.exists() and (p.parent / "aa_model.py").exists():
            return p.resolve(), p.parent.resolve()

    logger.info("  [AUTO-SETUP] Clonage automatique de RFdiffusion All-Atom...")
    import subprocess as _sp
    import shutil
    # Si un dossier RFdiffusion existe mais n'est pas All-Atom (ex: RosettaCommons standard), on le remplace
    rf_dir = Path("RFdiffusion")
    if rf_dir.exists() and not (rf_dir / "aa_model.py").exists():
        logger.info("  [AUTO-SETUP] Remplacement de l'ancien RFdiffusion par rf_diffusion_all_atom...")
        try:
            shutil.rmtree(rf_dir)
        except Exception:
            pass

    _sp.run(["git", "clone", "--recurse-submodules", "https://github.com/baker-laboratory/rf_diffusion_all_atom.git", "RFdiffusion"], check=False)
    for p in candidates:
        if p.exists() and (p.parent / "aa_model.py").exists():
            return p.resolve(), p.parent.resolve()

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
    """Détecte le code résidu du ligand organique en ignorant les ions métalliques."""
    metals = {"ZN", "MG", "CA", "FE", "CU", "MN", "NA", "K", "CL"}
    if pdb_path.exists():
        try:
            with open(pdb_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.startswith("HETATM"):
                        res_name = line[17:20].strip()
                        if res_name and res_name not in metals:
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
    _patch_rfdiffusion(rfdiff_root)

    # Préfixe de sortie
    output_prefix = str(output_dir / f"{run_id}_scaffold")

    # ── Résolution des chemins sans espaces pour Hydra ───────────────────────
    import os
    import sys as _sys
    import platform

    def _safe_path(p: Path) -> str:
        """
        Retourne un chemin utilisable par Hydra/RFdiffusion.
        Sur Windows natif : utilise le chemin absolu avec forward slashes.
        Sur Linux/WSL : tente un lien symbolique /tmp pour éviter les espaces.
        """
        p_res = p.resolve()
        s = str(p_res).replace("\\", "/")

        # Sur Linux/macOS, on tente un lien symbolique dans /tmp
        if platform.system() != "Windows":
            project_root = Path(__file__).resolve().parent.parent.parent
            tmp_project = Path("/tmp/optobind_project")
            try:
                if not tmp_project.exists():
                    tmp_project.symlink_to(project_root, target_is_directory=True)
                s_root = str(project_root).replace("\\", "/")
                if s_root in s:
                    s = s.replace(s_root, "/tmp/optobind_project")
            except Exception:
                pass

        return s

    safe_target  = _safe_path(target_pdb)
    safe_prefix  = _safe_path(output_dir / f"{run_id}_scaffold")
    safe_weights = _safe_path(Path(model_weights)) if model_weights else ""

    # Détecter si le ligand organique est présent dans le PDB cible
    lig_code = _detect_ligand_name(target_pdb, ligand_file)

    # Format des arguments Hydra : chaque argument est un élément séparé de la liste.
    # subprocess ne passe PAS par un shell, donc les guillemets internes n'aident pas.
    # Hydra accepte les espaces dans les valeurs si l'argument est un élément de liste entier.
    # Pour les chemins avec espaces, on utilise la syntaxe Hydra avec guillemets YAML : '+key="val"'
    # ou on échappe via le lien symbolique /tmp (Linux/WSL) déjà géré dans _safe_path.
    input_pdb_val  = safe_target
    output_pfx_val = safe_prefix

    # Commande native pour rf_diffusion_all_atom
    # contigmap.contigs=[...] — les crochets font partie de la valeur Hydra (liste YAML)
    cmd = [
        sys.executable, str(inference_script),
        f"inference.input_pdb={input_pdb_val}",
        f"inference.output_prefix={output_pfx_val}",
        f"inference.num_designs={n_variants}",
        f"diffuser.T={num_steps}",
        f"contigmap.contigs=['{contigs}']",
        "inference.deterministic=True",
    ]

    if safe_weights:
        cmd.append(f"inference.ckpt_path={safe_weights}")

    # N'ajouter inference.ligand que si un ligand organique est présent dans le PDB
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

    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    total_steps = n_variants * num_steps
    pbar = tqdm(total=total_steps, desc="  [1/3] RFdiffusion 3D", unit="step", dynamic_ncols=True) if has_tqdm else None

    import re
    re_timestep = re.compile(r"Timestep\s+(\d+)")
    re_design   = re.compile(r"making design\s+(\d+)\s+of\s+(\d+):(\d+)", re.IGNORECASE)

    stdout_lines = []
    stderr_lines = []

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        cwd=str(rfdiff_root),
    )

    current_design = 0
    if proc.stdout:
        for line in iter(proc.stdout.readline, ""):
            stdout_lines.append(line)
            m_des = re_design.search(line)
            if m_des:
                current_design = int(m_des.group(1)) + 1
                if pbar:
                    pbar.set_description(f"  [1/3] RFdiffusion [Variant {current_design}/{n_variants}]")

            m_ts = re_timestep.search(line)
            if m_ts:
                if pbar:
                    pbar.update(1)
                    pbar.set_postfix_str(f"t={m_ts.group(1)}")
            elif "Reading checkpoint" in line or "Loading checkpoint" in line:
                if pbar:
                    pbar.set_postfix_str("Chargement poids...")

        proc.stdout.close()

    ret = proc.wait()
    if pbar:
        # Compléter la barre si nécessaire
        if pbar.n < total_steps and ret == 0:
            pbar.update(total_steps - pbar.n)
        pbar.close()

    if ret != 0:
        err_msg = "".join(stdout_lines[-40:])
        logger.error("[RFdiffusion] Erreur d'exécution :\n%s", textwrap.indent(err_msg, "    "))
        raise RuntimeError(
            f"RFdiffusion a échoué (code={ret}). "
            "Consultez les logs pour les détails."
        )

    # Collecte des PDB générés
    return _collect_scaffolds(output_dir, run_id, n_variants)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _collect_scaffolds(output_dir: Path, run_id: str, n_variants: int) -> list[Scaffold]:
    """Scanne le répertoire et retourne les Scaffold correspondants aux PDB générés."""
    scaffolds: list[Scaffold] = []
    # Recherche robuste par nommage direct ou glob
    try:
        from tqdm import tqdm
        iterator = tqdm(range(n_variants), desc="  [1/3] Génération scaffolds (dry-run)", unit="scaffold")
    except ImportError:
        iterator = range(n_variants)

    for i in iterator:
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

    try:
        from tqdm import tqdm
        iterator = tqdm(range(n_variants), desc="  [1/3] Génération scaffolds (dry-run)", unit="scaffold")
    except ImportError:
        iterator = range(n_variants)

    for i in iterator:
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
