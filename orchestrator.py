#!/usr/bin/env python3
"""
OptoBind-AI — Script d'orchestration principal
=================================================
Pipeline de design de novo de biosenseurs optiques (Zn²⁺ / DFHBI-1T à pH 7.4).

Enchaîne séquentiellement :
  1. RFdiffusion All-Atom  → génération des squelettes 3D
  2. ProteinMPNN           → optimisation des séquences
  3. AlphaFold 3 / Chai-1  → validation structurale et scoring

Usage
-----
    python orchestrator.py --target inputs/targets/zn2+_site.pdb \
                           --ligand  inputs/ligands/DFHBI-1T.sdf  \
                           --config  inputs/configs/default.yaml

Auteur  : OptoBind-AI Team
Version : 0.1.0
"""

from __future__ import annotations

# Patch de rétrocompatibilité Python 3.10+ (pour dgl et dépendances anciennes)
import collections
import collections.abc
for _attr in ("Mapping", "Iterable", "Sequence", "MutableMapping", "Callable"):
    if not hasattr(collections, _attr) and hasattr(collections.abc, _attr):
        setattr(collections, _attr, getattr(collections.abc, _attr))

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Sur Windows, redirection transparente vers l'environnement WSL2 Ubuntu
# qui contient PyTorch, DGL et toutes les dépendances GPU/Linux du pipeline.
# La redirection se fait TOUJOURS sur Windows sauf si on est déjà dans WSL2.
if sys.platform == "win32" and "--wsl-passthrough" not in sys.argv:
    if shutil.which("wsl"):
        cur_dir = str(Path.cwd().resolve())
        drive = cur_dir[0].lower()
        subpath = cur_dir[2:].replace("\\", "/")
        wsl_dir = f"/mnt/{drive}{subpath}"
        check_env = subprocess.run(
            ["wsl", "-d", "Ubuntu", "-e", "bash", "-c",
             "[ -f /home/colao/miniconda3/envs/optobind-ai/bin/python ] && echo ok"],
            capture_output=True, text=True,
        )
        if "ok" in check_env.stdout:
            # Créer un lien symbolique sans espace dans /tmp sous WSL2
            subprocess.run([
                "wsl", "-d", "Ubuntu", "-e", "bash", "-c",
                f"ln -sfn '{wsl_dir}' /tmp/optobind_project"
            ], check=False)
            safe_args = [a.replace("\\", "/") for a in sys.argv[1:]]
            cli_args = " ".join(f'"{a}"' for a in safe_args)
            print("[OptoBind-AI] Redirection vers WSL2 Ubuntu (env: optobind-ai)...")
            cmd = [
                "wsl", "-d", "Ubuntu", "-e", "bash", "-c",
                f"cd /tmp/optobind_project && PYTHONIOENCODING=utf-8 "
                f"/home/colao/miniconda3/envs/optobind-ai/bin/python orchestrator.py {cli_args}"
            ]
            ret = subprocess.run(cmd)
            sys.exit(ret.returncode)
        else:
            print("[AVERTISSEMENT] Environnement WSL2 optobind-ai introuvable. "
                  "Execution locale (modules GPU desactives).")
    else:
        print("[AVERTISSEMENT] WSL2 non detecte. Execution locale.")

from optobind_ai.utils.config     import load_config, validate_config
from optobind_ai.utils.logger     import setup_logger
from optobind_ai.utils.io_helpers import ensure_output_dirs, copy_inputs_snapshot
from optobind_ai.modules.scaffolding   import run_rfdiffusion, _build_contigs
from optobind_ai.modules.sequencing    import run_proteinmpnn
from optobind_ai.modules.validation    import run_alphafold
from optobind_ai.modules.reporting     import build_report, print_summary


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="orchestrator",
        description="OptoBind-AI — Pipeline de design de novo de biosenseurs optiques",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--target", "-t",
        type=Path,
        default=None,
        help="Fichier PDB de la cible (ex: site Zn²⁺). Si omis, lu depuis config ou défaut.",
    )
    parser.add_argument(
        "--ligand", "-l",
        type=Path,
        default=None,
        help="Fichier SDF/PDB du fluorophore (ex: DFHBI-1T). Si omis, lu depuis config ou défaut.",
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=Path("inputs/configs/default.yaml"),
        help="Fichier YAML de configuration du pipeline.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("outputs"),
        help="Répertoire racine des sorties.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Identifiant unique du run (généré automatiquement si absent).",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Sauter l'étape AlphaFold (pour des tests rapides).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simuler l'exécution sans appeler les outils externes.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Mode verbeux (niveau DEBUG).",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrateur principal
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()

    # ── 0. Initialisation ────────────────────────────────────────────────────
    run_id = args.run_id or f"run_{int(time.time())}"
    logger = setup_logger(
        name="optobind",
        log_dir=args.output_dir / "logs",
        run_id=run_id,
        verbose=args.verbose,
    )
    logger.info("=" * 60)
    logger.info(f"  OptoBind-AI  |  Run ID : {run_id}")
    logger.info("=" * 60)

    # ── 1. Chargement et validation de la configuration ───────────────────────
    logger.info("[CONFIG] Chargement de la configuration depuis %s", args.config)
    cfg = load_config(args.config)
    validate_config(cfg)

    n_variants  : int   = cfg["design"]["n_variants"]
    backbone_len: tuple = (cfg["design"]["min_length"], cfg["design"]["max_length"])
    ph_target   : float = cfg["design"]["pH"]

    logger.info(
        "[CONFIG] n_variants=%d | longueur=[%d-%d] AA | pH=%.1f",
        n_variants, *backbone_len, ph_target,
    )

    # ── 2. Vérification des fichiers d'entrée ─────────────────────────────────
    target_path = args.target or Path(cfg.get("inputs", {}).get("target", "inputs/targets/zn2+_site.pdb"))
    ligand_path = args.ligand or Path(cfg.get("inputs", {}).get("ligand", "inputs/ligands/DFHBI-1T.sdf"))

    for label, path in [("Cible (target)", target_path), ("Fluorophore (ligand)", ligand_path)]:
        if not path.exists():
            logger.error("[INPUT] Fichier introuvable : %s → %s", label, path)
            return 1
        logger.info("[INPUT] %s : %s  ✓", label, path)

    # ── 3. Création des répertoires de sortie ─────────────────────────────────
    out_dirs = ensure_output_dirs(args.output_dir, run_id)
    copy_inputs_snapshot(
        target=target_path,
        ligand=ligand_path,
        config=args.config,
        dest=out_dirs["inputs_snapshot"],
    )
    logger.info("[SETUP] Répertoires de sortie initialisés sous %s/", args.output_dir / run_id)

    # ══════════════════════════════════════════════════════════════════════════
    # MODULE 1 — Génération du squelette (RFdiffusion All-Atom)
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("┌─ MODULE 1 : Scaffolding (RFdiffusion All-Atom) ─────────────")
    t0 = time.time()
    scaffolds = run_rfdiffusion(
        target_pdb   = target_path,
        ligand_file  = ligand_path,
        output_dir   = out_dirs["pdbs"],
        n_variants   = n_variants,
        backbone_len = backbone_len,
        ph           = ph_target,
        dry_run      = args.dry_run,
        cfg          = cfg.get("rfdiffusion", {}),
        logger       = logger,
    )
    logger.info("└─ Scaffolding terminé en %.1fs — %d squelettes générés.", time.time() - t0, len(scaffolds))

    if not scaffolds:
        logger.error("[ERROR] Aucun squelette généré. Arrêt du pipeline.")
        return 1

    # ══════════════════════════════════════════════════════════════════════════
    # MODULE 2 — Design des séquences (ProteinMPNN)
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("┌─ MODULE 2 : Séquençage (ProteinMPNN) ───────────────────────")
    t0 = time.time()
    sequences = run_proteinmpnn(
        scaffolds   = scaffolds,
        output_dir  = out_dirs["fasta"],
        ph          = ph_target,
        dry_run     = args.dry_run,
        cfg         = {**cfg.get("proteinmpnn", {}),
                       "contigs": _build_contigs(cfg.get("rfdiffusion", {}), backbone_len)},
        logger      = logger,
    )
    logger.info("└─ Séquençage terminé en %.1fs — %d séquences optimisées.", time.time() - t0, len(sequences))

    # ══════════════════════════════════════════════════════════════════════════
    # MODULE 3 — Validation structurale (AlphaFold 3 / Chai-1)
    # ══════════════════════════════════════════════════════════════════════════
    validation_results = []
    if not args.skip_validation:
        logger.info("")
        logger.info("┌─ MODULE 3 : Validation (AlphaFold 3 / Chai-1) ──────────────")
        t0 = time.time()
        validation_results = run_alphafold(
            sequences   = sequences,
            ligand_file = ligand_path,
            target_pdb  = target_path,
            output_dir  = out_dirs["pdbs"],
            dry_run     = args.dry_run,
            cfg         = cfg.get("alphafold", {}),
            logger      = logger,
        )
        logger.info("└─ Validation terminée en %.1fs.", time.time() - t0)
    else:
        logger.warning("[SKIP] Étape de validation AlphaFold ignorée (--skip-validation).")

    # ══════════════════════════════════════════════════════════════════════════
    # RAPPORT FINAL
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("┌─ RAPPORT : Génération du rapport de validation ─────────────")
    report_path = build_report(
        scaffolds          = scaffolds,
        sequences          = sequences,
        validation_results = validation_results,
        output_path        = out_dirs["reports"] / "report.csv",
        run_id             = run_id,
        cfg                = cfg,
        logger             = logger,
    )
    logger.info("└─ Rapport sauvegardé : %s", report_path)

    # ── Résumé console ────────────────────────────────────────────────────────
    print_summary(
        scaffolds          = scaffolds,
        sequences          = sequences,
        validation_results = validation_results,
        report_path        = report_path,
        plddt_threshold    = cfg["validation"]["plddt_threshold"],
        rmsd_threshold     = cfg["validation"]["rmsd_threshold"],
    )

    logger.info("")
    logger.info("✓  Pipeline OptoBind-AI terminé avec succès. Run ID : %s", run_id)
    return 0


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sys.exit(main())
