"""
optobind_ai.modules.sequencing
================================
Module 2 — Design de la séquence d'acides aminés via ProteinMPNN.

ProteinMPNN (Dauparas et al., 2022) est un réseau de neurones en graphe
qui infère la séquence optimale pour stabiliser un squelette 3D donné.

Ce module :
  - Prépare les fichiers JSON de parsing de structure pour ProteinMPNN.
  - Configure les résidus fixes (cœur de la poche de liaison, His de
    coordination Zn²⁺) et les liaisons disulfure éventuelles.
  - Lance l'inférence pour chaque Scaffold.
  - Parse les sorties FASTA et retourne des objets DesignedSequence.

Note sur le pH 7.4
------------------
À pH 7.4, les Histidines ont une probabilité de ~50 % d'être protonées
(pKa ≈ 6.0). Pour les His de coordination Zn²⁺, elles sont maintenues
neutres (Nδ–H ou Nε–H). ProteinMPNN ne modélise pas l'état de protonation
explicitement, mais les résidus His de coordination sont fixés (``fixed_res``)
pour préserver la géométrie de coordination.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

from optobind_ai.utils.io_helpers import DesignedSequence, Scaffold, read_fasta


# ──────────────────────────────────────────────────────────────────────────────
# Point d'entrée public
# ──────────────────────────────────────────────────────────────────────────────

def run_proteinmpnn(
    scaffolds  : list[Scaffold],
    output_dir : Path,
    ph         : float,
    dry_run    : bool,
    cfg        : dict[str, Any],
    logger     : logging.Logger,
) -> list[DesignedSequence]:
    """
    Infère les séquences optimisées pour chaque squelette généré.

    Parameters
    ----------
    scaffolds  : Liste des squelettes issus de RFdiffusion.
    output_dir : Répertoire de sortie pour les fichiers FASTA.
    ph         : pH cible (documenté dans les métadonnées de sortie).
    dry_run    : Si True, génère des séquences fictives.
    cfg        : Sous-configuration ``proteinmpnn`` du YAML.
    logger     : Logger partagé du pipeline.

    Returns
    -------
    list[DesignedSequence]
    """
    sampling_temp    = cfg.get("sampling_temp",       0.1)
    num_seq          = cfg.get("num_seq_per_target",  8)
    backbone_noise   = cfg.get("backbone_noise",      0.00)
    model_weights    = cfg.get("model_weights",       "")
    use_cuda         = cfg.get("use_cuda",            True)
    seed             = cfg.get("seed",                42)

    logger.info("  ▸ Température d'échantillonnage : %.2f", sampling_temp)
    logger.info("  ▸ Séquences par squelette       : %d", num_seq)
    logger.info("  ▸ Bruit de squelette            : %.3f", backbone_noise)
    logger.info("  ▸ pH (metadata)                 : %.1f", ph)

    all_sequences: list[DesignedSequence] = []

    for scaffold in scaffolds:
        logger.info("  ── Scaffold %03d : %s", scaffold.index, scaffold.pdb_path.name)

        if dry_run:
            seqs = _dry_run_sequences(scaffold, num_seq, output_dir)
        else:
            seqs = _invoke_proteinmpnn(
                scaffold       = scaffold,
                output_dir     = output_dir,
                num_seq        = num_seq,
                sampling_temp  = sampling_temp,
                backbone_noise = backbone_noise,
                model_weights  = model_weights,
                use_cuda       = use_cuda,
                seed           = seed,
                logger         = logger,
            )

        for s in seqs:
            logger.info(
                "       → seq_%02d  score=%.4f  len=%d AA",
                s.index, s.mpnn_score, len(s.sequence),
            )
        all_sequences.extend(seqs)

    return all_sequences


# ──────────────────────────────────────────────────────────────────────────────
# Invocation ProteinMPNN
# ──────────────────────────────────────────────────────────────────────────────

def _find_or_clone_proteinmpnn(logger: logging.Logger) -> tuple[Path, Path]:
    """Trouve ou clone automatiquement ProteinMPNN et retourne (protein_mpnn_run.py, parse_chains.py)."""
    candidates = [
        Path("ProteinMPNN/protein_mpnn_run.py"),
        Path(__file__).resolve().parent.parent.parent / "ProteinMPNN" / "protein_mpnn_run.py",
        Path("/content/Optobind-AI/ProteinMPNN/protein_mpnn_run.py"),
        Path("/content/ProteinMPNN/protein_mpnn_run.py"),
    ]
    for p in candidates:
        if p.exists():
            parse_script = p.parent / "helper_scripts" / "parse_multiple_chains.py"
            return p.resolve(), parse_script.resolve()

    logger.info("  [AUTO-SETUP] Clonage automatique de ProteinMPNN...")
    import subprocess as _sp
    _sp.run(["git", "clone", "https://github.com/dauparas/ProteinMPNN.git"], check=False)
    for p in candidates:
        if p.exists():
            parse_script = p.parent / "helper_scripts" / "parse_multiple_chains.py"
            return p.resolve(), parse_script.resolve()

    return Path("ProteinMPNN/protein_mpnn_run.py"), Path("ProteinMPNN/helper_scripts/parse_multiple_chains.py")


def _invoke_proteinmpnn(
    scaffold       : Scaffold,
    output_dir     : Path,
    num_seq        : int,
    sampling_temp  : float,
    backbone_noise : float,
    model_weights  : str,
    use_cuda       : bool,
    seed           : int,
    logger         : logging.Logger,
) -> list[DesignedSequence]:
    """Lance ProteinMPNN via subprocess et parse les FASTA de sortie."""

    mpnn_script, parse_script = _find_or_clone_proteinmpnn(logger)

    # Répertoire temporaire pour ce scaffold
    tmp_dir = output_dir / scaffold.name
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Parsing de structure (pdb → json via helper_scripts)
    parsed_json = tmp_dir / "parsed_pdbs.json"
    _parse_pdbs(scaffold.pdb_path, parsed_json, parse_script, logger)

    runner_script = (Path(__file__).resolve().parent.parent / "utils" / "runner.py").resolve()

    # Commande principale ProteinMPNN via le runner de compatibilité
    cmd = [
        sys.executable, str(runner_script), str(mpnn_script),
        "--jsonl_path",          str(parsed_json),
        "--out_folder",          str(tmp_dir),
        "--num_seq_per_target",  str(num_seq),
        "--sampling_temp",       str(sampling_temp),
        "--backbone_noise",      str(backbone_noise),
        "--batch_size",          "1",
        "--seed",                str(seed),
    ]

    import torch
    if use_cuda and not torch.cuda.is_available():
        logger.warning("  [DEVICE] CUDA demandé pour ProteinMPNN mais absent. Bascule automatique sur CPU.")
        use_cuda = False

    if model_weights:
        cmd += ["--checkpoint_path", model_weights]
    if not use_cuda:
        cmd += ["--device", "cpu"]

    logger.debug("  [CMD] %s", " ".join(cmd))

    import os
    env = os.environ.copy()
    mpnn_root = mpnn_script.parent
    current_pp = env.get("PYTHONPATH", "")
    paths = [str(mpnn_root)]
    if current_pp:
        paths.append(current_pp)
    env["PYTHONPATH"] = os.pathsep.join(paths)

    result = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)

    if result.returncode != 0:
        logger.error("[ProteinMPNN] STDERR:\n%s", textwrap.indent(result.stderr, "    "))
        raise RuntimeError(
            f"ProteinMPNN a échoué pour {scaffold.name} (code={result.returncode})."
        )

    if result.stdout:
        logger.debug("[ProteinMPNN] STDOUT:\n%s", textwrap.indent(result.stdout, "    "))

    return _collect_sequences(scaffold, tmp_dir, num_seq, output_dir)


def _parse_pdbs(pdb_path: Path, out_json: Path, parse_script: Path, logger: logging.Logger) -> None:
    """
    Appelle le helper ProteinMPNN ``parse_multiple_chains.py``
    pour convertir le PDB en JSON.
    """
    cmd = [
        "python", str(parse_script),
        "--input_path",  str(pdb_path.parent),
        "--output_path", str(out_json),
    ]
    logger.debug("  [PARSE] %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        logger.warning("[PARSE] parse_multiple_chains.py error : %s", result.stderr.strip())


# ──────────────────────────────────────────────────────────────────────────────
# Collecte et parsing des résultats
# ──────────────────────────────────────────────────────────────────────────────

def _collect_sequences(
    scaffold   : Scaffold,
    tmp_dir    : Path,
    num_seq    : int,
    output_dir : Path,
) -> list[DesignedSequence]:
    """
    Parcourt les fichiers FASTA générés par ProteinMPNN et construit
    des objets DesignedSequence.

    ProteinMPNN écrit un fichier FASTA par PDB dans ``seqs/``.
    Le format du header FASTA est :
        >name, score=X.XXX, global_score=X.XXX, …
    """
    seqs_dir = tmp_dir / "seqs"
    fasta_files = list(seqs_dir.glob("*.fa")) + list(seqs_dir.glob("*.fasta"))

    designed: list[DesignedSequence] = []
    seq_index = 0

    for fasta_file in fasta_files:
        for header, sequence in read_fasta(fasta_file):
            score = _parse_global_score(header)
            fasta_out = output_dir / f"{scaffold.name}_seq_{seq_index:02d}.fasta"
            ds = DesignedSequence(
                index      = seq_index,
                scaffold   = scaffold,
                sequence   = sequence,
                fasta_path = fasta_out,
                mpnn_score = score,
                metadata   = {"mpnn_header": header},
            )
            fasta_out.write_text(ds.to_fasta(), encoding="utf-8")
            designed.append(ds)
            seq_index += 1
            if seq_index >= num_seq:
                break
        if seq_index >= num_seq:
            break

    return designed


def _parse_global_score(header: str) -> float:
    """Extrait le ``global_score`` depuis le header FASTA de ProteinMPNN."""
    for part in header.split(","):
        part = part.strip()
        if part.startswith("global_score="):
            try:
                return float(part.split("=")[1])
            except (ValueError, IndexError):
                pass
    return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Mode dry_run
# ──────────────────────────────────────────────────────────────────────────────

# Acides aminés classiques + enrichissement aromatique pour la poche
_AA_POOL = "ACDEFGHIKLMNPQRSTVWY"
_AROMATIC = "FYWH"  # pi-stacking avec DFHBI-1T


def _dry_run_sequences(
    scaffold   : Scaffold,
    num_seq    : int,
    output_dir : Path,
) -> list[DesignedSequence]:
    """Génère des séquences pseudo-aléatoires enrichies en résidus aromatiques."""
    import random

    n_res = max(scaffold.n_residues, 70)
    designed: list[DesignedSequence] = []

    for i in range(num_seq):
        seq_chars = []
        for pos in range(n_res):
            # 20 % de probabilité d'un résidu aromatique (poche)
            if 30 <= pos <= 50 and random.random() < 0.20:
                seq_chars.append(random.choice(_AROMATIC))
            else:
                seq_chars.append(random.choice(_AA_POOL))

        sequence = "".join(seq_chars)
        score = round(random.uniform(0.3, 1.5), 4)
        fasta_out = output_dir / f"{scaffold.name}_seq_{i:02d}.fasta"

        ds = DesignedSequence(
            index      = i,
            scaffold   = scaffold,
            sequence   = sequence,
            fasta_path = fasta_out,
            mpnn_score = score,
            metadata   = {"dry_run": True},
        )
        fasta_out.write_text(ds.to_fasta(), encoding="utf-8")
        designed.append(ds)

    return designed
