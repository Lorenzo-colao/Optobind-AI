"""
optobind_ai.utils.config
========================
Chargement, validation et gestion de la configuration YAML du pipeline.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


# ──────────────────────────────────────────────────────────────────────────────
# Valeurs par défaut
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG: dict[str, Any] = {
    "design": {
        "n_variants"  : 5,
        "min_length"  : 70,
        "max_length"  : 100,
        "pH"          : 7.4,
    },
    "rfdiffusion": {
        "model_weights"       : "weights/RFDiffusionAA_paper_weights.pt",
        "num_steps"           : 200,
        "noise_scale_ca"      : 1.0,
        "noise_scale_frame"   : 1.0,
        "contigs"             : "A1-5/0 70-100",
        "hotspot_residues"    : [],
        "auto_hotspots"       : True,
        "num_workers"         : 2,
        "use_cuda"            : True,
        "seed"                : 42,
    },
    "proteinmpnn": {
        "model_weights"       : "weights/ProteinMPNN_v_48_020.pt",
        "sampling_temp"       : 0.1,
        "num_seq_per_target"  : 8,
        "backbone_noise"      : 0.00,
        "use_cuda"            : True,
        "seed"                : 42,
    },
    "alphafold": {
        "backend"             : "chai-1",   # "alphafold3" | "chai-1" | "esmfold"
        "model_dir"           : "weights/chai-1/",
        "num_trunk_recycles"  : 1,
        "num_diffn_samples"   : 3,
        "use_esm_embeddings"  : True,
        "filter_native"       : True,
        "top_k_per_scaffold"  : 0,
        "use_cuda"            : True,
    },
    "validation": {
        "rmsd_threshold"      : 2.0,   # Å
        "plddt_threshold"     : 90.0,  # score AlphaFold
        "clash_vdw_cutoff"    : 0.6,   # fraction de rayon de Van der Waals
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# API publique
# ──────────────────────────────────────────────────────────────────────────────

def load_config(config_path: Path) -> dict[str, Any]:
    """
    Charge le fichier YAML et le fusionne avec la configuration par défaut.

    Les valeurs présentes dans le YAML ont la priorité sur les valeurs par
    défaut, permettant de ne surcharger que les paramètres souhaités.

    Parameters
    ----------
    config_path : Path
        Chemin vers le fichier ``default.yaml`` (ou tout autre fichier YAML).

    Returns
    -------
    dict[str, Any]
        Configuration complète et fusionnée.
    """
    cfg = copy.deepcopy(DEFAULT_CONFIG)

    if not config_path.exists():
        raise FileNotFoundError(
            f"[CONFIG] Fichier de configuration introuvable : {config_path}\n"
            "Créez le fichier ou copiez 'inputs/configs/default.yaml'."
        )

    with config_path.open("r", encoding="utf-8") as fh:
        user_cfg = yaml.safe_load(fh) or {}

    _deep_merge(cfg, user_cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    """
    Valide les contraintes critiques de la configuration.

    Raises
    ------
    ValueError
        Si une valeur obligatoire est manquante ou hors plage.
    """
    design = cfg.get("design", {})

    n = design.get("n_variants", 0)
    if not (1 <= n <= 50):
        raise ValueError(f"[CONFIG] n_variants={n} doit être compris entre 1 et 50.")

    lo, hi = design.get("min_length", 0), design.get("max_length", 0)
    if lo >= hi:
        raise ValueError(
            f"[CONFIG] min_length ({lo}) doit être strictement inférieur à max_length ({hi})."
        )
    if lo < 30 or hi > 500:
        raise ValueError(
            f"[CONFIG] Longueur protéique ({lo}–{hi} AA) hors de la plage raisonnable [30–500]."
        )

    ph = design.get("pH", 7.0)
    if not (4.0 <= ph <= 10.0):
        raise ValueError(f"[CONFIG] pH={ph} hors de la plage physio-chimique acceptable [4.0–10.0].")

    val = cfg.get("validation", {})
    if val.get("rmsd_threshold", 0) <= 0:
        raise ValueError("[CONFIG] rmsd_threshold doit être un nombre positif (en Å).")
    if not (0 < val.get("plddt_threshold", 0) <= 100):
        raise ValueError("[CONFIG] plddt_threshold doit être dans l'intervalle (0, 100].")


# ──────────────────────────────────────────────────────────────────────────────
# Interne
# ──────────────────────────────────────────────────────────────────────────────

def _deep_merge(base: dict, override: dict) -> None:
    """Fusionne ``override`` dans ``base`` en profondeur (in-place)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
