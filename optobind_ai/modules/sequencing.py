"""
optobind_ai.modules.sequencing
================================
Module 2 — Design de la séquence d'acides aminés via ProteinMPNN / LigandMPNN.

Ce module intègre une sélection intelligente et automatique du modèle de design :
  - **LigandMPNN** (Dauparas et al., 2023) : Sélectionné dès qu'un ligand chimique
    (petite molécule organique, fluorophore, cofacteur) ou un ion métallique
    (Zn²⁺, Fe²⁺, Mg²⁺, etc.) est présent. LigandMPNN intègre les coordonnées 3D
    et les éléments chimiques de ces hétéroatomes pour optimiser les contacts de poche,
    les liaisons hydrogène et l'agencement stérique.
  - **ProteinMPNN Vanilla** (Dauparas et al., 2022) : Sélectionné pour les cibles
    protéiques pures (sans hétéroatome). Optimal pour la stabilité thermodynamique
    du squelette carboné.

Ce module :
  - Détecte automatiquement le meilleur modèle (mode "auto") ou respecte le choix explicite.
  - Injecte les coordonnées atomiques du ligand / métal dans le complexe analysé par LigandMPNN.
  - Verrouille automatiquement les résidus du motif (catalyse, coordination Zn²⁺).
  - Fournit une bascule transparente (fallback) vers ProteinMPNN en cas d'erreur.
  - Parse les sorties FASTA et retourne des objets DesignedSequence enrichis.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

from optobind_ai.modules.biophysics import compute_physicochemical_properties
from optobind_ai.utils.io_helpers import DesignedSequence, Scaffold, read_fasta


# ──────────────────────────────────────────────────────────────────────────────
# Détection automatique du modèle optimal (Auto-Selection Engine)
# ──────────────────────────────────────────────────────────────────────────────

def detect_optimal_mpnn_model(
    target_pdb       : Path | str | None = None,
    ligand_file      : Path | str | None = None,
    scaffold_pdb     : Path | str | None = None,
    configured_model : str = "auto",
    logger           : logging.Logger | None = None,
) -> tuple[str, str]:
    """
    Détecte automatiquement si LigandMPNN ou ProteinMPNN est le meilleur choix.

    Critères de décision :
      1. Si ``configured_model`` est explicite ('ligandmpnn' ou 'proteinmpnn'),
         ce choix est strictement respecté.
      2. Si ``configured_model == 'auto'`` :
         - Si un fichier ligand existe (SDF, MOL2, PDB non vide)
         - OU si des hétéroatomes non-eau (HETATM) ou ions métalliques sont détectés
           dans la cible ou le squelette
         → Sélectionne **LigandMPNN** (chimie de poche et coordination all-atom).
         - Si la cible est une protéine pure (strictement des acides aminés standards)
         → Sélectionne **ProteinMPNN** (stabilité maximale du squelette carboné).

    Returns
    -------
    tuple[str, str]
        (selected_model: "ligandmpnn" | "proteinmpnn", justification)
    """
    mode = (configured_model or "auto").strip().lower()
    if mode in ("ligandmpnn", "ligand_mpnn"):
        return "ligandmpnn", "Choix forcé par configuration (LigandMPNN)"
    if mode in ("proteinmpnn", "protein_mpnn", "vanilla"):
        return "proteinmpnn", "Choix forcé par configuration (ProteinMPNN vanilla)"

    # Mode automatique : analyse physique de la cible et des ligands
    has_ligand_file = False
    if ligand_file is not None:
        p_lig = Path(ligand_file)
        if p_lig.exists() and p_lig.is_file() and p_lig.stat().st_size > 0:
            has_ligand_file = True

    has_hetatm = False
    has_metal = False
    metals = {"ZN", "FE", "CU", "MG", "MN", "NI", "CO", "CD", "CA", "NA", "K"}
    std_aa = {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
        "HOH", "WAT", "TIP",
    }

    pdbs_to_scan = []
    if target_pdb is not None and Path(target_pdb).exists():
        pdbs_to_scan.append(Path(target_pdb))
    if scaffold_pdb is not None and Path(scaffold_pdb).exists():
        pdbs_to_scan.append(Path(scaffold_pdb))

    for pdb_path in pdbs_to_scan:
        try:
            with open(pdb_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.startswith("HETATM"):
                        res_name = line[17:20].strip().upper()
                        atom_name = line[12:16].strip().upper()
                        elem = line[76:78].strip().upper() if len(line) >= 78 else ""
                        if res_name not in std_aa:
                            has_hetatm = True
                        if elem in metals or any(atom_name.startswith(m) for m in metals) or res_name in metals:
                            has_metal = True
                            has_hetatm = True
        except Exception as exc:
            if logger:
                logger.debug("  [AUTO-DETECT] Erreur scan %s : %s", pdb_path.name, exc)

    if has_ligand_file or has_hetatm or has_metal:
        reasons = []
        if has_metal:
            reasons.append("ion(s) métallique(s) détecté(s)")
        if has_ligand_file or (has_hetatm and not has_metal):
            reasons.append("ligand organique / petite molécule détecté(e)")
        reason_str = " + ".join(reasons) if reasons else "hétéroatomes détectés"
        return "ligandmpnn", f"Automatique : {reason_str} → LigandMPNN optimal pour la poche"

    return "proteinmpnn", "Automatique : cible protéique pure (aucun hétéroatome) → ProteinMPNN optimal"


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
    Sélectionne automatiquement LigandMPNN ou ProteinMPNN selon la cible.

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
    sampling_temp       = cfg.get("sampling_temp",       0.1)
    num_seq             = cfg.get("num_seq_per_target",  8)
    backbone_noise      = cfg.get("backbone_noise",      0.00)
    model_weights       = cfg.get("model_weights",       "")
    ligandmpnn_weights  = cfg.get("ligandmpnn_weights",  "weights/ligandmpnn_v_32_010_25.pt")
    use_cuda            = cfg.get("use_cuda",            True)
    seed                = cfg.get("seed",                42)
    fix_motif           = cfg.get("fix_motif",           True)
    contigs             = cfg.get("contigs",             "")
    model_mode          = cfg.get("model",               "auto")
    target_pdb          = cfg.get("target_pdb",          None)
    ligand_file         = cfg.get("ligand_file",         None)
    ligand_cutoff       = cfg.get("ligand_cutoff",       8.0)

    # ── Détection automatique du modèle optimal ──────────────────────────────
    first_scaffold_pdb = scaffolds[0].pdb_path if scaffolds else None
    selected_model, model_reason = detect_optimal_mpnn_model(
        target_pdb       = target_pdb,
        ligand_file      = ligand_file,
        scaffold_pdb     = first_scaffold_pdb,
        configured_model = model_mode,
        logger           = logger,
    )

    logger.info("  ▸ Modèle sélectionné          : %s", selected_model.upper())
    logger.info("  ▸ Justification               : %s", model_reason)
    logger.info("  ▸ Température d'échantillonnage : %.2f", sampling_temp)
    logger.info("  ▸ Séquences par squelette       : %d", num_seq)
    logger.info("  ▸ Bruit de squelette            : %.3f", backbone_noise)
    logger.info("  ▸ pH (metadata)                 : %.1f", ph)

    all_sequences: list[DesignedSequence] = []

    if dry_run:
        for scaffold in scaffolds:
            logger.info("  ── Scaffold %03d : %s", scaffold.index, scaffold.pdb_path.name)
            seqs = _dry_run_sequences(scaffold, num_seq, output_dir)
            for s in seqs:
                s.metadata["mpnn_model"] = selected_model
                logger.info(
                    "       → seq_%02d  score=%.4f  len=%d AA  [%s]",
                    s.index, s.mpnn_score, len(s.sequence), selected_model,
                )
            all_sequences.extend(seqs)
        return all_sequences

    # Pré-calcul des positions exactes du motif par squelette (.trb + géométrie 3D + contigs)
    fixed_by_scaffold: dict[str, list[int]] = {}
    for scaffold in scaffolds:
        fixed_by_scaffold[scaffold.name] = (
            motif_positions_from_contigs(
                contigs,
                scaffold.n_residues,
                logger,
                scaffold_pdb=scaffold.pdb_path,
                target_pdb=target_pdb,
            )
            if fix_motif else []
        )

    # Optimisation Batch GPU : Chargement unique de LigandMPNN en VRAM si > 1 squelette
    if selected_model == "ligandmpnn" and len(scaffolds) > 1:
        try:
            batch_seqs = _invoke_ligandmpnn_batch(
                scaffolds         = scaffolds,
                output_dir        = output_dir,
                num_seq           = num_seq,
                sampling_temp     = sampling_temp,
                model_weights     = ligandmpnn_weights,
                use_cuda          = use_cuda,
                seed              = seed,
                logger            = logger,
                fixed_by_scaffold = fixed_by_scaffold,
                target_pdb        = target_pdb,
                ligand_cutoff     = ligand_cutoff,
                pack_side_chains  = bool(cfg.get("pack_side_chains", False)),
            )
            for s in batch_seqs:
                s.metadata["mpnn_model"] = "ligandmpnn"
                logger.info(
                    "       → %s  score=%.4f  len=%d AA  [ligandmpnn-batch]",
                    s.name, s.mpnn_score, len(s.sequence),
                )
            return batch_seqs
        except Exception as exc:
            logger.warning(
                "  [LIGANDMPNN BATCH] Repli sur exécution séquentielle (%s).", exc
            )

    try:
        from tqdm import tqdm
        scaffold_iter = tqdm(
            scaffolds,
            desc=f"  [2/3] {selected_model.upper()} Séquençage",
            unit="scaffold",
            dynamic_ncols=True,
        )
    except ImportError:
        scaffold_iter = scaffolds

    for scaffold in scaffold_iter:
        logger.info("  ── Scaffold %03d : %s", scaffold.index, scaffold.pdb_path.name)
        if hasattr(scaffold_iter, "set_postfix_str"):
            scaffold_iter.set_postfix_str(f"{scaffold.pdb_path.name} ({scaffold.n_residues} AA)")

        fixed_positions = fixed_by_scaffold.get(scaffold.name, [])

        # Exécution conditionnelle avec bascule transparente en cas d'erreur
        if selected_model == "ligandmpnn":
            try:
                seqs = _invoke_ligandmpnn(
                    scaffold        = scaffold,
                    output_dir      = output_dir,
                    num_seq         = num_seq,
                    sampling_temp   = sampling_temp,
                    model_weights   = ligandmpnn_weights,
                    use_cuda        = use_cuda,
                    seed            = seed,
                    logger          = logger,
                    fixed_positions = fixed_positions,
                    target_pdb      = target_pdb,
                    ligand_cutoff   = ligand_cutoff,
                )
            except Exception as exc:
                logger.warning(
                    "  [LIGANDMPNN FALLBACK] Échec LigandMPNN (%s). "
                    "Bascule automatique sur ProteinMPNN vanilla.", exc
                )
                seqs = _invoke_proteinmpnn(
                    scaffold        = scaffold,
                    output_dir      = output_dir,
                    num_seq         = num_seq,
                    sampling_temp   = sampling_temp,
                    backbone_noise  = backbone_noise,
                    model_weights   = model_weights,
                    use_cuda        = use_cuda,
                    seed            = seed,
                    logger          = logger,
                    fixed_positions = fixed_positions,
                )
        else:
            seqs = _invoke_proteinmpnn(
                scaffold        = scaffold,
                output_dir      = output_dir,
                num_seq         = num_seq,
                sampling_temp   = sampling_temp,
                backbone_noise  = backbone_noise,
                model_weights   = model_weights,
                use_cuda        = use_cuda,
                seed            = seed,
                logger          = logger,
                fixed_positions = fixed_positions,
            )

        for s in seqs:
            s.metadata["mpnn_model"] = selected_model
            logger.info(
                "       → seq_%02d  score=%.4f  len=%d AA  [%s]",
                s.index, s.mpnn_score, len(s.sequence), selected_model,
            )
        all_sequences.extend(seqs)

    return all_sequences


# ──────────────────────────────────────────────────────────────────────────────
# Analyse exacte des positions du motif (.trb + Géométrie 3D + Contigs)
# ──────────────────────────────────────────────────────────────────────────────

def motif_positions_from_contigs(
    contigs: str,
    n_residues: int,
    logger: logging.Logger | None = None,
    scaffold_pdb: Path | str | None = None,
    target_pdb: Path | str | None = None,
) -> list[int]:
    """
    Déduit les positions (1-indexées) du motif à conserver depuis les contigs
    RFdiffusion, y compris pour les motifs multi-boucles à longueurs variables !

    Ordre de résolution :
      1. Métadonnées exactes RFdiffusion ``.trb`` (``con_hal_idx0``) si disponibles.
      2. Calcul arithmétique direct si 0 ou 1 segment libre.
      3. Alignement géométrique 3D invariant par matrice de distances Cα-Cα entre
         ``target_pdb`` et ``scaffold_pdb`` contraint par les bornes des boucles.
    """
    # 1. Lecture directe du fichier .trb généré par RFdiffusion
    if scaffold_pdb:
        trb_path = Path(scaffold_pdb).with_suffix(".trb")
        if trb_path.is_file():
            try:
                import pickle
                with trb_path.open("rb") as fh:
                    trb = pickle.load(fh)
                if "con_hal_idx0" in trb and len(trb["con_hal_idx0"]) > 0:
                    return sorted(int(idx) + 1 for idx in trb["con_hal_idx0"])
            except Exception:
                pass

    if not contigs:
        return []

    # Si binder inter-chaînes pur ("A1-100/0 70-100"), la chaîne générée n'a pas de motif fixe interne
    if "/0 " in contigs or "/0," in contigs:
        after_break = contigs.split("/0", 1)[1].strip(" ,")
        if not re.search(r"[A-Za-z]", after_break):
            return []

    segs = [s for s in re.split(r"[,/\s]+", contigs.strip("[]'\" ")) if s and s != "0"]
    parsed: list[tuple[bool, int | None, tuple[str, int, int] | tuple[int, int]]] = []
    for s in segs:
        m = re.fullmatch(r"([A-Za-z])(\d+)(?:-(\d+))?", s)
        if m:
            ch = m.group(1).upper()
            a = int(m.group(2))
            b = int(m.group(3) or a)
            parsed.append((True, b - a + 1, (ch, a, b)))
            continue
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", s)
        if m:
            lo = int(m.group(1))
            hi = int(m.group(2) or lo)
            parsed.append((False, None, (lo, hi)))

    motif_total = sum(n for is_m, n, _ in parsed if is_m and n is not None)
    if motif_total == 0:
        return []

    n_free = sum(1 for is_m, _, _ in parsed if not is_m)
    if n_free <= 1:
        free_len = max(0, n_residues - motif_total) if n_free == 1 else 0
        positions: list[int] = []
        cursor = 0
        for is_motif, n, _ in parsed:
            if is_motif and n is not None:
                positions += list(range(cursor + 1, cursor + n + 1))
                cursor += n
            else:
                cursor += free_len
        return positions

    # 3. Résolution géométrique 3D exacte pour contigs multi-segments (ex: 20-27,A1,3-5,A2...)
    if scaffold_pdb and Path(scaffold_pdb).is_file():
        resolved_3d = _resolve_multi_segment_motif_3d(parsed, Path(scaffold_pdb), Path(target_pdb) if target_pdb else None)
        if resolved_3d:
            return resolved_3d

    # Fallback proportionnel déterministe sur la valeur médiane des intervalles libres
    free_bounds = [meta for is_m, _, meta in parsed if not is_m]
    rem_total = max(0, n_residues - motif_total)
    min_sum = sum(lo for lo, _ in free_bounds)  # type: ignore[misc]
    span_sum = sum(max(0, hi - lo) for lo, hi in free_bounds)  # type: ignore[misc]

    positions = []
    cursor = 0
    for is_motif, n, meta in parsed:
        if is_motif and n is not None:
            positions += list(range(cursor + 1, cursor + n + 1))
            cursor += n
        else:
            lo, hi = meta  # type: ignore[misc]
            if span_sum > 0 and rem_total >= min_sum:
                ratio = min(1.0, (rem_total - min_sum) / span_sum)
                alloc = lo + int(round(ratio * (hi - lo)))
            else:
                alloc = lo
            cursor += alloc
    return [p for p in positions if 1 <= p <= n_residues]


def _resolve_multi_segment_motif_3d(
    parsed_segs: list[Any],
    scaffold_pdb: Path,
    target_pdb: Path | None,
) -> list[int]:
    """
    Identifie exactement les indices 1-based des blocs du motif dans ``scaffold_pdb``
    en minimisant l'erreur de matrice de distances Cα-Cα par rapport à ``target_pdb``
    sous les contraintes topologiques exactes des intervalles de boucles.
    """
    try:
        import numpy as np

        def _extract_ca_map(p: Path) -> tuple[list[tuple[str, int, str, np.ndarray]], dict[tuple[str, int], tuple[str, np.ndarray]]]:
            ordered = []
            by_id = {}
            seen = set()
            with p.open("r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    if line.startswith("ATOM  ") and line[12:16].strip() == "CA":
                        ch = line[21].strip() or "A"
                        rseq = int(line[22:26].strip())
                        rname = line[17:20].strip().upper()
                        if (ch, rseq) in seen:
                            continue
                        seen.add((ch, rseq))
                        xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=np.float64)
                        ordered.append((ch, rseq, rname, xyz))
                        by_id[(ch, rseq)] = (rname, xyz)
            return ordered, by_id

        scaf_ordered, _ = _extract_ca_map(scaffold_pdb)
        if not scaf_ordered:
            return []
        L = len(scaf_ordered)
        scaf_xyz = np.array([x[3] for x in scaf_ordered], dtype=np.float64)

        tgt_by_id = {}
        if target_pdb and target_pdb.is_file():
            _, tgt_by_id = _extract_ca_map(target_pdb)

        # Construire la séquence des blocs motifs et des intervalles libres
        # Normaliser pour alterner : [free_0, motif_0, free_1, motif_1, ..., free_K]
        motif_blocks: list[list[tuple[str, int]]] = []
        free_intervals: list[tuple[int, int]] = []

        cur_free_lo, cur_free_hi = 0, 0
        for is_m, _, meta in parsed_segs:
            if not is_m:
                lo, hi = meta
                cur_free_lo += lo
                cur_free_hi += hi
            else:
                ch, a, b = meta
                if motif_blocks and cur_free_lo == 0 and cur_free_hi == 0:
                    motif_blocks[-1].extend((ch, r) for r in range(a, b + 1))
                else:
                    free_intervals.append((cur_free_lo, cur_free_hi))
                    motif_blocks.append([(ch, r) for r in range(a, b + 1)])
                cur_free_lo, cur_free_hi = 0, 0
        free_intervals.append((cur_free_lo, cur_free_hi))

        if not motif_blocks:
            return []

        # Coordonnées Cα de référence pour le premier résidu de chaque bloc motif
        ref_anchors = []
        ref_names = []
        has_ref_3d = True
        for blk in motif_blocks:
            ch, rseq = blk[0]
            if (ch, rseq) in tgt_by_id:
                rname, xyz = tgt_by_id[(ch, rseq)]
                ref_names.append(rname)
                ref_anchors.append(xyz)
            else:
                has_ref_3d = False
                break

        K = len(motif_blocks)
        block_lens = [len(b) for b in motif_blocks]

        # Recherche récursive bornée des positions de départ (0-indexées) de chaque bloc
        best_starts: list[int] | None = None
        best_err = float("inf")

        ref_dist_mat = None
        if has_ref_3d and len(ref_anchors) == K:
            ref_arr = np.array(ref_anchors, dtype=np.float64)
            ref_dist_mat = np.linalg.norm(ref_arr[:, None, :] - ref_arr[None, :, :], axis=-1)

        def _dfs(idx: int, prev_end: int, chosen: list[int]) -> None:
            nonlocal best_starts, best_err
            lo_g, hi_g = free_intervals[idx]
            min_start = prev_end + lo_g
            max_start = min(prev_end + hi_g, L - sum(block_lens[idx:]))

            for s_pos in range(min_start, max_start + 1):
                end_pos = s_pos + block_lens[idx]
                if idx == K - 1:
                    tail_lo, tail_hi = free_intervals[K]
                    rem = L - end_pos
                    if not (tail_lo <= rem <= tail_hi):
                        continue
                    cand = chosen + [s_pos]
                    err = 0.0
                    if ref_dist_mat is not None:
                        c_xyz = scaf_xyz[cand]
                        c_mat = np.linalg.norm(c_xyz[:, None, :] - c_xyz[None, :, :], axis=-1)
                        err = float(np.max(np.abs(c_mat - ref_dist_mat)))
                        # Bonus si les noms d'acides aminés correspondent
                        for m_i, st in enumerate(cand):
                            if scaf_ordered[st][2] != ref_names[m_i]:
                                err += 0.25
                    if err < best_err:
                        best_err = err
                        best_starts = cand
                else:
                    _dfs(idx + 1, end_pos, chosen + [s_pos])

        _dfs(0, 0, [])
        if best_starts is not None:
            out_pos: list[int] = []
            for b_i, st in enumerate(best_starts):
                out_pos.extend(range(st + 1, st + block_lens[b_i] + 1))
            return out_pos
    except Exception:
        pass
    return []


# ──────────────────────────────────────────────────────────────────────────────
# Invocation LigandMPNN (Context-Aware All-Atom Model)
# ──────────────────────────────────────────────────────────────────────────────

def _find_or_clone_ligandmpnn(logger: logging.Logger) -> Path:
    """Localise ou clone automatiquement le dépôt LigandMPNN."""
    candidates = [
        Path("LigandMPNN/run.py"),
        Path(__file__).resolve().parent.parent.parent / "LigandMPNN" / "run.py",
        Path("/content/Optobind-AI/LigandMPNN/run.py"),
        Path("/content/LigandMPNN/run.py"),
        Path("/tmp/optobind_project/LigandMPNN/run.py"),
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()

    logger.info("  [AUTO-SETUP] Clonage automatique de LigandMPNN...")
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/dauparas/LigandMPNN.git", "LigandMPNN"],
        check=False,
    )
    for p in candidates:
        if p.exists():
            return p.resolve()

    return Path("LigandMPNN/run.py")


def _resolve_ligandmpnn_weights(model_weights: str, ligand_script: Path, logger: logging.Logger) -> Path:
    """Résout ou télécharge automatiquement le fichier de poids LigandMPNN."""
    candidates = []
    if model_weights:
        candidates.append(Path(model_weights))
    candidates.extend([
        Path("weights/ligandmpnn_v_32_010_25.pt"),
        ligand_script.parent / "model_params" / "ligandmpnn_v_32_010_25.pt",
        Path("/content/drive/MyDrive/optobind_weights_cache/weights/ligandmpnn_v_32_010_25.pt"),
    ])
    for c in candidates:
        if c.exists() and c.stat().st_size > 5_000_000:
            return c.resolve()

    target = Path("weights/ligandmpnn_v_32_010_25.pt")
    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info("  [AUTO-SETUP] Téléchargement des poids LigandMPNN (~10 Mo)...")
    url = "https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_010_25.pt"
    try:
        import urllib.request
        urllib.request.urlretrieve(url, str(target))
        if target.exists() and target.stat().st_size > 5_000_000:
            return target.resolve()
    except Exception as exc:
        logger.warning("  [AUTO-SETUP] Échec téléchargement direct LigandMPNN : %s", exc)

    return target.resolve()


def _prepare_scaffold_for_ligandmpnn(
    scaffold   : Scaffold,
    target_pdb : Path | str | None,
    tmp_dir    : Path,
    logger     : logging.Logger,
) -> Path:
    """
    S'assure que le fichier PDB transmis à LigandMPNN contient à la fois
    les résidus protéiques du scaffold ET les atomes HETATM (ligand/métal).
    Si le scaffold est dépourvu de HETATM, ceux du target_pdb sont transférés.
    """
    scaffold_pdb = Path(scaffold.pdb_path)
    if not scaffold_pdb.exists():
        return scaffold_pdb

    scaffold_has_hetatm = False
    try:
        with open(scaffold_pdb, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("HETATM"):
                    scaffold_has_hetatm = True
                    break
    except Exception:
        pass

    if scaffold_has_hetatm or target_pdb is None or not Path(target_pdb).exists():
        return scaffold_pdb

    hetatm_lines: list[str] = []
    try:
        with open(target_pdb, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith(("HETATM", "LINK", "CONECT")):
                    hetatm_lines.append(line)
    except Exception as exc:
        logger.debug("  [LIGANDMPNN] Impossible d'extraire les HETATM de %s : %s", target_pdb, exc)
        return scaffold_pdb

    if not hetatm_lines:
        return scaffold_pdb

    complex_pdb = tmp_dir / f"{scaffold.name}_complex.pdb"
    with open(complex_pdb, "w", encoding="utf-8") as out_f:
        with open(scaffold_pdb, "r", encoding="utf-8", errors="ignore") as in_f:
            for line in in_f:
                if line.startswith("END"):
                    continue
                out_f.write(line)
        for h_line in hetatm_lines:
            out_f.write(h_line)
        out_f.write("END\n")

    logger.debug("  [LIGANDMPNN] Complexe enrichi généré : %s (+%d lignes HETATM)", complex_pdb.name, len(hetatm_lines))
    return complex_pdb


def _invoke_ligandmpnn_batch(
    scaffolds         : list[Scaffold],
    output_dir        : Path,
    num_seq           : int,
    sampling_temp     : float,
    model_weights     : str,
    use_cuda          : bool,
    seed              : int,
    logger            : logging.Logger,
    fixed_by_scaffold : dict[str, list[int]],
    target_pdb        : Path | str | None = None,
    ligand_cutoff     : float = 8.0,
    pack_side_chains  : bool = False,
) -> list[DesignedSequence]:
    """
    Exécute LigandMPNN en session GPU unique (``--pdb_path_multi`` + ``--fixed_residues_multi``)
    sur l'ensemble des squelettes simultanément, divisant le temps d'initialisation par N.
    """
    ligand_script = _find_or_clone_ligandmpnn(logger)
    weights_path = _resolve_ligandmpnn_weights(model_weights, ligand_script, logger)
    if not weights_path.exists() or weights_path.stat().st_size < 1_000_000:
        raise FileNotFoundError(f"Poids LigandMPNN introuvables : {weights_path}")

    batch_dir = output_dir / "_ligandmpnn_batch"
    batch_dir.mkdir(parents=True, exist_ok=True)

    pdb_multi: dict[str, str] = {}
    fixed_multi: dict[str, str] = {}
    stem_by_scaffold: dict[str, str] = {}

    for sc in scaffolds:
        input_pdb = _prepare_scaffold_for_ligandmpnn(sc, target_pdb, batch_dir, logger)
        pdb_key = str(input_pdb.resolve())
        pdb_multi[pdb_key] = ""
        stem_by_scaffold[sc.name] = input_pdb.stem

        f_pos = fixed_by_scaffold.get(sc.name, [])
        if f_pos:
            chain_id = "A"
            try:
                with input_pdb.open("r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if line.startswith("ATOM"):
                            chain_id = line[21].strip() or "A"
                            break
            except Exception:
                pass
            fixed_multi[pdb_key] = " ".join(f"{chain_id}{p}" for p in f_pos)
        else:
            fixed_multi[pdb_key] = ""

    pdb_json_path = batch_dir / "pdb_path_multi.json"
    fixed_json_path = batch_dir / "fixed_residues_multi.json"
    pdb_json_path.write_text(json.dumps(pdb_multi, indent=2), encoding="utf-8")
    fixed_json_path.write_text(json.dumps(fixed_multi, indent=2), encoding="utf-8")

    runner_script = (Path(__file__).resolve().parent.parent / "utils" / "runner.py").resolve()
    batch_size = min(num_seq, 8)
    number_of_batches = max(1, (num_seq + batch_size - 1) // batch_size)

    cmd = [
        sys.executable, str(runner_script), str(ligand_script),
        "--model_type",                   "ligand_mpnn",
        "--checkpoint_ligand_mpnn",       str(weights_path),
        "--pdb_path_multi",               str(pdb_json_path),
        "--fixed_residues_multi",         str(fixed_json_path),
        "--out_folder",                   str(batch_dir),
        "--temperature",                  str(sampling_temp),
        "--batch_size",                   str(batch_size),
        "--number_of_batches",            str(number_of_batches),
        "--seed",                         str(seed),
        "--ligand_mpnn_use_atom_context", "1",
        "--ligand_mpnn_cutoff_for_score", str(ligand_cutoff),
    ]

    if pack_side_chains:
        sc_weights = Path("weights/ligandmpnn_sc_v_32_002_16.pt")
        if sc_weights.is_file() and sc_weights.stat().st_size > 500_000:
            cmd += ["--pack_side_chains", "1", "--checkpoint_path_sc", str(sc_weights.resolve())]

    env = os.environ.copy()
    mpnn_root = ligand_script.parent
    current_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join([str(mpnn_root)] + ([current_pp] if current_pp else []))

    logger.info("  ⚡ [LIGANDMPNN BATCH] Session GPU unique lancée pour %d squelettes...", len(scaffolds))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"LigandMPNN batch code={result.returncode} : {result.stderr.strip()[:200]}")

    all_designed: list[DesignedSequence] = []
    seqs_dir = batch_dir / "seqs"
    for sc in scaffolds:
        stem = stem_by_scaffold[sc.name]
        fa_file = seqs_dir / f"{stem}.fa"
        if not fa_file.exists():
            fa_file = seqs_dir / f"{stem}.fasta"
        if not fa_file.exists():
            raise FileNotFoundError(f"Fichier FASTA LigandMPNN manquant pour {sc.name} ({fa_file})")

        seq_index = 0
        n_designed = 0
        for header, sequence in read_fasta(fa_file):
            is_native_header = (
                seq_index == 0
                and "id=" not in header
                and "overall_confidence=" not in header
                and "global_score=" not in header
                and "score=" not in header
            )
            score = _parse_global_score(header)
            fasta_out = output_dir / f"{sc.name}_seq_{seq_index:02d}.fasta"
            props = compute_physicochemical_properties(sequence)
            ds = DesignedSequence(
                index=seq_index,
                scaffold=sc,
                sequence=sequence,
                fasta_path=fasta_out,
                mpnn_score=score,
                pi=props["pi"],
                charge_ph7=props["charge_ph7"],
                gravy=props["gravy"],
                mw_kda=props["mw_kda"],
                metadata={
                    "mpnn_header": header,
                    "mpnn_model": "ligandmpnn",
                    "instability": props["instability"],
                },
            )
            fasta_out.write_text(ds.to_fasta(), encoding="utf-8")
            all_designed.append(ds)
            seq_index += 1
            if not is_native_header:
                n_designed += 1
            if n_designed >= num_seq:
                break

    return all_designed


def _invoke_ligandmpnn(
    scaffold        : Scaffold,
    output_dir      : Path,
    num_seq         : int,
    sampling_temp   : float,
    model_weights   : str,
    use_cuda        : bool,
    seed            : int,
    logger          : logging.Logger,
    fixed_positions : list[int] | None = None,
    target_pdb      : Path | str | None = None,
    ligand_cutoff   : float = 8.0,
) -> list[DesignedSequence]:
    """Exécute LigandMPNN en mode atom-context-aware et parse les séquences."""
    ligand_script = _find_or_clone_ligandmpnn(logger)
    weights_path = _resolve_ligandmpnn_weights(model_weights, ligand_script, logger)

    if not weights_path.exists() or weights_path.stat().st_size < 1_000_000:
        raise FileNotFoundError(f"Poids LigandMPNN introuvables : {weights_path}")

    tmp_dir = output_dir / scaffold.name
    tmp_dir.mkdir(parents=True, exist_ok=True)

    input_pdb = _prepare_scaffold_for_ligandmpnn(scaffold, target_pdb, tmp_dir, logger)
    runner_script = (Path(__file__).resolve().parent.parent / "utils" / "runner.py").resolve()

    batch_size = min(num_seq, 8)
    number_of_batches = max(1, (num_seq + batch_size - 1) // batch_size)

    cmd = [
        sys.executable, str(runner_script), str(ligand_script),
        "--model_type",                   "ligand_mpnn",
        "--checkpoint_ligand_mpnn",       str(weights_path),
        "--pdb_path",                     str(input_pdb),
        "--out_folder",                   str(tmp_dir),
        "--temperature",                  str(sampling_temp),
        "--batch_size",                   str(batch_size),
        "--number_of_batches",            str(number_of_batches),
        "--seed",                         str(seed),
        "--ligand_mpnn_use_atom_context", "1",
        "--ligand_mpnn_cutoff_for_score", str(ligand_cutoff),
    ]

    if fixed_positions:
        chain_id = "A"
        try:
            with open(input_pdb, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.startswith("ATOM"):
                        chain_id = line[21].strip() or "A"
                        break
        except Exception:
            pass
        fixed_str = " ".join(f"{chain_id}{p}" for p in fixed_positions)
        cmd += ["--fixed_residues", fixed_str]
        logger.info("  ▸ Résidus verrouillés (LigandMPNN) : %s", fixed_str)

    env = os.environ.copy()
    mpnn_root = ligand_script.parent
    current_pp = env.get("PYTHONPATH", "")
    paths = [str(mpnn_root)]
    if current_pp:
        paths.append(current_pp)
    env["PYTHONPATH"] = os.pathsep.join(paths)

    logger.debug("  [LIGANDMPNN CMD] %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)

    if result.returncode != 0:
        logger.warning("[LigandMPNN] STDERR:\n%s", textwrap.indent(result.stderr, "    "))
        raise RuntimeError(f"LigandMPNN code={result.returncode} : {result.stderr.strip()[:200]}")

    return _collect_sequences(scaffold, tmp_dir, num_seq, output_dir)


# ──────────────────────────────────────────────────────────────────────────────
# Invocation ProteinMPNN Vanilla (Backbone-Only Model)
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
    subprocess.run(["git", "clone", "https://github.com/dauparas/ProteinMPNN.git"], check=False)
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
    fixed_positions: list[int] | None = None,
) -> list[DesignedSequence]:
    """Lance ProteinMPNN Vanilla via subprocess et parse les FASTA de sortie."""
    mpnn_script, _ = _find_or_clone_proteinmpnn(logger)

    tmp_dir = output_dir / scaffold.name
    tmp_dir.mkdir(parents=True, exist_ok=True)

    runner_script = (Path(__file__).resolve().parent.parent / "utils" / "runner.py").resolve()

    cmd = [
        sys.executable, str(runner_script), str(mpnn_script),
        "--pdb_path",            str(scaffold.pdb_path),
        "--out_folder",          str(tmp_dir),
        "--num_seq_per_target",  str(num_seq),
        "--sampling_temp",       str(sampling_temp),
        "--backbone_noise",      str(backbone_noise),
        "--batch_size",          str(min(num_seq, 8)),
        "--seed",                str(seed),
    ]

    if fixed_positions:
        chain_id = "A"
        for line in Path(scaffold.pdb_path).read_text(errors="ignore").splitlines():
            if line.startswith("ATOM"):
                chain_id = line[21].strip() or "A"
                break
        fixed_jsonl = tmp_dir / "fixed_positions.jsonl"
        fixed_jsonl.write_text(
            json.dumps({Path(scaffold.pdb_path).stem: {chain_id: fixed_positions}}) + "\n",
            encoding="utf-8",
        )
        cmd += ["--fixed_positions_jsonl", str(fixed_jsonl)]
        logger.info("  ▸ Résidus du motif verrouillés (ProteinMPNN, chaîne %s) : %s", chain_id, fixed_positions)

    weights_dir = None
    model_name = "v_48_020"

    if model_weights:
        w_path = Path(model_weights)
        if w_path.is_file():
            weights_dir = w_path.parent
            for m_cand in ("v_48_002", "v_48_010", "v_48_020", "v_48_030"):
                if m_cand in w_path.name:
                    model_name = m_cand
                    break
            target_file = weights_dir / f"{model_name}.pt"
            if not target_file.exists():
                try:
                    shutil.copyfile(w_path, target_file)
                except Exception:
                    pass
        elif w_path.is_dir():
            weights_dir = w_path

    mpnn_root = mpnn_script.parent
    vanilla_dir = mpnn_root / "vanilla_model_weights"
    if weights_dir is None or not (weights_dir / f"{model_name}.pt").exists():
        if (vanilla_dir / f"{model_name}.pt").exists():
            weights_dir = vanilla_dir
        elif (Path("weights") / f"{model_name}.pt").exists():
            weights_dir = Path("weights")
        elif (Path("weights") / f"ProteinMPNN_{model_name}.pt").exists():
            shutil.copyfile(Path("weights") / f"ProteinMPNN_{model_name}.pt", Path("weights") / f"{model_name}.pt")
            weights_dir = Path("weights")

    if weights_dir is not None:
        cmd += ["--path_to_model_weights", str(weights_dir), "--model_name", model_name]

    env = os.environ.copy()
    current_pp = env.get("PYTHONPATH", "")
    paths = [str(mpnn_root)]
    if current_pp:
        paths.append(current_pp)
    env["PYTHONPATH"] = os.pathsep.join(paths)

    result = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)

    if result.returncode != 0:
        logger.error("[ProteinMPNN] STDERR:\n%s", textwrap.indent(result.stderr, "    "))
        raise RuntimeError(f"ProteinMPNN a échoué pour {scaffold.name} (code={result.returncode}).")

    return _collect_sequences(scaffold, tmp_dir, num_seq, output_dir)


# ──────────────────────────────────────────────────────────────────────────────
# Collecte et parsing des séquences FASTA générées
# ──────────────────────────────────────────────────────────────────────────────

def _collect_sequences(
    scaffold   : Scaffold,
    tmp_dir    : Path,
    num_seq    : int,
    output_dir : Path,
) -> list[DesignedSequence]:
    """
    Parcourt les fichiers FASTA générés par ProteinMPNN ou LigandMPNN
    et construit des objets DesignedSequence complets.
    """
    seqs_dir = tmp_dir / "seqs"
    fasta_files = list(seqs_dir.glob("*.fa")) + list(seqs_dir.glob("*.fasta"))

    designed: list[DesignedSequence] = []
    seq_index = 0
    n_designed = 0

    for fasta_file in fasta_files:
        for header, sequence in read_fasta(fasta_file):
            is_native_header = (
                seq_index == 0
                and "id=" not in header
                and "overall_confidence=" not in header
                and "global_score=" not in header
                and "score=" not in header
            )
            score = _parse_global_score(header)
            fasta_out = output_dir / f"{scaffold.name}_seq_{seq_index:02d}.fasta"
            props = compute_physicochemical_properties(sequence)
            ds = DesignedSequence(
                index      = seq_index,
                scaffold   = scaffold,
                sequence   = sequence,
                fasta_path = fasta_out,
                mpnn_score = score,
                pi         = props["pi"],
                charge_ph7 = props["charge_ph7"],
                gravy      = props["gravy"],
                mw_kda     = props["mw_kda"],
                metadata   = {"mpnn_header": header, "instability": props["instability"]},
            )
            fasta_out.write_text(ds.to_fasta(), encoding="utf-8")
            designed.append(ds)
            seq_index += 1
            if not is_native_header:
                n_designed += 1
            if n_designed >= num_seq:
                break
        if n_designed >= num_seq:
            break

    return designed


def _parse_global_score(header: str) -> float:
    """
    Extrait le score de confiance global depuis le header FASTA
    compatible avec ProteinMPNN (global_score=X, score=X)
    et LigandMPNN (overall_confidence=X, ligand_confidence=X).
    """
    for part in header.split(","):
        part = part.strip()
        for prefix in ("global_score=", "overall_confidence=", "score=", "ligand_confidence="):
            if part.startswith(prefix):
                try:
                    return float(part.split("=")[1].strip())
                except (ValueError, IndexError):
                    pass
    return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Mode dry_run (tests unitaires sans GPU)
# ──────────────────────────────────────────────────────────────────────────────

_AA_POOL = "ACDEFGHIKLMNPQRSTVWY"
_AROMATIC = "FYWH"


def _dry_run_sequences(
    scaffold   : Scaffold,
    num_seq    : int,
    output_dir : Path,
) -> list[DesignedSequence]:
    """Génère des séquences pseudo-aléatoires enrichies pour les tests sans GPU."""
    import random

    n_res = max(scaffold.n_residues, 70)
    designed: list[DesignedSequence] = []

    for i in range(num_seq):
        seq_chars = []
        for pos in range(n_res):
            if 30 <= pos <= 50 and random.random() < 0.20:
                seq_chars.append(random.choice(_AROMATIC))
            else:
                seq_chars.append(random.choice(_AA_POOL))

        sequence = "".join(seq_chars)
        score = round(random.uniform(0.3, 1.5), 4)
        fasta_out = output_dir / f"{scaffold.name}_seq_{i:02d}.fasta"
        props = compute_physicochemical_properties(sequence)

        ds = DesignedSequence(
            index      = i,
            scaffold   = scaffold,
            sequence   = sequence,
            fasta_path = fasta_out,
            mpnn_score = score,
            pi         = props["pi"],
            charge_ph7 = props["charge_ph7"],
            gravy      = props["gravy"],
            mw_kda     = props["mw_kda"],
            metadata   = {"dry_run": True, "instability": props["instability"]},
        )
        fasta_out.write_text(ds.to_fasta(), encoding="utf-8")
        designed.append(ds)

    return designed
