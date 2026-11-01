"""
optobind_ai.modules.biophysics
==============================
Calcul des propriétés biophysiques et prédiction de la solubilité des variants.

Ce module combine :
  1. Le calcul analytique local (0 ms, 100% hors-ligne via Biopython ProtParam) :
     - Point isoélectrique (pI)
     - Charge nette à pH 7.4
     - Indice d'hydropathie globale (GRAVY)
     - Poids moléculaire (kDa) et indice d'instabilité
  2. L'interrogation sécurisée de l'API externe Protein-Sol (Université de Manchester)
     réservée uniquement aux meilleurs candidats finaux (Top 3).
     Cette requête est totalement non-bloquante : en cas de timeout réseau ou
     de panne du serveur distant, le pipeline continue sans interruption.
"""

from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import Any

from Bio.SeqUtils.ProtParam import ProteinAnalysis

log = logging.getLogger("optobind")

_STANDARD_AAS = set("ACDEFGHIKLMNPQRSTVWY")


def compute_physicochemical_properties(sequence: str) -> dict[str, float]:
    """
    Calcule les constantes physico-chimiques d'une séquence protéique en local.

    Parameters
    ----------
    sequence : Séquence d'acides aminés (chaîne brute).

    Returns
    -------
    dict avec 'pi', 'charge_ph7', 'gravy', 'mw_kda', 'instability'.
    """
    clean_seq = "".join(aa for aa in sequence.upper() if aa in _STANDARD_AAS)
    if not clean_seq:
        return {
            "pi": 7.0,
            "charge_ph7": 0.0,
            "gravy": 0.0,
            "mw_kda": 0.0,
            "instability": 0.0,
        }

    try:
        pa = ProteinAnalysis(clean_seq)
        return {
            "pi": round(float(pa.isoelectric_point()), 2),
            "charge_ph7": round(float(pa.charge_at_pH(7.4)), 2),
            "gravy": round(float(pa.gravy()), 3),
            "mw_kda": round(float(pa.molecular_weight()) / 1000.0, 2),
            "instability": round(float(pa.instability_index()), 1),
        }
    except Exception as exc:
        log.debug("[BIOPHYSICS] Erreur calcul ProtParam sur séquence : %s", exc)
        return {
            "pi": 7.0,
            "charge_ph7": 0.0,
            "gravy": 0.0,
            "mw_kda": 0.0,
            "instability": 0.0,
        }


def analyze_sequence_liabilities(sequence: str) -> dict[str, Any]:
    """
    Analyse les motifs de vulnérabilité chimique et d'instabilité de séquence (Sequence Liabilities) :
      - Déamidation spontanée : 'NG', 'NS'
      - Isomérisation de l'aspartate : 'DG', 'DS'
      - Clivage acide spontané : 'DP'
      - Motifs de N-glycosylation non désirés : 'N[^P][ST]'
      - Répétitions de faible complexité : >= 4 résidus identiques consécutifs

    Returns
    -------
    dict avec 'n_liabilities', 'liabilities', 'details'.
    """
    clean_seq = "".join(aa for aa in sequence.upper() if aa in _STANDARD_AAS)
    liabilities: list[str] = []

    # 1. Déamidation spontanée
    for match in re.finditer(r"N[GS]", clean_seq):
        pos = match.start() + 1
        motif = match.group()
        liabilities.append(f"{motif}(pos:{pos}:deamidation)")

    # 2. Isomérisation
    for match in re.finditer(r"D[GS]", clean_seq):
        pos = match.start() + 1
        motif = match.group()
        liabilities.append(f"{motif}(pos:{pos}:isomerization)")

    # 3. Clivage acide
    for match in re.finditer(r"DP", clean_seq):
        pos = match.start() + 1
        liabilities.append(f"DP(pos:{pos}:acid_cleavage)")

    # 4. N-glycosylation consensus N-X-S/T (où X != P)
    for match in re.finditer(r"N[^P][ST]", clean_seq):
        pos = match.start() + 1
        motif = match.group()
        liabilities.append(f"{motif}(pos:{pos}:n_glyco)")

    # 5. Homopolymères consécutifs (>= 4 du même acide aminé)
    for match in re.finditer(r"([A-Z])\1{3,}", clean_seq):
        pos = match.start() + 1
        motif = match.group()
        liabilities.append(f"{motif}(pos:{pos}:poly_{motif[0]})")

    details = ", ".join(liabilities) if liabilities else "-"
    return {
        "n_liabilities": len(liabilities),
        "liabilities": liabilities,
        "details": details,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Smart Pre-Filter Universel (CPU, 0 ms) avant inférence 3D GPU
# ──────────────────────────────────────────────────────────────────────────────

def smart_pre_filter_sequences(
    sequences  : list[Any],
    target_pdb : Path | str,
    cfg        : dict[str, Any],
    logger     : logging.Logger | None = None,
) -> tuple[list[Any], list[tuple[Any, str]]]:
    """
    Pré-filtrage analytique universel (CPU, 0 ms) pour qualifier uniquement
    les séquences d'élite vers l'inférence 3D Chai-1 / AlphaFold.

    Critères universels :
      1. Sequence Liabilities (max_liabilities = 0)
      2. Parité et compatibilité universelle des Cystéines (avec ou sans ion métallique)
      3. Charge nette et pI (évitement du pI neutre et charge nulle)
      4. Score ProteinMPNN (max_mpnn_score)
      5. Quota Top-K par squelette (diversité des scaffolds)

    Returns
    -------
    (passed_sequences, rejected_sequences_with_reasons)
    """
    _log = logger or log
    pref_cfg = cfg.get("pre_filter", {})
    if not pref_cfg.get("enabled", True):
        _log.info("[PRE-FILTER] Pré-filtrage désactivé via configuration.")
        return sequences, []

    max_liab        = int(pref_cfg.get("max_liabilities", 0))
    filter_cys      = bool(pref_cfg.get("filter_cysteines", True))
    min_charge      = float(pref_cfg.get("min_net_charge", 0.8))
    avoid_pi        = bool(pref_cfg.get("avoid_neutral_pi", True))
    max_mpnn        = pref_cfg.get("max_mpnn_score", 1.40)
    max_mpnn        = float(max_mpnn) if max_mpnn is not None else None
    top_k           = int(pref_cfg.get("top_k_per_scaffold", 3))
    filter_native   = bool(pref_cfg.get("filter_native", True))

    # Détection automatique de présence d'un ion métallique dans la cible
    has_metal = False
    tgt_p = Path(target_pdb) if target_pdb else None
    if tgt_p and tgt_p.is_file():
        metal_symbols = {"ZN", "FE", "CU", "MG", "MN", "NI", "CO", "CD"}
        std_aas = {
            "ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU",
            "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR"
        }
        try:
            with tgt_p.open("r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    if line.startswith(("ATOM", "HETATM")):
                        rname = line[17:20].strip().upper()
                        if rname not in std_aas:
                            aname = line[12:16].strip().upper()
                            elem = line[76:78].strip().upper() if len(line) >= 78 else aname[:2]
                            if elem in metal_symbols or any(aname.startswith(m) for m in metal_symbols):
                                has_metal = True
                                break
        except Exception as exc:
            _log.debug("[PRE-FILTER] Erreur lecture métaux cible : %s", exc)

    survivors: list[Any] = []
    rejected: list[tuple[Any, str]] = []
    stats = {
        "liabilities": 0,
        "cysteines": 0,
        "precipitation": 0,
        "mpnn_score": 0,
        "native": 0,
        "quota": 0,
    }

    for seq in sequences:
        # Exclure seq_00 si demandé (squelette brut non optimisé)
        if filter_native and getattr(seq, "index", -1) == 0:
            rejected.append((seq, "Native seq_00 (non optimisée)"))
            stats["native"] += 1
            continue

        raw_seq = getattr(seq, "sequence", "")
        mpnn_sc = getattr(seq, "mpnn_score", 0.0)

        # 1. Pièges chimiques de séquence
        liab_info = analyze_sequence_liabilities(raw_seq)
        if liab_info["n_liabilities"] > max_liab:
            rejected.append((seq, f"Liabilities: {liab_info['details']}"))
            stats["liabilities"] += 1
            continue

        # 2. Règle universelle des cystéines
        if filter_cys:
            c_count = raw_seq.count("C")
            if not has_metal:
                # Cible sans métal : toute Cys doit pouvoir s'apparier en pont disulfure
                if c_count % 2 != 0:
                    rejected.append((seq, f"Cys impaire ({c_count}) sans ion métallique (orpheline garantie)"))
                    stats["cysteines"] += 1
                    continue
            else:
                # Cible métallique : la sphère de coordination admet 2 à 4 Cys
                if c_count < 2 or c_count > 4:
                    rejected.append((seq, f"Cys={c_count} incompatible avec coordination métallique [2-4]"))
                    stats["cysteines"] += 1
                    continue

        # 3. Risque de précipitation (pI neutre + charge quasi-nulle)
        if avoid_pi:
            phys = compute_physicochemical_properties(raw_seq)
            if abs(phys["charge_ph7"]) < min_charge and (7.1 <= phys["pi"] <= 7.7):
                rejected.append((seq, f"Risque précipitation (pI={phys['pi']:.2f}, q={phys['charge_ph7']:.2f})"))
                stats["precipitation"] += 1
                continue

        # 4. Score ProteinMPNN
        if max_mpnn is not None and mpnn_sc > max_mpnn:
            rejected.append((seq, f"Score MPNN ({mpnn_sc:.2f}) > {max_mpnn:.2f}"))
            stats["mpnn_score"] += 1
            continue

        survivors.append(seq)

    # 5. Quota Top-K par squelette (pour garantir la diversité)
    final_qualified: list[Any] = []
    if top_k > 0 and survivors:
        by_scaffold: dict[str, list[Any]] = {}
        for s in survivors:
            scaf_id = getattr(getattr(s, "scaffold", None), "name", "scaffold")
            by_scaffold.setdefault(scaf_id, []).append(s)

        for scaf_id, scaf_seqs in by_scaffold.items():
            scaf_seqs.sort(key=lambda x: getattr(x, "mpnn_score", 999.0))
            kept = scaf_seqs[:top_k]
            dropped = scaf_seqs[top_k:]
            final_qualified.extend(kept)
            for d in dropped:
                rejected.append((d, f"Dépassement quota Top {top_k} du scaffold"))
                stats["quota"] += 1
    else:
        final_qualified = survivors

    n_total = len(sequences)
    n_kept = len(final_qualified)
    n_rej = len(rejected)
    gpu_saving = round(100.0 * (1.0 - n_kept / n_total), 1) if n_total > 0 else 0.0

    _log.info("  🎯 Smart Pre-Filter Universel (CPU) : %d séquences → %d qualifiées pour Chai-1 (%d éliminées)",
              n_total, n_kept, n_rej)
    _log.info("     ↳ Rejets : %d Liabilities | %d Cys incompatibles | %d Précipitation | %d Score MPNN | %d Native | %d Quota Top-%d",
              stats["liabilities"], stats["cysteines"], stats["precipitation"], stats["mpnn_score"], stats["native"], stats["quota"], top_k)
    _log.info("     ↳ Économie GPU estimée : ~%.1f%% de temps de calcul économisé", gpu_saving)

    return final_qualified, rejected


def query_protein_sol(
    sequence: str,
    timeout: float = 8.0,
    logger: logging.Logger | None = None,
) -> dict[str, float] | None:
    """
    Interroge le serveur Protein-Sol (Université de Manchester) pour obtenir
    le score prédit d'expression soluble dans E. coli.

    Garanti non-bloquant : capture toute exception réseau ou de parsing et
    renvoie None pour ne jamais faire échouer le pipeline.

    Parameters
    ----------
    sequence : Séquence d'acides aminés.
    timeout  : Délai maximal d'attente en secondes (défaut: 8s).
    logger   : Logger partagé.

    Returns
    -------
    dict avec 'scaled_sol' et 'percent_sol' si succès, sinon None.
    """
    _log = logger or log
    clean_seq = "".join(aa for aa in sequence.upper() if aa in _STANDARD_AAS)
    if not clean_seq:
        return None

    try:
        import requests
        url = "https://protein-sol.manchester.ac.uk/cgi-bin/solubility/sequenceprediction.php"
        resp = requests.post(url, data={"sequence-input": clean_seq}, timeout=timeout)
        if resp.status_code != 200:
            _log.debug("[SOLUBILITY] Protein-Sol HTTP %s", resp.status_code)
            return None

        # Extraction de l'identifiant timestamp généré par le serveur
        match = re.search(r'var timestamp = "([a-f0-9]+)";', resp.text)
        if not match:
            _log.debug("[SOLUBILITY] Identifiant de résultat introuvable dans la réponse.")
            return None

        ts = match.group(1)
        res_url = f"https://protein-sol.manchester.ac.uk/results/solubility/run-{ts}/seq_prediction.txt"
        res_resp = requests.get(res_url, timeout=timeout)
        if res_resp.status_code != 200:
            return None

        for line in res_resp.text.splitlines():
            if line.startswith("SEQUENCE PREDICTIONS"):
                # Format: SEQUENCE PREDICTIONS,> Protein-sol,86.827, 0.756, 0.446, 7.890
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    percent_sol = float(parts[2])
                    scaled_sol = float(parts[3])
                    return {
                        "scaled_sol": round(scaled_sol, 3),
                        "percent_sol": round(percent_sol, 1),
                    }
        return None

    except Exception as exc:
        _log.warning(
            "  ⚠️ [Solubilité] Serveur externe Protein-Sol injoignable ou délai dépassé (%s) — étape ignorée en toute sécurité.",
            exc.__class__.__name__,
        )
        return None


def annotate_top_candidates_solubility(
    validation_results: list[Any],
    top_n: int = 3,
    logger: logging.Logger | None = None,
) -> None:
    """
    Sélectionne les top_n meilleurs candidats validés et les annote avec le score Protein-Sol.

    Parameters
    ----------
    validation_results : Liste des objets ValidationResult.
    top_n              : Nombre de candidats finaux à tester (défaut: 3).
    logger             : Logger partagé.
    """
    _log = logger or log
    if not validation_results or top_n <= 0:
        return

    # Tri : candidats 'passed' en priorité, puis pLDDT décroissant, puis RMSD croissant
    ranked = sorted(
        validation_results,
        key=lambda v: (
            1 if v.passed else 0,
            v.plddt_mean,
            -v.rmsd if v.rmsd != float("inf") else -999.0,
        ),
        reverse=True,
    )

    top_candidates = ranked[:top_n]
    _log.info(
        "  ▸ Analyse de solubilité Protein-Sol sur les %d meilleurs candidats :",
        len(top_candidates),
    )

    for cand in top_candidates:
        if cand.solubility_score is not None:
            continue
        res = query_protein_sol(cand.sequence.sequence, timeout=8.0, logger=_log)
        if res:
            cand.solubility_score = res["scaled_sol"]
            cand.solubility_percent = res["percent_sol"]
            status_sol = "SOLUBLE" if res["scaled_sol"] > 0.45 else "RISQUE D'AGREGATION"
            _log.info(
                "    ✓ %s : score=%.3f (seuil > 0.45) | solubilité=%.1f%% → %s",
                cand.sequence.name,
                res["scaled_sol"],
                res["percent_sol"],
                status_sol,
            )
        else:
            cand.solubility_score = None
            cand.solubility_percent = None


# ──────────────────────────────────────────────────────────────────────────────
# Analyse des cystéines orphelines (Unpaired / Free Cysteines)
# ──────────────────────────────────────────────────────────────────────────────

def analyze_cysteines(
    pdb_path: Path | str | None,
    metal_cutoff: float = 3.2,
    disulfide_cutoff: float = 3.0,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """
    Analyse les résidus CYS dans la structure 3D pour détecter les cystéines orphelines.

    Une cystéine est considérée comme 'sécurisée' si :
      1. Elle coordonne un ion métallique (SG à <= metal_cutoff Å d'un ion Zn²⁺, Fe, Cu...).
      2. Elle forme un pont disulfure structural (SG à <= disulfide_cutoff Å d'un autre CYS).
    
    Toute autre cystéine est classée comme 'orpheline' (unpaired), avec indication
    de son exposition au solvant (SASA) :
      - Exposée (SASA >= 10 Å²) : risque maximal d'oxydation et d'agrégation intermoléculaire.
      - Enfouie (SASA < 10 Å²) : enfouie dans le cœur, risque moindre mais déstabilisant.

    Parameters
    ----------
    pdb_path         : Chemin vers le fichier PDB ou mmCIF.
    metal_cutoff     : Distance max SG - métal pour la coordination (Å, défaut 3.2).
    disulfide_cutoff : Distance max SG - SG pour un pont disulfure (Å, défaut 3.0).
    logger           : Logger partagé.

    Returns
    -------
    dict avec 'total_cys', 'metal_coord', 'disulfides', 'unpaired_cys', 'n_unpaired', 'details'.
    """
    _log = logger or log
    default_result: dict[str, Any] = {
        "total_cys": 0,
        "metal_coord": [],
        "disulfides": [],
        "unpaired_cys": [],
        "n_unpaired": 0,
        "details": "-",
    }

    if not pdb_path:
        return default_result

    path = Path(pdb_path)
    if not path.is_file():
        return default_result

    try:
        from Bio.PDB import PDBParser, MMCIFParser
        from Bio.PDB.SASA import ShrakeRupley

        is_cif = path.suffix.lower() in [".cif", ".mmcif"]
        parser = MMCIFParser(QUIET=True) if is_cif else PDBParser(QUIET=True)
        structure = parser.get_structure("pred_struct", str(path))

        # Calcul SASA au niveau résidu
        try:
            sr = ShrakeRupley()
            sr.compute(structure, level="R")
        except Exception:
            pass

        # 1. Extraction des atomes métalliques (hors acides aminés standards)
        std_aa = {
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
            "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
        }
        metal_elements = {"ZN", "FE", "CU", "MG", "MN", "NI", "CO", "CD", "CA"}
        metal_atoms = []
        for atom in structure.get_atoms():
            if atom.parent.resname.upper() in std_aa:
                continue
            elem = atom.element.strip().upper() if atom.element else ""
            name = atom.name.strip().upper()
            if elem in metal_elements or any(name.startswith(m) for m in ["ZN", "FE", "CU", "MG", "MN", "NI", "CO", "CD"]):
                metal_atoms.append(atom)

        # 2. Identification des résidus CYS
        cys_residues = [r for r in structure.get_residues() if r.resname.upper() == "CYS"]
        if not cys_residues:
            return default_result

        # 3. Détection des Cys coordinant un métal
        metal_coord: set[int] = set()
        for r in cys_residues:
            if "SG" in r:
                for m in metal_atoms:
                    if (r["SG"] - m) <= metal_cutoff:
                        metal_coord.add(r.id[1])
                        break

        # 4. Détection des ponts disulfures entre Cys restantes
        paired_cys = set(metal_coord)
        disulfides: list[tuple[int, int]] = []
        remaining = [r for r in cys_residues if r.id[1] not in paired_cys and "SG" in r]

        for i in range(len(remaining)):
            r1 = remaining[i]
            if r1.id[1] in paired_cys:
                continue
            for j in range(i + 1, len(remaining)):
                r2 = remaining[j]
                if r2.id[1] in paired_cys:
                    continue
                if (r1["SG"] - r2["SG"]) <= disulfide_cutoff:
                    disulfides.append((r1.id[1], r2.id[1]))
                    paired_cys.add(r1.id[1])
                    paired_cys.add(r2.id[1])
                    break

        # 5. Détection des Cys orphelines (libres)
        unpaired: list[int] = [r.id[1] for r in cys_residues if r.id[1] not in paired_cys]

        details_list = []
        for r in cys_residues:
            if r.id[1] in unpaired:
                sasa = getattr(r, "sasa", 0.0)
                tag = "exp" if sasa >= 10.0 else "bur"
                details_list.append(f"C{r.id[1]}({tag}:{sasa:.0f}Å²)")

        details_str = ", ".join(details_list) if details_list else "-"

        return {
            "total_cys": len(cys_residues),
            "metal_coord": sorted(metal_coord),
            "disulfides": disulfides,
            "unpaired_cys": unpaired,
            "n_unpaired": len(unpaired),
            "details": details_str,
        }

    except Exception as exc:
        _log.debug("[BIOPHYSICS] Erreur lors de l'analyse des cystéines : %s", exc)
        return default_result


# ──────────────────────────────────────────────────────────────────────────────
# Analyse des interactions à l'interface (Liaisons H et Ponts Salins)
# ──────────────────────────────────────────────────────────────────────────────

def analyze_interface_interactions(
    pdb_path: Path | str | None,
    hbond_cutoff: float = 3.5,
    salt_bridge_cutoff: float = 4.0,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """
    Détecte et quantifie les liaisons hydrogène (H-bonds) et les ponts salins (salt bridges)
    formés à l'interface entre la protéine de novo (Chaîne principale) et ses partenaires
    (cible, récepteur ou ligand/fluorophore).

    Critères géométriques :
      - Liaisons hydrogène : Atomes électronégatifs donneur/accepteur (N, O, S) séparés par <= 3.5 Å.
      - Ponts salins       : Oxygène anionique (Asp/Glu) et azote cationique (Lys/Arg/His) séparés par <= 4.0 Å.

    Parameters
    ----------
    pdb_path           : Chemin du fichier PDB/CIF du complexe.
    hbond_cutoff       : Distance maximale pour liaison H (Å, défaut 3.5).
    salt_bridge_cutoff : Distance maximale pour pont salin (Å, défaut 4.0).
    logger             : Logger partagé.

    Returns
    -------
    dict avec 'n_hbonds', 'n_salt_bridges', 'hbonds_details', 'salt_bridges_details'.
    """
    _log = logger or log
    default_result: dict[str, Any] = {
        "n_hbonds": 0,
        "n_salt_bridges": 0,
        "hbonds_details": [],
        "salt_bridges_details": [],
    }

    if not pdb_path:
        return default_result

    path = Path(pdb_path)
    if not path.is_file():
        return default_result

    try:
        from Bio.PDB import PDBParser, MMCIFParser
        import numpy as np

        is_cif = path.suffix.lower() in [".cif", ".mmcif"]
        parser = MMCIFParser(QUIET=True) if is_cif else PDBParser(QUIET=True)
        structure = parser.get_structure("complex", str(path))
        if len(structure) == 0:
            return default_result

        model = structure[0]
        chains = list(model.get_chains())
        if len(chains) <= 1:
            return default_result

        # Chaîne A = binder / chaîne principale ; autres chaînes = cible, ligand, cofacteur
        prot_chain = chains[0]
        other_chains = chains[1:]

        prot_atoms = list(prot_chain.get_atoms())
        other_atoms = [a for ch in other_chains for a in ch.get_atoms()]

        # 1. Liaisons hydrogène à l'interface (Donneurs/Accepteurs N, O, S)
        prot_polar = [a for a in prot_atoms if a.element in {"N", "O", "S"}]
        other_polar = [a for a in other_atoms if a.element in {"N", "O", "S"}]

        hbonds = []
        for pa in prot_polar:
            for oa in other_polar:
                d = float(np.linalg.norm(pa.coord - oa.coord))
                if d <= hbond_cutoff:
                    pr = pa.parent
                    opr = oa.parent
                    hbonds.append(f"{pr.resname}{pr.id[1]}:{pa.name}--{opr.resname}{opr.id[1]}:{oa.name} ({d:.1f}Å)")

        # 2. Ponts salins à l'interface (Anion: Asp/Glu, Cation: Lys/Arg/His)
        anion_p = [a for a in prot_atoms if (a.parent.resname == "ASP" and a.name in ["OD1", "OD2"]) or (a.parent.resname == "GLU" and a.name in ["OE1", "OE2"])]
        cation_p = [a for a in prot_atoms if (a.parent.resname == "LYS" and a.name == "NZ") or (a.parent.resname == "ARG" and a.name in ["NH1", "NH2", "NE"]) or (a.parent.resname == "HIS" and a.name in ["ND1", "NE2"])]

        anion_o = [a for a in other_atoms if (a.parent.resname == "ASP" and a.name in ["OD1", "OD2"]) or (a.parent.resname == "GLU" and a.name in ["OE1", "OE2"])]
        cation_o = [a for a in other_atoms if (a.parent.resname == "LYS" and a.name == "NZ") or (a.parent.resname == "ARG" and a.name in ["NH1", "NH2", "NE"]) or (a.parent.resname == "HIS" and a.name in ["ND1", "NE2"])]

        salt_bridges = []
        for ca in cation_p:
            for aa in anion_o:
                d = float(np.linalg.norm(ca.coord - aa.coord))
                if d <= salt_bridge_cutoff:
                    salt_bridges.append(f"{ca.parent.resname}{ca.parent.id[1]}:{ca.name}--{aa.parent.resname}{aa.parent.id[1]}:{aa.name} ({d:.1f}Å)")

        for aa in anion_p:
            for ca in cation_o:
                d = float(np.linalg.norm(aa.coord - ca.coord))
                if d <= salt_bridge_cutoff:
                    salt_bridges.append(f"{aa.parent.resname}{aa.parent.id[1]}:{aa.name}--{ca.parent.resname}{ca.parent.id[1]}:{ca.name} ({d:.1f}Å)")

        return {
            "n_hbonds": len(hbonds),
            "n_salt_bridges": len(salt_bridges),
            "hbonds_details": hbonds,
            "salt_bridges_details": salt_bridges,
        }

    except Exception as exc:
        _log.debug("[BIOPHYSICS] Erreur lors de l'analyse des interactions d'interface : %s", exc)
        return default_result


# ──────────────────────────────────────────────────────────────────────────────
# Analyse des structures secondaires et du diagramme de Ramachandran
# ──────────────────────────────────────────────────────────────────────────────

def analyze_secondary_structure_and_ramachandran(
    pdb_path: Path | str | None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """
    Analyse les angles dièdres du squelette (φ, ψ) de la protéine de novo pour :
      1. Assigner les structures secondaires (% Hélice α, % Brin β, % Boucle/Coil)
         avec filtre physique de consécutivité (Hélice >= 4 résidus, Brin >= 2 résidus).
      2. Détecter les résidus en régions énergétiquement interdites (Ramachandran Outliers).

    Fonctionne en pur Python/NumPy sans aucune dépendance externe (mkdssp / C++ non requis).

    Returns
    -------
    dict avec :
      'helix_pct', 'beta_pct', 'coil_pct',
      'n_helix', 'n_beta', 'n_coil',
      'ramachandran_outliers_pct', 'n_ramachandran_outliers', 'ramachandran_details'.
    """
    _log = logger or log
    default_result: dict[str, Any] = {
        "helix_pct": 0.0,
        "beta_pct": 0.0,
        "coil_pct": 100.0,
        "n_helix": 0,
        "n_beta": 0,
        "n_coil": 0,
        "ramachandran_outliers_pct": 0.0,
        "n_ramachandran_outliers": 0,
        "ramachandran_details": "-",
    }

    if not pdb_path:
        return default_result

    path = Path(pdb_path)
    if not path.is_file():
        return default_result

    try:
        from Bio.PDB import PDBParser, MMCIFParser, Polypeptide

        is_cif = path.suffix.lower() in [".cif", ".mmcif"]
        parser = MMCIFParser(QUIET=True) if is_cif else PDBParser(QUIET=True)
        structure = parser.get_structure("struct_dssp", str(path))

        ppb = Polypeptide.PPBuilder()
        peptides = ppb.build_peptides(structure)

        if not peptides:
            return default_result

        phi_psi_map: list[tuple[str, int, float | None, float | None]] = []
        for pp in peptides:
            for res, (phi, psi) in zip(pp, pp.get_phi_psi_list()):
                p_deg = round(phi * 180.0 / 3.14159265, 1) if phi is not None else None
                s_deg = round(psi * 180.0 / 3.14159265, 1) if psi is not None else None
                phi_psi_map.append((res.resname, res.id[1], p_deg, s_deg))

        total = len(phi_psi_map)
        if total == 0:
            return default_result

        raw_ss: list[str] = []
        outliers: list[str] = []

        for rname, rid, phi, psi in phi_psi_map:
            if phi is None or psi is None:
                raw_ss.append("C")
                continue

            # ── Vérification Ramachandran (zones autorisées vs interdites) ──
            is_outlier = False
            if rname == "GLY":
                if abs(phi) < 20 and abs(psi) < 20:
                    is_outlier = True
            elif rname == "PRO":
                if not (-110 <= phi <= -30 and -80 <= psi <= 190):
                    is_outlier = True
            else:
                alpha_zone = (-160 <= phi <= -25 and -90 <= psi <= 50)
                beta_zone = (-180 <= phi <= -40 and (70 <= psi <= 180 or -180 <= psi <= -150))
                l_alpha_zone = (30 <= phi <= 100 and 10 <= psi <= 90)
                if not (alpha_zone or beta_zone or l_alpha_zone):
                    is_outlier = True

            if is_outlier:
                outliers.append(f"{rname}{rid}(φ={int(phi)}°,ψ={int(psi)}°)")

            # ── Affectation locale selon les bassins dièdres ──
            if -140 <= phi <= -30 and -90 <= psi <= -10:
                raw_ss.append("H")
            elif -180 <= phi <= -50 and (90 <= psi <= 180 or -180 <= psi <= -160):
                raw_ss.append("E")
            else:
                raw_ss.append("C")

        # ── Filtre de consécutivité (Hélice >= 4 résidus, Brin >= 2 résidus) ──
        filtered_ss = ["C"] * total

        # Hélices α (runs >= 4)
        i = 0
        while i < total:
            if raw_ss[i] == "H":
                j = i
                while j < total and raw_ss[j] == "H":
                    j += 1
                if j - i >= 4:
                    for k in range(i, j):
                        filtered_ss[k] = "H"
                i = j
            else:
                i += 1

        # Brins β (runs >= 2)
        i = 0
        while i < total:
            if raw_ss[i] == "E":
                j = i
                while j < total and raw_ss[j] == "E":
                    j += 1
                if j - i >= 2:
                    for k in range(i, j):
                        filtered_ss[k] = "E"
                i = j
            else:
                i += 1

        n_h = sum(1 for s in filtered_ss if s == "H")
        n_e = sum(1 for s in filtered_ss if s == "E")
        n_c = total - n_h - n_e

        helix_pct = round(100.0 * n_h / total, 1)
        beta_pct = round(100.0 * n_e / total, 1)
        coil_pct = round(100.0 * n_c / total, 1)

        evaluated_angles = sum(1 for _, _, p, s in phi_psi_map if p is not None and s is not None)
        outlier_pct = round(100.0 * len(outliers) / evaluated_angles, 1) if evaluated_angles else 0.0

        details = ", ".join(outliers[:3]) if outliers else "-"

        return {
            "helix_pct": helix_pct,
            "beta_pct": beta_pct,
            "coil_pct": coil_pct,
            "n_helix": n_h,
            "n_beta": n_e,
            "n_coil": n_c,
            "ramachandran_outliers_pct": outlier_pct,
            "n_ramachandran_outliers": len(outliers),
            "ramachandran_details": details,
        }

    except Exception as exc:
        _log.debug("[BIOPHYSICS] Erreur analyse structures secondaires / Ramachandran : %s", exc)
        return default_result


# ──────────────────────────────────────────────────────────────────────────────
# Analyse et Préservation de la Coordination Métallique (Zn2+, etc.)
# ──────────────────────────────────────────────────────────────────────────────

def ensure_metal_coordination_conect(
    pdb_path: Path | str,
    cutoff: float = 2.8,
) -> dict[str, Any]:
    """
    Détecte les ions métalliques et les atomes coordinateurs (Cys SG, His NE2/ND1,
    Asp OD, Glu OE) dans un PDB et y injecte explicitement les enregistrements LINK
    et CONECT pour garantir le maintien visuel et structural des liaisons de coordination.
    """
    path = Path(pdb_path)
    if not path.is_file():
        return {"metals": 0, "bonds": 0}

    import numpy as _np

    lines = path.read_text(encoding="utf-8").splitlines()
    metal_elements = {"ZN", "FE", "CU", "MG", "MN", "NI", "CO", "CD", "CA"}

    metals = []
    coord_candidates = []
    existing_conect = set()
    existing_links = []

    for line in lines:
        if line.startswith("LINK"):
            existing_links.append(line)
        elif line.startswith("CONECT"):
            existing_conect.add(line)
        elif line.startswith(("ATOM", "HETATM")):
            try:
                serial = int(line[6:11].strip())
                name = line[12:16].strip().upper()
                res_name = line[17:20].strip().upper()
                chain = line[21]
                res_seq = int(line[22:26].strip())
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                pos = _np.array([x, y, z])
            except Exception:
                continue

            if line.startswith("HETATM") and (res_name in metal_elements or name in metal_elements):
                metals.append({
                    "serial": serial, "name": name, "res_name": res_name,
                    "chain": chain, "res_seq": res_seq, "pos": pos,
                })
            elif (res_name == "CYS" and name == "SG") or \
                 (res_name == "HIS" and name in ["NE2", "ND1"]) or \
                 (res_name == "ASP" and name in ["OD1", "OD2"]) or \
                 (res_name == "GLU" and name in ["OE1", "OE2"]):
                coord_candidates.append({
                    "serial": serial, "name": name, "res_name": res_name,
                    "chain": chain, "res_seq": res_seq, "pos": pos,
                })

    if not metals or not coord_candidates:
        return {"metals": len(metals), "bonds": 0}

    new_links = []
    new_conects = []
    bonds_count = 0

    for m in metals:
        bonded_serials = []
        for c in coord_candidates:
            d = _np.linalg.norm(m["pos"] - c["pos"])
            if d <= cutoff:
                bonded_serials.append(c["serial"])
                bonds_count += 1
                link_str = (
                    f"LINK         {c['name']:<4s}{c['res_name']:>3s} {c['chain']}{c['res_seq']:4d}                 "
                    f"{m['name']:<4s}{m['res_name']:>3s} {m['chain']}{m['res_seq']:4d}     1555   1555  {d:4.2f}"
                )
                new_links.append(link_str)
                new_conects.append(f"CONECT{c['serial']:5d}{m['serial']:5d}")

        if bonded_serials:
            for chunk_start in range(0, len(bonded_serials), 4):
                chunk = bonded_serials[chunk_start:chunk_start + 4]
                s_str = "".join(f"{s:5d}" for s in chunk)
                new_conects.append(f"CONECT{m['serial']:5d}{s_str}")

    header = []
    body = []
    conects = []

    for line in lines:
        if line.startswith("LINK"):
            continue
        elif line.startswith("CONECT"):
            conects.append(line)
        elif line.startswith("END"):
            continue
        elif line.startswith(("REMARK", "HEADER", "TITLE")):
            header.append(line)
        else:
            body.append(line)

    all_links = list(dict.fromkeys(new_links + existing_links))
    all_conects = list(dict.fromkeys(conects + new_conects))

    final_lines = header + all_links + body + all_conects + ["END"]
    path.write_text("\n".join(final_lines) + "\n", encoding="utf-8")

    return {"metals": len(metals), "bonds": bonds_count}


def analyze_metal_coordination(
    pdb_path: Path | str | None,
    cutoff: float = 2.8,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """
    Analyse quantitative de la sphère de coordination métallique (Zn2+, etc.).
    Retourne le nombre de ligands coordinateurs, les distances et la géométrie.
    """
    _log = logger or log
    default_res: dict[str, Any] = {
        "metals_count": 0,
        "coordination_number": 0,
        "is_tetrahedral": False,
        "coordination_residues": [],
        "coordination_details": "-",
    }
    if not pdb_path:
        return default_res

    path = Path(pdb_path)
    if not path.is_file():
        return default_res

    try:
        import numpy as _np
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        metal_elements = {"ZN", "FE", "CU", "MG", "MN", "NI", "CO", "CD", "CA"}

        metals = []
        candidates = []

        for line in lines:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            try:
                name = line[12:16].strip().upper()
                res_name = line[17:20].strip().upper()
                chain = line[21]
                res_seq = int(line[22:26].strip())
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                pos = _np.array([x, y, z])
            except Exception:
                continue

            if line.startswith("HETATM") and (res_name in metal_elements or name in metal_elements):
                metals.append({"name": name, "res": res_name, "chain": chain, "seq": res_seq, "pos": pos})
            elif (res_name == "CYS" and name == "SG") or \
                 (res_name == "HIS" and name in ["NE2", "ND1"]) or \
                 (res_name == "ASP" and name in ["OD1", "OD2"]) or \
                 (res_name == "GLU" and name in ["OE1", "OE2"]):
                candidates.append({"name": name, "res": res_name, "chain": chain, "seq": res_seq, "pos": pos})

        if not metals:
            return default_res

        # On évalue le premier métal principal (ex: Zn)
        m = metals[0]
        coord_bonds = []
        for c in candidates:
            d = _np.linalg.norm(m["pos"] - c["pos"])
            if d <= cutoff:
                coord_bonds.append((f"{c['res']}{c['seq']}({c['name']})", round(float(d), 2)))

        cn = len(coord_bonds)
        details = ", ".join(f"{res}:{d}Å" for res, d in coord_bonds) if coord_bonds else "-"

        return {
            "metals_count": len(metals),
            "coordination_number": cn,
            "is_tetrahedral": cn == 4,
            "coordination_residues": [r for r, _ in coord_bonds],
            "coordination_details": details,
        }

    except Exception as exc:
        _log.debug("[BIOPHYSICS] Erreur analyse coordination métallique : %s", exc)
        return default_res

