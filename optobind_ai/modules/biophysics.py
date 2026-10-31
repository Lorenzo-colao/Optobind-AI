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
