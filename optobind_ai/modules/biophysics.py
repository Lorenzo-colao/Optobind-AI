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
