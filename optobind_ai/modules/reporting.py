"""
optobind_ai.modules.reporting
================================
Génération du rapport de validation in silico (CSV) et affichage
du résumé console final.

Le fichier CSV (``outputs/<run_id>/reports/report.csv``) contient
une ligne par variant avec les colonnes décrites dans le cahier des charges.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

from optobind_ai.utils.io_helpers import (
    DesignedSequence,
    Scaffold,
    ValidationResult,
)


# ──────────────────────────────────────────────────────────────────────────────
# Construction du rapport CSV
# ──────────────────────────────────────────────────────────────────────────────

# En-têtes du rapport CSV
_CSV_HEADERS = [
    "run_id",
    "variant_name",
    "scaffold_index",
    "sequence_index",
    "sequence_length",
    "mpnn_score",
    "plddt_mean",
    "plddt_pocket",
    "rmsd_angstrom",
    "n_steric_clashes",
    "passed",
    "scaffold_pdb",
    "predicted_pdb",
    "fasta_path",
]


def build_report(
    scaffolds          : list[Scaffold],
    sequences          : list[DesignedSequence],
    validation_results : list[ValidationResult],
    output_path        : Path,
    run_id             : str,
    cfg                : dict[str, Any],
    logger             : logging.Logger,
) -> Path:
    """
    Génère le rapport CSV de validation in silico.

    Parameters
    ----------
    scaffolds          : Squelettes générés par RFdiffusion.
    sequences          : Séquences optimisées par ProteinMPNN.
    validation_results : Résultats de validation AlphaFold / Chai-1.
    output_path        : Chemin du fichier CSV à écrire.
    run_id             : Identifiant unique du run.
    cfg                : Configuration complète (pour inclure les seuils).
    logger             : Logger partagé du pipeline.

    Returns
    -------
    Path
        Chemin du fichier CSV généré.
    """
    # Index des résultats de validation par nom de séquence
    val_index: dict[str, ValidationResult] = {
        v.sequence.name: v for v in validation_results
    }

    rows: list[dict] = []

    for seq in sequences:
        val = val_index.get(seq.name)
        row: dict = {
            "run_id"            : run_id,
            "variant_name"      : seq.name,
            "scaffold_index"    : seq.scaffold.index,
            "sequence_index"    : seq.index,
            "sequence_length"   : len(seq.sequence),
            "mpnn_score"        : f"{seq.mpnn_score:.4f}",
            "plddt_mean"        : f"{val.plddt_mean:.2f}"    if val else "N/A",
            "plddt_pocket"      : f"{val.plddt_pocket:.2f}"  if val else "N/A",
            "rmsd_angstrom"     : f"{val.rmsd:.4f}"          if val else "N/A",
            "n_steric_clashes"  : str(val.n_clashes)         if val else "N/A",
            "passed"            : str(val.passed)            if val else "N/A",
            "scaffold_pdb"      : str(seq.scaffold.pdb_path),
            "predicted_pdb"     : str(val.pdb_path or "N/A") if val else "N/A",
            "fasta_path"        : str(seq.fasta_path),
        }
        rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_HEADERS)
        writer.writeheader()
        writer.writerows(rows)

    # Export automatique au format Excel natif (.xlsx) pour compatibilité universelle FR & EN
    xlsx_path = output_path.with_suffix(".xlsx")
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="OptoBind_Validation")
        logger.debug("[REPORT] Rapport Excel natif écrit : %s", xlsx_path)
    except Exception as exc:
        logger.debug("[REPORT] Export .xlsx ignoré : %s", exc)

    logger.debug("[REPORT] %d lignes écrites dans %s", len(rows), output_path)
    return output_path


# ──────────────────────────────────────────────────────────────────────────────
# Résumé console
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(
    scaffolds          : list[Scaffold],
    sequences          : list[DesignedSequence],
    validation_results : list[ValidationResult],
    report_path        : Path,
    plddt_threshold    : float,
    rmsd_threshold     : float,
) -> None:
    """
    Affiche un tableau récapitulatif coloré en console.
    """
    import sys as _sys
    # Sur Windows, assurer l'encodage UTF-8 pour les caractères spéciaux (═─✓✗)
    if _sys.platform == "win32":
        try:
            import io
            _out = io.TextIOWrapper(_sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
        except AttributeError:
            _out = _sys.stdout
    else:
        _out = _sys.stdout

    def _print(*args, **kwargs):
        kwargs.setdefault("file", _out)
        print(*args, **kwargs)

    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"

    n_pass = sum(1 for v in validation_results if v.passed)
    n_fail = len(validation_results) - n_pass
    n_no_val = len(sequences) - len(validation_results)

    sep = "-" * 72

    _print(f"\n{BOLD}{'=' * 72}{RESET}")
    _print(f"{BOLD}  RESUME OptoBind-AI{RESET}")
    _print(f"{sep}")
    _print(f"  Squelettes generes  : {len(scaffolds)}")
    _print(f"  Sequences optimisees: {len(sequences)}")
    _print(f"  Variants valides    : {GREEN}{n_pass} PASS{RESET}  /  {RED}{n_fail} FAIL{RESET}"
           + (f"  /  {YELLOW}{n_no_val} non valides{RESET}" if n_no_val else ""))
    _print(f"  Seuils appliques    : pLDDT > {plddt_threshold}  |  RMSD < {rmsd_threshold} A")
    _print(f"{sep}")

    if validation_results:
        header = (
            f"  {'Variant':<35} {'pLDDT':>7} {'RMSD(A)':>8} "
            f"{'Clashes':>8} {'Status':>7}"
        )
        _print(header)
        _print(f"  {sep[2:]}")
        for v in validation_results:
            color  = GREEN if v.passed else RED
            status = "PASS OK" if v.passed else "FAIL  X"
            _print(
                f"  {v.sequence.name:<35} {v.plddt_mean:>7.2f} "
                f"{v.rmsd:>8.3f} {v.n_clashes:>8d} "
                f"{color}{status:>7}{RESET}"
            )
    else:
        _print(f"  {YELLOW}Aucune validation effectuee (--skip-validation){RESET}")

    _print(f"{sep}")
    _print(f"  Rapport CSV   : {report_path}")
    _print(f"  Rapport Excel : {report_path.with_suffix('.xlsx')}")
    _print(f"{'=' * 72}{RESET}\n")
