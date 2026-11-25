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
    "pi",
    "charge_ph7",
    "gravy",
    "mw_kda",
    "solubility_score",
    "solubility_percent",
    "plddt_mean",
    "plddt_pocket",
    "ptm",
    "iptm",
    "pae_inter",
    "sasa",
    "delta_sasa",
    "hydrophobic_sasa_pct",
    "n_interface_hbonds",
    "hbond_density",
    "n_interface_salt_bridges",
    "n_unpaired_cys",
    "unpaired_cys_details",
    "n_liabilities",
    "liabilities_details",
    "helix_pct",
    "beta_pct",
    "coil_pct",
    "ramachandran_outliers_pct",
    "n_ramachandran_outliers",
    "ramachandran_details",
    "rmsd_angstrom",
    "motif_rmsd_angstrom",
    "n_steric_clashes",
    "ligand_burial_pct",
    "n_metal_coord",
    "metal_coord_details",
    "composite_score",
    "passed",
    "sequence",
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

    from optobind_ai.modules.biophysics import _effective_nll_score

    for seq in sequences:
        val = val_index.get(seq.name)
        row: dict = {
            "_rank_nll"         : _effective_nll_score(seq),
            "run_id"            : run_id,
            "variant_name"      : seq.name,
            "scaffold_index"    : seq.scaffold.index,
            "sequence_index"    : seq.index,
            "sequence_length"   : len(seq.sequence),
            "mpnn_score"        : f"{seq.mpnn_score:.4f}",
            "pi"                : f"{seq.pi:.2f}",
            "charge_ph7"        : f"{seq.charge_ph7:.2f}",
            "gravy"             : f"{seq.gravy:.3f}",
            "mw_kda"            : f"{seq.mw_kda:.1f}",
            "solubility_score"  : f"{val.solubility_score:.3f}" if val and val.solubility_score is not None else "N/A",
            "solubility_percent": f"{val.solubility_percent:.1f}%" if val and val.solubility_percent is not None else "N/A",
            "plddt_mean"        : f"{val.plddt_mean:.2f}"    if val else "N/A",
            "plddt_pocket"      : f"{val.plddt_pocket:.2f}"  if val else "N/A",
            "ptm"               : f"{val.ptm:.4f}"           if val else "N/A",
            "iptm"              : f"{val.iptm:.4f}"          if val else "N/A",
            "pae_inter"         : f"{val.pae_inter:.2f}"     if val else "N/A",
            "sasa"                    : f"{val.sasa:.1f}"                  if val else "N/A",
            "delta_sasa"              : f"{val.delta_sasa:.1f}"            if val else "N/A",
            "hydrophobic_sasa_pct"    : f"{val.hydrophobic_sasa_pct:.1f}%" if val else "N/A",
            "n_interface_hbonds"      : str(val.n_interface_hbonds)        if val else "N/A",
            "hbond_density"           : f"{val.hbond_density:.2f}"         if val else "N/A",
            "n_interface_salt_bridges": str(val.n_interface_salt_bridges)  if val else "N/A",
            "n_unpaired_cys"          : str(val.n_unpaired_cys)            if val else "N/A",
            "unpaired_cys_details"    : val.unpaired_details               if val else "N/A",
            "n_liabilities"           : str(val.n_liabilities)             if val else "N/A",
            "liabilities_details"     : val.liabilities_details            if val else "N/A",
            "helix_pct"               : f"{val.helix_pct:.1f}%"            if val else "N/A",
            "beta_pct"                : f"{val.beta_pct:.1f}%"             if val else "N/A",
            "coil_pct"                : f"{val.coil_pct:.1f}%"             if val else "N/A",
            "ramachandran_outliers_pct": f"{val.ramachandran_outliers_pct:.1f}%" if val else "N/A",
            "n_ramachandran_outliers" : str(val.n_ramachandran_outliers)   if val else "N/A",
            "ramachandran_details"    : val.ramachandran_details           if val else "N/A",
            "rmsd_angstrom"           : f"{val.rmsd:.4f}"                  if val else "N/A",
            "motif_rmsd_angstrom"     : f"{val.motif_rmsd:.4f}"            if val else "N/A",
            "n_steric_clashes"        : str(val.n_clashes)                 if val else "N/A",
            "ligand_burial_pct"       : f"{val.ligand_burial_pct:.1f}%"    if val else "N/A",
            "n_metal_coord"           : str(val.n_metal_coord)             if val else "N/A",
            "metal_coord_details"     : val.metal_coord_details            if val else "N/A",
            "composite_score"         : f"{val.composite_score:.1f}"       if val else "0.0",
            "passed"            : str(val.passed) if val else ("False (Pre-Filter)" if seq.metadata.get("rejection_reason") else "N/A"),
            "sequence"          : seq.sequence,
            "scaffold_pdb"      : str(seq.scaffold.pdb_path),
            "predicted_pdb"     : str(val.pdb_path or "N/A") if val else (f"EXCLUDED: {seq.metadata['rejection_reason']}" if seq.metadata.get("rejection_reason") else "N/A"),
            "fasta_path"        : str(seq.fasta_path),
        }
        rows.append(row)

    # Trier par candidats validés d'abord puis par composite_score décroissant
    rows.sort(
        key=lambda r: (
            r["passed"] == "True",
            float(r["composite_score"]) if r["composite_score"] not in ("N/A", "") else -1.0,
            -float(r.get("_rank_nll", 999.0)),
        ),
        reverse=True,
    )
    for r in rows:
        r.pop("_rank_nll", None)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_HEADERS)
        writer.writeheader()
        writer.writerows(rows)

    # Export automatique au format Excel natif (.xlsx) via openpyxl direct (sans dépendance lourde pandas)
    xlsx_path = output_path.with_suffix(".xlsx")
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "OptoBind_Validation"
        ws.append(_CSV_HEADERS)
        header_fill = PatternFill(start_color="181B22", end_color="181B22", fill_type="solid")
        header_font = Font(color="FFFFFF", bold=True)
        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = header_font
        for r in rows:
            ws.append([r.get(h, "") for h in _CSV_HEADERS])
        wb.save(xlsx_path)
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
            _sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass
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

    sep = "-" * 144

    _print(f"\n{BOLD}{'=' * 144}{RESET}")
    _print(f"{BOLD}  RESUME OptoBind-AI{RESET}")
    _print(f"{sep}")
    _print(f"  Squelettes generes  : {len(scaffolds)}")
    _print(f"  Sequences optimisees: {len(sequences)}")
    _print(f"  Variants valides    : {GREEN}{n_pass} PASS{RESET}  /  {RED}{n_fail} FAIL{RESET}"
           + (f"  /  {YELLOW}{n_no_val} non valides{RESET}" if n_no_val else ""))
    _print(f"  Seuils appliques    : pLDDT > {plddt_threshold}  |  RMSD < {rmsd_threshold} A  |  mRMSD <= 1.5 A  |  Cys = 0  |  %Hydro <= 35%  |  %Coil <= 45%  |  Rama <= 2%  |  Liab = 0")
    _print(f"{sep}")

    if validation_results:
        header = (
            f"  {'Variant':<26} {'pLDDT':>5} {'pTM':>5} {'ipTM':>5} {'PAE':>4} {'dSASA':>5} "
            f"{'%Hyd':>5} {'HB':>2} {'HBD':>4} {'SB':>2} {'pI':>5} {'GRAVY':>5} {'Solub':>5} {'SS(H/E/C)':>9} {'Rama':>5} {'Liab':>4} {'Cys':>4} {'RMSD':>5} {'mRMSD':>5} {'Clash':>5} {'Status':>7}"
        )
        _print(header)
        _print(f"  {sep[2:]}")
        for v in validation_results:
            color  = GREEN if v.passed else RED
            status = "PASS OK" if v.passed else "FAIL  X"
            sol_str = f"{v.solubility_score:.2f}" if v.solubility_score is not None else "-"
            cys_str = "0" if v.n_unpaired_cys == 0 else f"{v.n_unpaired_cys}(C{v.unpaired_details.split('(')[0][1:]})"[:4]
            m_rmsd_str = f"{v.motif_rmsd:.2f}" if v.motif_rmsd != float("inf") else "inf"
            ss_str = f"{int(v.helix_pct):>2}/{int(v.beta_pct):>2}/{int(v.coil_pct):>2}%"
            rama_str = f"{v.ramachandran_outliers_pct:>4.1f}%"
            liab_str = "0" if v.n_liabilities == 0 else f"{v.n_liabilities}({v.liabilities_details.split('(')[0]})"[:4]
            hbd_str = f"{v.hbond_density:.2f}" if v.hbond_density > 0 else "0.00"
            _print(
                f"  {v.sequence.name:<26} {v.plddt_mean:>5.1f} "
                f"{v.ptm:>5.2f} {v.iptm:>5.2f} {v.pae_inter:>4.1f} {v.delta_sasa:>5.0f} "
                f"{v.hydrophobic_sasa_pct:>4.1f}% {v.n_interface_hbonds:>2d} {hbd_str:>4} {v.n_interface_salt_bridges:>2d} "
                f"{v.sequence.pi:>5.2f} {v.sequence.gravy:>5.2f} {sol_str:>5} {ss_str:>9} {rama_str:>5} {liab_str:>4} {cys_str:>4} "
                f"{v.rmsd:>5.2f} {m_rmsd_str:>5} {v.n_clashes:>5d} "
                f"{color}{status:>7}{RESET}"
            )
    else:
        _print(f"  {YELLOW}Aucune validation effectuee (--skip-validation){RESET}")

    _print(f"{sep}")
    _print(f"  Rapport CSV   : {report_path}")
    _print(f"  Rapport Excel : {report_path.with_suffix('.xlsx')}")
    _print(f"{'=' * 144}{RESET}\n")
