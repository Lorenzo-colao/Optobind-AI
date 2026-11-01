"""
optobind_ai.utils.io_helpers
=============================
Gestion des entrées/sorties du pipeline : création des répertoires de sortie,
copie de snapshots d'entrée, parsers PDB/FASTA légers.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

log = logging.getLogger("optobind")


# ──────────────────────────────────────────────────────────────────────────────
# Structures de données légères
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Scaffold:
    """Représente un squelette 3D généré par RFdiffusion."""

    index      : int
    pdb_path   : Path
    run_id     : str
    n_residues : int  = 0
    metadata   : dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"{self.run_id}_scaffold_{self.index:03d}"


@dataclass
class DesignedSequence:
    """Représente une séquence optimisée par ProteinMPNN."""

    index        : int
    scaffold     : Scaffold
    sequence     : str
    fasta_path   : Path
    mpnn_score   : float = 0.0   # score global_score (log-vraisemblance négative)
    pi           : float = 7.0   # point isoélectrique théorique
    charge_ph7   : float = 0.0   # charge nette à pH 7.4
    gravy        : float = 0.0   # indice d'hydropathie GRAVY (< 0 = hydrophile)
    mw_kda       : float = 0.0   # masse moléculaire en kDa
    metadata     : dict  = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"{self.scaffold.name}_seq_{self.index:02d}"

    def to_fasta(self) -> str:
        return f">{self.name}\n{self.sequence}\n"


@dataclass
class ValidationResult:
    """Résultat de la validation structurale AlphaFold / Chai-1."""

    sequence           : DesignedSequence
    pdb_path           : Path | None = None
    plddt_mean         : float = 0.0
    plddt_pocket       : float = 0.0     # pLDDT moyen sur la poche de liaison
    rmsd               : float = float("inf")
    motif_rmsd         : float = float("inf") # RMSD Cα sur les résidus du motif fonctionnel (Å)
    n_clashes          : int   = 0
    ptm                : float = 0.0     # predicted TM-score (qualité du repliement global)
    iptm               : float = 0.0     # interface predicted TM-score (complexe / ligand)
    pae_mean           : float = 0.0     # PAE moyen global (Å)
    pae_inter          : float = 0.0     # PAE inter-domaines / inter-chaînes (Å)
    sasa                     : float = 0.0     # Surface accessible au solvant totale (Å²)
    delta_sasa               : float = 0.0     # Surface enfouie à l'interface (ΔSASA en Å²)
    hydrophobic_sasa_pct     : float = 0.0     # Pourcentage de surface hydrophobe (V, L, I, F, W, M)
    n_interface_hbonds       : int = 0         # Nombre de liaisons H à l'interface (d <= 3.5 Å)
    n_interface_salt_bridges : int = 0         # Nombre de ponts salins à l'interface (d <= 4.0 Å)
    solubility_score         : float | None = None  # Score mis à l'échelle Protein-Sol (> 0.45)
    solubility_percent       : float | None = None  # % de solubilité estimé Protein-Sol
    n_unpaired_cys           : int = 0              # Nombre de cystéines orphelines (non coordonnées / non pontées)
    unpaired_details         : str = ""             # Détail des cystéines libres (ex: C4(exp:29Å²))
    helix_pct                : float = 0.0          # % résidus en hélice alpha (runs >= 4)
    beta_pct                 : float = 0.0          # % résidus en brin beta (runs >= 2)
    coil_pct                 : float = 0.0          # % résidus en boucle/coil
    ramachandran_outliers_pct: float = 0.0          # % résidus outliers Ramachandran (< 2.0%)
    n_ramachandran_outliers  : int   = 0            # Nombre absolu d'outliers
    ramachandran_details     : str   = ""           # Détail des résidus outliers (ex: GLY28...)
    passed               : bool  = False
    metadata             : dict  = field(default_factory=dict)

    def validate(
        self,
        rmsd_thresh: float = 2.0,
        plddt_thresh: float = 90.0,
        *,
        rmsd_threshold: float | None = None,
        motif_rmsd_threshold: float | None = 1.5,
        plddt_threshold: float | None = None,
        max_unpaired_cys: int | None = 0,
        max_hydrophobic_sasa_pct: float | None = 35.0,
        max_coil_pct: float | None = 45.0,
        max_ramachandran_outliers_pct: float | None = 2.0,
    ) -> bool:
        """Applique les critères de succès et met à jour ``passed``."""
        r_thresh = rmsd_threshold if rmsd_threshold is not None else rmsd_thresh
        m_thresh = motif_rmsd_threshold
        p_thresh = plddt_threshold if plddt_threshold is not None else plddt_thresh
        cys_ok   = (self.n_unpaired_cys <= max_unpaired_cys) if max_unpaired_cys is not None else True
        hydro_ok = (self.hydrophobic_sasa_pct <= max_hydrophobic_sasa_pct) if max_hydrophobic_sasa_pct is not None else True
        motif_ok = (self.motif_rmsd < m_thresh) if (m_thresh is not None and self.motif_rmsd != float("inf")) else True
        coil_ok  = (self.coil_pct <= max_coil_pct) if max_coil_pct is not None else True
        rama_ok  = (self.ramachandran_outliers_pct <= max_ramachandran_outliers_pct) if max_ramachandran_outliers_pct is not None else True
        self.passed = (
            self.rmsd             < r_thresh   and
            self.plddt_mean       > p_thresh   and
            motif_ok                           and
            cys_ok                             and
            hydro_ok                           and
            coil_ok                            and
            rama_ok
        )
        return self.passed


# ──────────────────────────────────────────────────────────────────────────────
# Gestion des répertoires
# ──────────────────────────────────────────────────────────────────────────────

def ensure_output_dirs(base: Path, run_id: str) -> dict[str, Path]:
    """
    Crée la structure de répertoires de sortie pour un run donné.

    Returns
    -------
    dict[str, Path]
        Mapping nom logique → chemin absolu.
    """
    run_root = base / run_id
    dirs: dict[str, Path] = {
        "root"            : run_root,
        "pdbs"            : run_root / "pdbs",
        "fasta"           : run_root / "fasta",
        "reports"         : run_root / "reports",
        "logs"            : base    / "logs",          # logs partagés
        "inputs_snapshot" : run_root / "inputs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    log.debug("[IO] Répertoires de sortie créés : %s", run_root)
    return dirs


def copy_inputs_snapshot(
    target : Path,
    ligand : Path,
    config : Path,
    dest   : Path,
) -> None:
    """
    Copie les fichiers d'entrée dans le snapshot du run pour reproductibilité.
    """
    for src in [target, ligand, config]:
        dst = dest / src.name
        shutil.copy2(src, dst)
        log.debug("[IO] Snapshot copié : %s → %s", src.name, dst)


# ──────────────────────────────────────────────────────────────────────────────
# Parsers légers
# ──────────────────────────────────────────────────────────────────────────────

def read_fasta(fasta_path: Path) -> Iterator[tuple[str, str]]:
    """
    Générateur de paires (header, sequence) depuis un fichier FASTA.
    Supporte les séquences multi-lignes.
    """
    header, seqs = "", []
    with fasta_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header:
                    yield header, "".join(seqs)
                header, seqs = line[1:], []
            else:
                seqs.append(line)
    if header:
        yield header, "".join(seqs)


def count_pdb_residues(pdb_path: Path) -> int:
    """
    Compte le nombre de résidus uniques (ATOM records) dans un fichier PDB.
    """
    seen: set[tuple[str, int, str]] = set()
    with pdb_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("ATOM"):
                chain   = line[21]
                res_seq = int(line[22:26].strip())
                ins_code = line[26].strip()
                seen.add((chain, res_seq, ins_code))
    return len(seen)


def write_fasta(sequences: list[DesignedSequence], fasta_path: Path) -> None:
    """Écrit une liste de DesignedSequence dans un seul fichier FASTA."""
    with fasta_path.open("w", encoding="utf-8") as fh:
        for seq in sequences:
            fh.write(seq.to_fasta())
    log.debug("[IO] FASTA écrit : %s (%d séquences)", fasta_path, len(sequences))
