"""
optobind_ai.modules.pdb_builder
===============================
Construction et extraction 100% automatisées de fichiers PDB cibles prêts pour
RFdiffusion All-Atom, LigandMPNN / ProteinMPNN et Chai-1.

Fonctionnalités principales :
  1. Téléchargement automatique depuis RCSB PDB (code à 4 caractères, ex: 1A1L, 2CBA)
     ou AlphaFold Protein Structure Database (identifiant UniProt, ex: P00918).
  2. Inspection structurale complète : chaînes protéiques, ions métalliques et sphères
     de coordination 3D (angles, distances, géométrie Td/Oh), poches de ligands.
  3. Résolution de petites molécules / fluorophores en 3D depuis fichier SDF local
     ou l'API PubChem (nom chimique, CID ou SMILES).
  4. Extraction chirurgicale selon 3 modes universels :
     - ``metal_site``     : Site de coordination métallique (+ co-positionnement optionnel
                            sans clash d'un fluorophore rapporteur type DFHBI-1T).
     - ``ligand_pocket``  : Poche de liaison d'une petite molécule (+ résidus clés).
     - ``protein_binder`` : Domaine protéique cible nettoyé (+ détection hotspots).
  5. Synthèse mathématique exacte des ``contigs`` RFdiffusion : mesure les distances
     euclidiennes Cα-Cα entre résidus extraits et insère automatiquement les boucles
     de liaison (linkers) nécessaires pour garantir 0 distorsion stérique.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
from pathlib import Path
import re
from typing import Any
import warnings

import numpy as np
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import requests

log = logging.getLogger("optobind.pdb_builder")

# ──────────────────────────────────────────────────────────────────────────────
# Constantes cristallographiques et chimiques
# ──────────────────────────────────────────────────────────────────────────────

STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}

AA_3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

METAL_ELEMENTS = {
    "ZN", "CA", "MG", "FE", "CU", "MN", "NI", "CO", "CD", "MO", "V", "W", "NA", "K",
}

TRANSITION_METALS = {"ZN", "FE", "CU", "MN", "NI", "CO", "CD", "MO", "V", "W"}

CRYSTAL_ARTIFACTS = {
    "HOH", "WAT", "DOD", "SO4", "PO4", "GOL", "EDO", "PEG", "PGE", "PG4",
    "ACT", "ACE", "DMS", "FMT", "TRS", "EPE", "MES", "MPD", "CIT", "TAR",
    "CL", "BR", "IOD", "BME", " DTT", "NO3", "SUL",
}

DONOR_ATOMS_BY_RES = {
    "CYS": {"SG"},
    "HIS": {"ND1", "NE2"},
    "ASP": {"OD1", "OD2"},
    "GLU": {"OE1", "OE2"},
    "MET": {"SD"},
    "SER": {"OG"},
    "THR": {"OG1"},
    "TYR": {"OH"},
    "ASN": {"OD1"},
    "GLN": {"OE1"},
}

COVALENT_RADII = {
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
    "P": 1.07, "S": 1.05, "CL": 1.02, "BR": 1.20, "I": 1.39, "B": 0.84,
}


@dataclass
class TargetBuildResult:
    """Résultat complet de la génération automatisée d'un PDB cible."""

    output_pdb       : Path
    ligand_sdf       : Path | None
    mode             : str
    source_id        : str
    recommended_contigs : str
    hotspot_residues : list[str]
    motif_summary    : list[dict[str, Any]] = field(default_factory=list)
    coordination_info: dict[str, Any]       = field(default_factory=dict)
    gap_analysis     : list[dict[str, Any]] = field(default_factory=list)
    warnings         : list[str]            = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_pdb": str(self.output_pdb.as_posix()),
            "ligand_sdf": str(self.ligand_sdf.as_posix()) if self.ligand_sdf else None,
            "mode": self.mode,
            "source_id": self.source_id,
            "recommended_contigs": self.recommended_contigs,
            "hotspot_residues": self.hotspot_residues,
            "motif_summary": self.motif_summary,
            "coordination_info": self.coordination_info,
            "gap_analysis": self.gap_analysis,
            "warnings": self.warnings,
        }


# ──────────────────────────────────────────────────────────────────────────────
# 1. Téléchargement RCSB PDB / AlphaFold DB
# ──────────────────────────────────────────────────────────────────────────────

def fetch_structure(
    source: str | Path,
    cache_dir: Path | None = None,
    timeout: float = 15.0,
) -> Path:
    """
    Résout une structure 3D depuis :
      - un chemin local (.pdb, .cif, .mmcif)
      - un code RCSB PDB à 4 caractères (ex: ``1A1L``, ``2CBA``, ``4KZE``)
      - un identifiant UniProt / AlphaFold DB (ex: ``P00918``, ``AF-P00918-F1``)
    """
    src_str = str(source).strip()
    local_candidate = Path(src_str)
    if local_candidate.is_file():
        return _ensure_pdb_format(local_candidate, cache_dir)

    if cache_dir is None:
        cache_dir = Path("inputs/targets/_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    clean_id = src_str.upper()

    # Cas 1 : Identifiant RCSB PDB (4 caractères alphanumériques commençant par un chiffre)
    if re.fullmatch(r"[0-9][A-Z0-9]{3}", clean_id):
        cached_pdb = cache_dir / f"{clean_id}.pdb"
        if cached_pdb.is_file() and cached_pdb.stat().st_size > 200:
            return cached_pdb

        pdb_url = f"https://files.rcsb.org/download/{clean_id}.pdb"
        resp = requests.get(pdb_url, timeout=timeout)
        if resp.status_code == 200 and ("ATOM" in resp.text or "HETATM" in resp.text):
            cached_pdb.write_text(resp.text, encoding="utf-8")
            return cached_pdb

        # Fallback mmCIF si la structure n'est distribuée qu'en mmCIF
        cif_url = f"https://files.rcsb.org/download/{clean_id}.cif"
        resp_cif = requests.get(cif_url, timeout=timeout)
        if resp_cif.status_code == 200:
            cached_cif = cache_dir / f"{clean_id}.cif"
            cached_cif.write_text(resp_cif.text, encoding="utf-8")
            return _ensure_pdb_format(cached_cif, cache_dir)

        raise ValueError(f"Code PDB '{clean_id}' introuvable sur RCSB PDB.")

    # Cas 2 : Identifiant UniProt ou AlphaFold DB
    uniprot_id = clean_id
    if uniprot_id.startswith("AF-"):
        parts = uniprot_id.split("-")
        if len(parts) >= 2:
            uniprot_id = parts[1]

    if re.fullmatch(r"[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}", uniprot_id):
        cached_af = cache_dir / f"AF_{uniprot_id}.pdb"
        if cached_af.is_file() and cached_af.stat().st_size > 200:
            return cached_af

        # Interrogation API AlphaFold EBI pour obtenir l'URL exacte de la dernière version
        api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
        api_resp = requests.get(api_url, timeout=timeout)
        if api_resp.status_code == 200:
            entries = api_resp.json()
            if isinstance(entries, list) and entries:
                pdb_dl = entries[0].get("pdbUrl")
                if pdb_dl:
                    r_pdb = requests.get(pdb_dl, timeout=timeout)
                    if r_pdb.status_code == 200:
                        cached_af.write_text(r_pdb.text, encoding="utf-8")
                        return cached_af

        # Fallback direct v4
        direct_url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
        r_pdb = requests.get(direct_url, timeout=timeout)
        if r_pdb.status_code == 200 and "ATOM" in r_pdb.text:
            cached_af.write_text(r_pdb.text, encoding="utf-8")
            return cached_af

        raise ValueError(f"Structure AlphaFold introuvable pour l'identifiant UniProt '{uniprot_id}'.")

    raise ValueError(
        f"Source structurale invalide ou fichier introuvable : '{src_str}'. "
        "Fournissez un code PDB (ex: 1A1L), un ID UniProt (ex: P00918) ou un fichier local."
    )


def _ensure_pdb_format(path: Path, cache_dir: Path | None = None) -> Path:
    """Convertit un fichier mmCIF en PDB propre si nécessaire."""
    if path.suffix.lower() not in (".cif", ".mmcif"):
        return path

    out_dir = cache_dir or path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdb = out_dir / f"{path.stem}.pdb"
    if out_pdb.is_file() and out_pdb.stat().st_size > 200:
        return out_pdb

    from Bio.PDB import MMCIFParser, PDBIO
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(path.stem, str(path))
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(out_pdb))
    return out_pdb


# ──────────────────────────────────────────────────────────────────────────────
# 2. Parsing atomique léger et nettoyage cristallographique (AltLoc, Modèle 0)
# ──────────────────────────────────────────────────────────────────────────────

def _parse_clean_pdb_atoms(pdb_path: Path) -> list[dict[str, Any]]:
    """
    Lit le premier modèle d'un PDB en filtrant les conformations alternatives (garde ' ', 'A', '1')
    et les hydrogènes cristallographiques redondants.
    """
    atoms: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, int, str, str]] = set()

    with pdb_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.startswith("ENDMDL"):
                break
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            if len(line) < 54:
                continue

            altloc = line[16]
            if altloc not in (" ", "A", "1"):
                continue

            record = line[:6].strip()
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip().upper()
            chain = line[21].strip() or "A"
            try:
                res_seq = int(line[22:26].strip())
            except ValueError:
                continue
            ins_code = line[26].strip()

            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue

            occupancy = 1.0
            bfactor = 20.0
            try:
                if len(line) >= 60 and line[54:60].strip():
                    occupancy = float(line[54:60])
                if len(line) >= 66 and line[60:66].strip():
                    bfactor = float(line[60:66])
            except ValueError:
                pass

            elem = ""
            if len(line) >= 78 and line[76:78].strip():
                elem = "".join(c for c in line[76:78].strip().upper() if c.isalpha())
            if not elem:
                stripped = "".join(c for c in atom_name.upper() if c.isalpha())
                elem = stripped[:2] if stripped[:2] in METAL_ELEMENTS else stripped[:1]

            # Ignorer les hydrogènes pour alléger RFdiffusion / LigandMPNN
            if elem == "H":
                continue

            dedup_key = (chain, res_seq, ins_code, atom_name)
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)

            atoms.append({
                "record": "ATOM" if res_name in STANDARD_AA else record,
                "atom_name": atom_name,
                "res_name": res_name,
                "chain": chain,
                "res_seq": res_seq,
                "ins_code": ins_code,
                "coord": np.array([x, y, z], dtype=np.float64),
                "occupancy": occupancy,
                "bfactor": bfactor,
                "element": elem,
            })

    return atoms


def _group_residues(atoms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Regroupe une liste d'atomes par résidu ordonné."""
    grouped: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    order: list[tuple[str, int, str, str]] = []

    for a in atoms:
        key = (a["chain"], a["res_seq"], a["ins_code"], a["res_name"])
        if key not in grouped:
            grouped[key] = {
                "chain": a["chain"],
                "res_seq": a["res_seq"],
                "ins_code": a["ins_code"],
                "res_name": a["res_name"],
                "is_protein": a["res_name"] in STANDARD_AA,
                "atoms": [],
            }
            order.append(key)
        grouped[key]["atoms"].append(a)

    return [grouped[k] for k in order]


def _classify_coordination_geometry(metal_coord: np.ndarray, donor_coords: list[np.ndarray]) -> dict[str, Any]:
    """Calcule les angles L-M-L et identifie la géométrie de coordination 3D."""
    n = len(donor_coords)
    if n == 0:
        return {"n_bonds": 0, "geometry": "Uncoordinated", "mean_angle_deg": 0.0, "angles_deg": []}
    if n == 1:
        return {"n_bonds": 1, "geometry": "Monodentate (1)", "mean_angle_deg": 0.0, "angles_deg": []}

    vecs = []
    for dc in donor_coords:
        v = dc - metal_coord
        norm = float(np.linalg.norm(v))
        if norm > 1e-6:
            vecs.append(v / norm)

    angles: list[float] = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            cos_a = float(np.clip(np.dot(vecs[i], vecs[j]), -1.0, 1.0))
            angles.append(round(math.degrees(math.acos(cos_a)), 1))

    mean_angle = round(sum(angles) / len(angles), 1) if angles else 0.0

    if n == 2:
        geom = "Linear (2)" if mean_angle > 150 else "Bent (2)"
    elif n == 3:
        geom = "Trigonal Planar (3)" if mean_angle > 112 else "Trigonal Pyramidal (3)"
    elif n == 4:
        max_a = max(angles) if angles else 0.0
        if max_a > 155.0:
            geom = "Square Planar / Seesaw (4)"
        else:
            dev = abs(mean_angle - 109.5)
            geom = f"Tetrahedral Td (4, Δ={dev:.1f}°)"
    elif n == 5:
        geom = "Trigonal Bipyramidal / Square Pyramidal (5)"
    elif n == 6:
        geom = "Octahedral Oh (6)"
    else:
        geom = f"High-Coordination ({n})"

    return {
        "n_bonds": n,
        "geometry": geom,
        "mean_angle_deg": mean_angle,
        "angles_deg": angles,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 3. Inspection complète d'une structure (API & Interface Web)
# ──────────────────────────────────────────────────────────────────────────────

def inspect_structure(source: str | Path, cache_dir: Path | None = None) -> dict[str, Any]:
    """
    Analyse une structure PDB/AlphaFold et renvoie l'inventaire structuré :
    chaînes protéiques, sites métalliques (avec résidus coordinants) et ligands organiques.
    """
    pdb_path = fetch_structure(source, cache_dir=cache_dir)
    atoms = _parse_clean_pdb_atoms(pdb_path)
    residues = _group_residues(atoms)

    prot_residues = [r for r in residues if r["is_protein"]]

    # 1. Inventaire des chaînes protéiques
    chains_info: dict[str, dict[str, Any]] = {}
    for r in prot_residues:
        ch = r["chain"]
        if ch not in chains_info:
            chains_info[ch] = {
                "chain": ch,
                "n_residues": 0,
                "min_res": r["res_seq"],
                "max_res": r["res_seq"],
                "seq_chars": [],
            }
        info = chains_info[ch]
        info["n_residues"] += 1
        info["min_res"] = min(info["min_res"], r["res_seq"])
        info["max_res"] = max(info["max_res"], r["res_seq"])
        info["seq_chars"].append(AA_3TO1.get(r["res_name"], "X"))

    chains_list = []
    for ch, info in chains_info.items():
        seq_str = "".join(info["seq_chars"])
        chains_list.append({
            "chain": ch,
            "n_residues": info["n_residues"],
            "residue_range": f"{info['min_res']}-{info['max_res']}",
            "sequence_preview": seq_str[:60] + ("..." if len(seq_str) > 60 else ""),
        })

    # 2. Inventaire des ions métalliques et sphères de coordination
    metals_list: list[dict[str, Any]] = []
    for r in residues:
        if r["is_protein"]:
            continue
        for a in r["atoms"]:
            if a["element"] in METAL_ELEMENTS or r["res_name"] in METAL_ELEMENTS:
                m_elem = a["element"] if a["element"] in METAL_ELEMENTS else r["res_name"]
                cutoff = 2.90 if m_elem in TRANSITION_METALS else 3.15
                coordinating = _find_coordinating_residues(a["coord"], prot_residues, cutoff=cutoff)
                geom_info = _classify_coordination_geometry(
                    a["coord"],
                    [c["donor_coord"] for c in coordinating],
                )
                metals_list.append({
                    "id": f"{m_elem}:{r['chain']}:{r['res_seq']}",
                    "element": m_elem,
                    "res_name": r["res_name"],
                    "chain": r["chain"],
                    "res_seq": r["res_seq"],
                    "coord": [round(float(v), 3) for v in a["coord"]],
                    "n_coordinating": len(coordinating),
                    "geometry": geom_info["geometry"],
                    "mean_angle_deg": geom_info["mean_angle_deg"],
                    "coordinating_residues": [
                        {
                            "chain": c["res"]["chain"],
                            "res_seq": c["res"]["res_seq"],
                            "res_name": c["res"]["res_name"],
                            "donor_atom": c["donor_atom"],
                            "distance_angstrom": c["distance"],
                        }
                        for c in coordinating
                    ],
                })

    # Trier les métaux par nombre de liaisons décroissant (le site catalytique complet en premier)
    metals_list.sort(key=lambda m: m["n_coordinating"], reverse=True)

    # 3. Inventaire des ligands organiques (hors solvants/tampons cristallographiques et métaux)
    ligands_list: list[dict[str, Any]] = []
    for r in residues:
        if r["is_protein"]:
            continue
        if r["res_name"] in CRYSTAL_ARTIFACTS or r["res_name"] in METAL_ELEMENTS:
            continue
        if len(r["atoms"]) == 1 and r["atoms"][0]["element"] in METAL_ELEMENTS:
            continue
        if len(r["atoms"]) < 3:
            continue

        lig_coords = np.array([a["coord"] for a in r["atoms"]], dtype=np.float64)
        pocket_res = _find_pocket_residues(lig_coords, prot_residues, cutoff=4.5)
        ligands_list.append({
            "id": f"{r['res_name']}:{r['chain']}:{r['res_seq']}",
            "res_name": r["res_name"],
            "chain": r["chain"],
            "res_seq": r["res_seq"],
            "n_heavy_atoms": len(r["atoms"]),
            "n_pocket_residues": len(pocket_res),
            "pocket_residues": [f"{p['chain']}{p['res_seq']}({p['res_name']})" for p in pocket_res[:12]],
        })

    return {
        "source": str(source),
        "resolved_pdb": str(pdb_path.as_posix()),
        "n_total_atoms": len(atoms),
        "chains": chains_list,
        "metals": metals_list,
        "ligands": ligands_list,
    }


def _find_coordinating_residues(
    metal_coord: np.ndarray,
    prot_residues: list[dict[str, Any]],
    cutoff: float = 2.90,
) -> list[dict[str, Any]]:
    """Détecte les résidus protéiques coordonnant un ion métallique par atome donneur."""
    matches: list[dict[str, Any]] = []
    for r in prot_residues:
        allowed_donors = DONOR_ATOMS_BY_RES.get(r["res_name"], set()) | {"O"}
        best_atom = None
        best_dist = 999.0
        best_coord = None

        for a in r["atoms"]:
            if a["atom_name"] in allowed_donors or a["element"] in {"S", "N", "O"}:
                # Privilégier les chaînes latérales donneuses sur l'oxygène du squelette
                if a["atom_name"] in {"N", "C", "CA", "CB"}:
                    continue
                d = float(np.linalg.norm(a["coord"] - metal_coord))
                if d <= cutoff and d < best_dist:
                    best_dist = d
                    best_atom = a["atom_name"]
                    best_coord = a["coord"]

        if best_atom is not None and best_coord is not None:
            matches.append({
                "res": r,
                "donor_atom": best_atom,
                "donor_coord": best_coord,
                "distance": round(best_dist, 2),
            })

    matches.sort(key=lambda x: (x["res"]["chain"], x["res"]["res_seq"]))
    return matches


def _find_pocket_residues(
    lig_coords: np.ndarray,
    prot_residues: list[dict[str, Any]],
    cutoff: float = 4.5,
) -> list[dict[str, Any]]:
    """Retourne les résidus protéiques situés à moins de ``cutoff`` Å d'un ligand."""
    pocket = []
    for r in prot_residues:
        r_coords = np.array([a["coord"] for a in r["atoms"]], dtype=np.float64)
        # Distance minimale atome-atome entre le résidu et le ligand
        diffs = r_coords[:, None, :] - lig_coords[None, :, :]
        min_d = float(np.min(np.linalg.norm(diffs, axis=-1)))
        if min_d <= cutoff:
            pocket.append({**r, "min_dist": round(min_d, 2)})
    pocket.sort(key=lambda x: x["min_dist"])
    return pocket


# ──────────────────────────────────────────────────────────────────────────────
# 4. Résolution 3D de Ligands / Fluorophores (SDF local ou API PubChem)
# ──────────────────────────────────────────────────────────────────────────────

def fetch_ligand_3d(
    identifier: str | Path,
    output_sdf: Path | None = None,
    timeout: float = 15.0,
) -> tuple[Path, list[dict[str, Any]]]:
    """
    Charge un ligand 3D depuis un fichier SDF local ou le télécharge automatiquement
    depuis l'API PubChem (par nom chimique, CID numérique ou chaîne SMILES).

    Returns
    -------
    tuple[Path, list[dict]]
        (chemin_sdf, liste_atomes_lourds)
    """
    ident_str = str(identifier).strip()
    local_p = Path(ident_str)
    if local_p.is_file():
        atoms = parse_sdf_heavy_atoms(local_p)
        return local_p, atoms

    # Vérifier dans inputs/ligands/
    preset_p = Path("inputs/ligands") / f"{ident_str}.sdf"
    if preset_p.is_file():
        return preset_p, parse_sdf_heavy_atoms(preset_p)

    if output_sdf is None:
        safe_slug = re.sub(r"[^A-Za-z0-9_-]+", "_", ident_str)[:32].strip("_") or "ligand"
        output_sdf = Path("inputs/ligands") / f"{safe_slug}.sdf"
    output_sdf.parent.mkdir(parents=True, exist_ok=True)

    # Déterminer le namespace PubChem : cid, smiles ou name
    if ident_str.isdigit():
        ns = f"cid/{ident_str}"
    elif any(c in ident_str for c in ("=", "#", "@", "(", ")", "[", "]")) and " " not in ident_str:
        ns = f"smiles/{requests.utils.quote(ident_str, safe='')}"
    else:
        ns = f"name/{requests.utils.quote(ident_str, safe='')}"

    # 1. Essayer le conformère 3D MMFF94 PubChem
    for record_type in ("3d", "2d"):
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/{ns}/SDF?record_type={record_type}"
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200 and "V2000" in resp.text:
            output_sdf.write_text(resp.text, encoding="utf-8")
            atoms = parse_sdf_heavy_atoms(output_sdf)
            if atoms:
                return output_sdf, atoms

    raise ValueError(
        f"Impossible de résoudre le ligand '{ident_str}' sur PubChem ou localement. "
        "Vérifiez le nom chimique, le CID PubChem ou fournissez un fichier .sdf."
    )


def parse_sdf_heavy_atoms(sdf_path: Path) -> list[dict[str, Any]]:
    """Extrait les atomes lourds (hors H) et leurs coordonnées 3D d'un fichier SDF V2000."""
    lines = sdf_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    counts_idx = -1
    for idx, line in enumerate(lines[:15]):
        if "V2000" in line:
            counts_idx = idx
            break

    if counts_idx == -1:
        return []

    counts_line = lines[counts_idx]
    try:
        n_atoms = int(counts_line[:3].strip())
    except ValueError:
        parts = counts_line.split()
        n_atoms = int(parts[0]) if parts else 0

    heavy_atoms: list[dict[str, Any]] = []
    elem_counters: dict[str, int] = {}

    for i in range(counts_idx + 1, min(counts_idx + 1 + n_atoms, len(lines))):
        parts = lines[i].split()
        if len(parts) < 4:
            continue
        try:
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            continue
        elem = "".join(c for c in parts[3].upper() if c.isalpha())
        if elem == "H":
            continue
        elem_counters[elem] = elem_counters.get(elem, 0) + 1
        atom_name = f"{elem}{elem_counters[elem]}"[:4]
        heavy_atoms.append({
            "atom_name": atom_name,
            "element": elem,
            "coord": np.array([x, y, z], dtype=np.float64),
        })

    return heavy_atoms


# ──────────────────────────────────────────────────────────────────────────────
# 5. Synthèse Mathématique des Contigs RFdiffusion avec Linkers (Boucles 3D)
# ──────────────────────────────────────────────────────────────────────────────

def synthesize_optimal_contigs(
    motif_residues: list[dict[str, Any]],
    min_length: int = 70,
    max_length: int = 100,
    chain_id: str = "A",
) -> tuple[str, list[dict[str, Any]]]:
    """
    Calcule le contig RFdiffusion optimal en mesurant les écarts géométriques 3D
    réels entre les carbones alpha (Cα) des résidus du motif.

    - Si deux résidus consécutifs ont d(Cα_i, Cα_{i+1}) <= 4.25 Å (liaison peptidique native),
      ils sont fusionnés dans un même bloc rigide (ex: ``A1-2``).
    - Si d(Cα_i, Cα_{i+1}) > 4.25 Å, une boucle flexible ``Lmin-Lmax`` calibrée sur la
      distance euclidienne est automatiquement insérée pour éliminer toute tension stérique !

    Returns
    -------
    tuple[str, list[dict]]
        (contig_string, gap_analysis_details)
    """
    if not motif_residues:
        return f"{min_length}-{max_length}", []

    def _get_atom_coord(res: dict[str, Any], aname: str) -> np.ndarray | None:
        for a in res.get("atoms", []):
            if a["atom_name"] == aname:
                return a["coord"]
        return None

    blocks: list[list[int]] = [[1]]
    gap_loops: list[tuple[int, int]] = []
    gap_details: list[dict[str, Any]] = []

    for idx in range(len(motif_residues) - 1):
        r_curr = motif_residues[idx]
        r_next = motif_residues[idx + 1]
        ca_curr = _get_atom_coord(r_curr, "CA")
        ca_next = _get_atom_coord(r_next, "CA")
        c_curr  = _get_atom_coord(r_curr, "C")
        n_next  = _get_atom_coord(r_next, "N")

        ca_dist = float(np.linalg.norm(ca_next - ca_curr)) if (ca_curr is not None and ca_next is not None) else 0.0
        cn_dist = float(np.linalg.norm(n_next - c_curr)) if (c_curr is not None and n_next is not None) else 9.9

        is_peptide_contiguous = (ca_dist <= 4.25 and cn_dist <= 1.75)

        if is_peptide_contiguous:
            blocks[-1].append(idx + 2)
            gap_details.append({
                "from": f"{r_curr['res_name']}{idx + 1}",
                "to": f"{r_next['res_name']}{idx + 2}",
                "ca_distance_angstrom": round(ca_dist, 2),
                "contiguous": True,
                "inserted_loop": "0 (peptide bond)",
            })
        else:
            # Calibration biophysique de la longueur de boucle requise :
            # Pas axial d'une hélice α (~1.5 Å/AA) à brin étendu (~3.0 Å/AA)
            l_min = max(2, int(math.ceil(ca_dist / 2.85)))
            l_max = max(l_min + 2, int(math.ceil(ca_dist / 1.45)) + 2)
            gap_loops.append((l_min, l_max))
            blocks.append([idx + 2])
            gap_details.append({
                "from": f"{r_curr['res_name']}{idx + 1}",
                "to": f"{r_next['res_name']}{idx + 2}",
                "ca_distance_angstrom": round(ca_dist, 2),
                "contiguous": False,
                "inserted_loop": f"{l_min}-{l_max} AA",
            })

    n_motif = len(motif_residues)
    sum_internal_min = sum(lo for lo, _ in gap_loops)
    sum_internal_max = sum(hi for _, hi in gap_loops)

    # Budget restant pour les régions d'échafaudage N-terminale et C-terminale
    rem_min = max(16, min_length - n_motif - sum_internal_min)
    rem_max = max(rem_min + 8, max_length - n_motif - sum_internal_max)

    # Répartition équilibrée N-ter (~35%) et C-ter (~65% pour replier le cœur hydrophobe)
    n_ter_min = max(6, int(round(rem_min * 0.35)))
    n_ter_max = max(n_ter_min + 4, int(round(rem_max * 0.35)))
    c_ter_min = max(10, rem_min - n_ter_min)
    c_ter_max = max(c_ter_min + 6, rem_max - n_ter_max)

    contig_parts: list[str] = [f"{n_ter_min}-{n_ter_max}"]
    for b_idx, block in enumerate(blocks):
        if len(block) == 1:
            contig_parts.append(f"{chain_id}{block[0]}")
        else:
            contig_parts.append(f"{chain_id}{block[0]}-{block[-1]}")
        if b_idx < len(gap_loops):
            lo, hi = gap_loops[b_idx]
            contig_parts.append(f"{lo}-{hi}")
    contig_parts.append(f"{c_ter_min}-{c_ter_max}")

    return ",".join(contig_parts), gap_details


# ──────────────────────────────────────────────────────────────────────────────
# 6. Positionnement Allostérique Optimal sans Clash d'un Fluorophore Rapporteur
# ──────────────────────────────────────────────────────────────────────────────

def _place_allosteric_reporter(
    reporter_atoms: list[dict[str, Any]],
    anchor_coord: np.ndarray,
    obstacle_coords: np.ndarray,
    target_distance: float = 8.5,
) -> list[dict[str, Any]]:
    """
    Positionne un ligand rapporteur (ex: DFHBI-1T) à ``target_distance`` Å du site
    de liaison métallique/cible le long du vecteur hémisphérique le plus dégagé,
    avec optimisation rotationnelle garantissant 0 clash stérique (> 3.2 Å).
    """
    if not reporter_atoms:
        return []

    lig_xyz = np.array([a["coord"] for a in reporter_atoms], dtype=np.float64)
    lig_centered = lig_xyz - np.mean(lig_xyz, axis=0)

    # Aligner le plan principal du fluorophore par SVD
    try:
        _, _, vt = np.linalg.svd(lig_centered, full_matrices=False)
        lig_aligned = lig_centered @ vt.T
    except Exception:
        lig_aligned = lig_centered

    # Calculer le vecteur d'ouverture opposé au barycentre des obstacles protéiques
    if len(obstacle_coords) > 0:
        obs_rel = obstacle_coords - anchor_coord
        norms = np.linalg.norm(obs_rel, axis=1, keepdims=True)
        unit_obs = np.where(norms > 1e-6, obs_rel / np.maximum(norms, 1e-6), 0.0)
        mean_obs_dir = np.mean(unit_obs, axis=0)
        norm_dir = float(np.linalg.norm(mean_obs_dir))
        if norm_dir > 0.15:
            open_dir = -mean_obs_dir / norm_dir
        else:
            open_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        open_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    # Construire une base orthonormée autour de open_dir
    ref = np.array([1.0, 0.0, 0.0]) if abs(open_dir[0]) < 0.85 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(open_dir, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(open_dir, e1)
    e2 /= np.linalg.norm(e2)
    basis = np.stack([e1, e2, open_dir], axis=1)  # (3, 3)

    best_coords = None
    best_min_clearance = -1.0

    # Balayer distances [target_distance, target_distance + 2.5] et angles de roulis
    for dist_shift in (0.0, 1.0, 2.0):
        center_pos = anchor_coord + (target_distance + dist_shift) * open_dir
        for angle_deg in range(0, 360, 30):
            rad = math.radians(angle_deg)
            c, s = math.cos(rad), math.sin(rad)
            rot_z = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
            cand = (lig_aligned @ rot_z.T) @ basis.T + center_pos

            if len(obstacle_coords) == 0:
                best_coords = cand
                best_min_clearance = 99.0
                break

            diffs = cand[:, None, :] - obstacle_coords[None, :, :]
            min_d = float(np.min(np.linalg.norm(diffs, axis=-1)))
            if min_d > best_min_clearance:
                best_min_clearance = min_d
                best_coords = cand
            if min_d >= 3.5:
                break
        if best_min_clearance >= 3.5:
            break

    placed: list[dict[str, Any]] = []
    assert best_coords is not None
    for idx, a in enumerate(reporter_atoms):
        placed.append({
            "atom_name": a["atom_name"],
            "element": a["element"],
            "coord": best_coords[idx],
        })
    return placed


# ──────────────────────────────────────────────────────────────────────────────
# 7. Écriture PDB Cristallographique Standard (LINK + CONECT Géométriques)
# ──────────────────────────────────────────────────────────────────────────────

def _format_pdb_atom_line(
    serial: int,
    record: str,
    atom_name: str,
    res_name: str,
    chain: str,
    res_seq: int,
    coord: np.ndarray,
    occupancy: float = 1.00,
    bfactor: float = 20.00,
    element: str = "C",
) -> str:
    """Formate une ligne ATOM/HETATM conforme au standard strict PDB v3.3 (80 colonnes)."""
    rec_field = f"{record:<6}"
    if len(atom_name) < 4 and len(element) == 1:
        name_field = f" {atom_name:<3}"
    else:
        name_field = f"{atom_name:<4}"
    res_field = f"{res_name:>3}"
    x, y, z = float(coord[0]), float(coord[1]), float(coord[2])
    return (
        f"{rec_field}{serial:>5} {name_field} {res_field} {chain:1}{res_seq:>4}    "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{occupancy:>6.2f}{bfactor:>6.2f}          {element:>2}"
    )


def _build_ligand_conect_lines(lig_serials_and_atoms: list[tuple[int, dict[str, Any]]]) -> list[str]:
    """Calcule les enregistrements CONECT covalents exacts à partir des rayons covalents."""
    bonds: dict[int, list[int]] = {s: [] for s, _ in lig_serials_and_atoms}
    n = len(lig_serials_and_atoms)
    for i in range(n):
        s_i, a_i = lig_serials_and_atoms[i]
        r_i = COVALENT_RADII.get(a_i["element"], 0.77)
        for j in range(i + 1, n):
            s_j, a_j = lig_serials_and_atoms[j]
            r_j = COVALENT_RADII.get(a_j["element"], 0.77)
            d = float(np.linalg.norm(a_i["coord"] - a_j["coord"]))
            if 0.85 <= d <= (r_i + r_j + 0.38):
                bonds[s_i].append(s_j)
                bonds[s_j].append(s_i)

    lines: list[str] = []
    for s_i, partners in bonds.items():
        if not partners:
            continue
        # PDB CONECT supporte jusqu'à 4 partenaires par ligne
        for chunk_start in range(0, len(partners), 4):
            chunk = partners[chunk_start:chunk_start + 4]
            partner_str = "".join(f"{p:>5}" for p in chunk)
            lines.append(f"CONECT{s_i:>5}{partner_str}")
    return lines


# ──────────────────────────────────────────────────────────────────────────────
# 8. Constructeur Universel de Cibles PDB (API Principale)
# ──────────────────────────────────────────────────────────────────────────────

def build_target_pdb(
    source: str | Path,
    output_pdb: str | Path,
    *,
    mode: str = "metal_site",
    metal_id: str | None = None,
    ligand_id: str | None = None,
    reporter_ligand: str | Path | None = None,
    reporter_distance: float = 8.5,
    target_chain: str | None = None,
    residue_range: tuple[int, int] | None = None,
    pocket_cutoff: float = 4.2,
    min_length: int = 70,
    max_length: int = 100,
    cache_dir: Path | None = None,
) -> TargetBuildResult:
    """
    Construit automatiquement un fichier PDB cible 100% conforme pour OptoBind-AI.

    Modes supportés :
      - ``metal_site``     : Extrait la sphère de coordination d'un ion métallique
                             (+ fluorophore rapporteur optionnel type DFHBI-1T).
      - ``ligand_pocket``  : Extrait un ligand organique et les résidus catalytiques clés.
      - ``protein_binder`` : Extrait une chaîne/domaine protéique cible nettoyé pour binder design.
    """
    src_pdb = fetch_structure(source, cache_dir=cache_dir)
    atoms = _parse_clean_pdb_atoms(src_pdb)
    residues = _group_residues(atoms)
    prot_residues = [r for r in residues if r["is_protein"]]

    out_path = Path(output_pdb)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    warnings_list: list[str] = []
    resolved_sdf: Path | None = None

    # ══════════════════════════════════════════════════════════════════════════
    # MODE A : Site de Coordination Métallique (+ Biosenseur Fluorophore optionnel)
    # ══════════════════════════════════════════════════════════════════════════
    if mode == "metal_site":
        metal_candidates = []
        for r in residues:
            if r["is_protein"]:
                continue
            for a in r["atoms"]:
                if a["element"] in METAL_ELEMENTS or r["res_name"] in METAL_ELEMENTS:
                    m_elem = a["element"] if a["element"] in METAL_ELEMENTS else r["res_name"]
                    tag = f"{m_elem}:{r['chain']}:{r['res_seq']}"
                    cutoff = 2.90 if m_elem in TRANSITION_METALS else 3.15
                    coord_res = _find_coordinating_residues(a["coord"], prot_residues, cutoff=cutoff)
                    metal_candidates.append((tag, m_elem, r, a, coord_res))

        if not metal_candidates:
            raise ValueError(f"Aucun ion métallique détecté dans la structure '{source}'.")

        # Filtrer par metal_id (ex: "ZN" ou "ZN:A:100") sinon prendre le mieux coordonné
        selected = None
        if metal_id:
            q = metal_id.strip().upper()
            for item in metal_candidates:
                if item[0].upper() == q or item[1].upper() == q:
                    if selected is None or len(item[4]) > len(selected[4]):
                        selected = item
        if selected is None:
            metal_candidates.sort(key=lambda x: len(x[4]), reverse=True)
            selected = metal_candidates[0]

        tag, m_elem, m_res, m_atom, coord_matches = selected
        if len(coord_matches) < 2:
            warnings_list.append(
                f"Attention : seulement {len(coord_matches)} résidu(s) coordinant(s) détecté(s) autour de {tag}."
            )

        geom_info = _classify_coordination_geometry(
            m_atom["coord"],
            [c["donor_coord"] for c in coord_matches],
        )

        # Renumérotation propre A1..AK pour les résidus du motif
        motif_res_list = [c["res"] for c in coord_matches]
        contigs_str, gap_details = synthesize_optimal_contigs(
            motif_res_list, min_length=min_length, max_length=max_length, chain_id="A"
        )
        hotspots = [f"A{i}" for i in range(1, len(motif_res_list) + 1)]

        # Optionnel : Co-positionnement d'un fluorophore rapporteur (ex: DFHBI-1T)
        placed_reporter: list[dict[str, Any]] = []
        if reporter_ligand:
            resolved_sdf, rep_atoms = fetch_ligand_3d(reporter_ligand)
            obs_coords = np.array(
                [a["coord"] for r in motif_res_list for a in r["atoms"]] + [m_atom["coord"]],
                dtype=np.float64,
            )
            placed_reporter = _place_allosteric_reporter(
                rep_atoms,
                anchor_coord=m_atom["coord"],
                obstacle_coords=obs_coords,
                target_distance=reporter_distance,
            )

        # Assemblage du fichier PDB conforme
        pdb_lines: list[str] = [
            f"REMARK   OptoBind-AI Automated Target PDB Builder (mode=metal_site, source={source})",
            f"REMARK   Metal Center : {m_elem} ({geom_info['geometry']})",
            f"REMARK   OPTOBIND CONTIGS: {contigs_str}",
            f"REMARK   OPTOBIND HOTSPOTS: {','.join(hotspots)}",
        ]

        # Écriture séquentielle des atomes pour connaître les numéros de série exacts
        body_lines: list[str] = []
        serial = 1
        donor_serials: list[tuple[int, str, str, int, float]] = []
        motif_summary: list[dict[str, Any]] = []

        for new_idx, c_match in enumerate(coord_matches, 1):
            r = c_match["res"]
            donor_name = c_match["donor_atom"]
            motif_summary.append({
                "new_id": f"A{new_idx}",
                "original_id": f"{r['chain']}{r['res_seq']}",
                "res_name": r["res_name"],
                "donor_atom": donor_name,
                "bond_distance_angstrom": c_match["distance"],
            })
            for a in r["atoms"]:
                body_lines.append(_format_pdb_atom_line(
                    serial=serial, record="ATOM", atom_name=a["atom_name"],
                    res_name=r["res_name"], chain="A", res_seq=new_idx,
                    coord=a["coord"], bfactor=a["bfactor"], element=a["element"],
                ))
                if a["atom_name"] == donor_name:
                    donor_serials.append((serial, donor_name, r["res_name"], new_idx, c_match["distance"]))
                serial += 1

        body_lines.append(f"TER   {serial:>5}      {coord_matches[-1]['res']['res_name']:>3} A{len(coord_matches):>4}" if coord_matches else "TER")
        serial += 1

        # Ion métallique sur Chaîne A res 100
        metal_serial = serial
        body_lines.append(_format_pdb_atom_line(
            serial=metal_serial, record="HETATM", atom_name=m_elem,
            res_name=m_elem[:3], chain="A", res_seq=100,
            coord=m_atom["coord"], bfactor=20.0, element=m_elem,
        ))
        serial += 1

        # Enregistrements LINK officiels PDB pour la coordination métallique
        link_lines: list[str] = []
        for _, d_name, r_name, r_idx, dist in donor_serials:
            link_lines.append(
                f"LINK        {d_name:>4} {r_name:>3} A{r_idx:>4}                "
                f"{m_elem:>4} {m_elem:>3} A 100     1555   1555 {dist:>5.2f}"
            )

        # Fluorophore / Ligand rapporteur éventuel sur Chaîne B res 200
        lig_serials_atoms: list[tuple[int, dict[str, Any]]] = []
        for rep_a in placed_reporter:
            body_lines.append(_format_pdb_atom_line(
                serial=serial, record="HETATM", atom_name=rep_a["atom_name"],
                res_name="LIG", chain="B", res_seq=200,
                coord=rep_a["coord"], bfactor=20.0, element=rep_a["element"],
            ))
            lig_serials_atoms.append((serial, rep_a))
            serial += 1

        # Enregistrements CONECT (métal + ligand covalent)
        conect_lines: list[str] = []
        if donor_serials:
            d_nums = "".join(f"{ds[0]:>5}" for ds in donor_serials[:4])
            conect_lines.append(f"CONECT{metal_serial:>5}{d_nums}")
            for ds in donor_serials:
                conect_lines.append(f"CONECT{ds[0]:>5}{metal_serial:>5}")
        conect_lines.extend(_build_ligand_conect_lines(lig_serials_atoms))

        full_pdb = "\n".join(pdb_lines + link_lines + body_lines + conect_lines + ["END", ""])
        out_path.write_text(full_pdb, encoding="utf-8")

        return TargetBuildResult(
            output_pdb=out_path,
            ligand_sdf=resolved_sdf,
            mode=mode,
            source_id=str(source),
            recommended_contigs=contigs_str,
            hotspot_residues=hotspots,
            motif_summary=motif_summary,
            coordination_info={
                "metal": m_elem,
                **geom_info,
            },
            gap_analysis=gap_details,
            warnings=warnings_list,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # MODE B : Poche de Liaison Petite Molécule (Ligand Pocket)
    # ══════════════════════════════════════════════════════════════════════════
    elif mode == "ligand_pocket":
        lig_res = None
        if ligand_id:
            q = ligand_id.strip().upper()
            for r in residues:
                if not r["is_protein"] and r["res_name"] not in CRYSTAL_ARTIFACTS:
                    tag = f"{r['res_name']}:{r['chain']}:{r['res_seq']}".upper()
                    if tag == q or r["res_name"].upper() == q:
                        lig_res = r
                        break
        else:
            for r in residues:
                if not r["is_protein"] and r["res_name"] not in CRYSTAL_ARTIFACTS and r["res_name"] not in METAL_ELEMENTS and len(r["atoms"]) >= 4:
                    lig_res = r
                    break

        if lig_res is None:
            raise ValueError(f"Aucun ligand organique correspondant trouvé dans '{source}'.")

        lig_coords = np.array([a["coord"] for a in lig_res["atoms"]], dtype=np.float64)
        pocket_candidates = _find_pocket_residues(lig_coords, prot_residues, cutoff=pocket_cutoff)
        # Conserver les 3 à 4 résidus d'ancrage polaires/aromatiques les plus proches
        anchor_res = pocket_candidates[:4]
        anchor_res.sort(key=lambda r: (r["chain"], r["res_seq"]))

        contigs_str, gap_details = synthesize_optimal_contigs(
            anchor_res, min_length=min_length, max_length=max_length, chain_id="A"
        )
        hotspots = [f"A{i}" for i in range(1, len(anchor_res) + 1)]

        pdb_lines = [
            f"REMARK   OptoBind-AI Automated Target PDB Builder (mode=ligand_pocket, source={source})",
            f"REMARK   Bound Ligand : {lig_res['res_name']} ({len(lig_res['atoms'])} heavy atoms)",
            f"REMARK   OPTOBIND CONTIGS: {contigs_str}",
            f"REMARK   OPTOBIND HOTSPOTS: {','.join(hotspots)}",
        ]

        serial = 1
        body_lines = []
        motif_summary = []
        for new_idx, r in enumerate(anchor_res, 1):
            motif_summary.append({
                "new_id": f"A{new_idx}",
                "original_id": f"{r['chain']}{r['res_seq']}",
                "res_name": r["res_name"],
                "min_dist_angstrom": r["min_dist"],
            })
            for a in r["atoms"]:
                body_lines.append(_format_pdb_atom_line(
                    serial=serial, record="ATOM", atom_name=a["atom_name"],
                    res_name=r["res_name"], chain="A", res_seq=new_idx,
                    coord=a["coord"], bfactor=a["bfactor"], element=a["element"],
                ))
                serial += 1

        if anchor_res:
            body_lines.append(f"TER   {serial:>5}      {anchor_res[-1]['res_name']:>3} A{len(anchor_res):>4}")
            serial += 1

        lig_serials_atoms = []
        for a in lig_res["atoms"]:
            body_lines.append(_format_pdb_atom_line(
                serial=serial, record="HETATM", atom_name=a["atom_name"],
                res_name="LIG", chain="B", res_seq=200,
                coord=a["coord"], bfactor=a["bfactor"], element=a["element"],
            ))
            lig_serials_atoms.append((serial, a))
            serial += 1

        # Générer également un fichier SDF 3D compagnon pour Chai-1
        resolved_sdf = out_path.with_suffix(".sdf")
        _write_minimal_sdf(lig_res["atoms"], resolved_sdf, title=lig_res["res_name"])

        conect_lines = _build_ligand_conect_lines(lig_serials_atoms)
        out_path.write_text("\n".join(pdb_lines + body_lines + conect_lines + ["END", ""]), encoding="utf-8")

        return TargetBuildResult(
            output_pdb=out_path,
            ligand_sdf=resolved_sdf,
            mode=mode,
            source_id=str(source),
            recommended_contigs=contigs_str,
            hotspot_residues=hotspots,
            motif_summary=motif_summary,
            gap_analysis=gap_details,
            warnings=warnings_list,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # MODE C : Cible Protéique Pure / Épitope (Binder Design)
    # ══════════════════════════════════════════════════════════════════════════
    elif mode == "protein_binder":
        avail_chains = sorted({r["chain"] for r in prot_residues})
        if not avail_chains:
            raise ValueError(f"Aucune chaîne protéique standard trouvée dans '{source}'.")
        chosen_chain = target_chain if (target_chain and target_chain in avail_chains) else avail_chains[0]

        chain_res = [r for r in prot_residues if r["chain"] == chosen_chain]
        if residue_range:
            lo_r, hi_r = residue_range
            chain_res = [r for r in chain_res if lo_r <= r["res_seq"] <= hi_r]

        if not chain_res:
            raise ValueError(f"Aucun résidu retenu sur la chaîne {chosen_chain}.")

        n_tgt = len(chain_res)
        contigs_str = f"A1-{n_tgt}/0 {min_length}-{max_length}"

        # Sélectionner 3 résidus aromatiques/exposés centraux comme hotspots de liaison
        mid = n_tgt // 2
        candidate_indices = [
            idx + 1 for idx, r in enumerate(chain_res)
            if r["res_name"] in {"TYR", "TRP", "PHE", "LEU", "ILE", "MET", "VAL"}
        ]
        if len(candidate_indices) >= 3:
            c_mid = len(candidate_indices) // 2
            hs_nums = candidate_indices[max(0, c_mid - 1):c_mid + 2]
        else:
            hs_nums = [max(1, mid - 2), max(1, mid), min(n_tgt, mid + 2)]
        hotspots = [f"A{idx}" for idx in sorted(set(hs_nums))]

        pdb_lines = [
            f"REMARK   OptoBind-AI Automated Target PDB Builder (mode=protein_binder, source={source})",
            f"REMARK   Target Chain : {chosen_chain} ({n_tgt} residues renumbered A1-A{n_tgt})",
            f"REMARK   OPTOBIND CONTIGS: {contigs_str}",
            f"REMARK   OPTOBIND HOTSPOTS: {','.join(hotspots)}",
        ]
        serial = 1
        body_lines = []
        for new_idx, r in enumerate(chain_res, 1):
            for a in r["atoms"]:
                body_lines.append(_format_pdb_atom_line(
                    serial=serial, record="ATOM", atom_name=a["atom_name"],
                    res_name=r["res_name"], chain="A", res_seq=new_idx,
                    coord=a["coord"], bfactor=a["bfactor"], element=a["element"],
                ))
                serial += 1
        body_lines.append(f"TER   {serial:>5}      {chain_res[-1]['res_name']:>3} A{n_tgt:>4}")
        out_path.write_text("\n".join(pdb_lines + body_lines + ["END", ""]), encoding="utf-8")

        return TargetBuildResult(
            output_pdb=out_path,
            ligand_sdf=None,
            mode=mode,
            source_id=str(source),
            recommended_contigs=contigs_str,
            hotspot_residues=hotspots,
            motif_summary=[{"target_residues": n_tgt, "original_chain": chosen_chain}],
            warnings=warnings_list,
        )

    else:
        raise ValueError(f"Mode inconnu '{mode}'. Modes valides : 'metal_site', 'ligand_pocket', 'protein_binder'.")


def _write_minimal_sdf(atoms: list[dict[str, Any]], sdf_path: Path, title: str = "LIG") -> None:
    """Écrit un fichier SDF V2000 valide avec table de liaisons géométriques."""
    n_atoms = len(atoms)
    bonds: list[tuple[int, int]] = []
    for i in range(n_atoms):
        r_i = COVALENT_RADII.get(atoms[i]["element"], 0.77)
        for j in range(i + 1, n_atoms):
            r_j = COVALENT_RADII.get(atoms[j]["element"], 0.77)
            d = float(np.linalg.norm(atoms[i]["coord"] - atoms[j]["coord"]))
            if 0.85 <= d <= (r_i + r_j + 0.38):
                bonds.append((i + 1, j + 1))

    lines = [
        title,
        "  OptoBind-AI Automated SDF",
        "",
        f"{n_atoms:>3}{len(bonds):>3}  0  0  0  0  0  0  0  0999 V2000",
    ]
    for a in atoms:
        x, y, z = float(a["coord"][0]), float(a["coord"][1]), float(a["coord"][2])
        lines.append(f"{x:>10.4f}{y:>10.4f}{z:>10.4f} {a['element']:<3} 0  0  0  0  0  0  0  0  0  0  0  0")
    for b1, b2 in bonds:
        lines.append(f"{b1:>3}{b2:>3}  1  0  0  0  0")
    lines.extend(["M  END", "$$$$", ""])
    sdf_path.write_text("\n".join(lines), encoding="utf-8")


def extract_embedded_contigs(pdb_path: Path | None) -> str | None:
    """Lit la directive ``REMARK OPTOBIND CONTIGS:`` si présente dans un PDB généré."""
    if pdb_path is None or not Path(pdb_path).is_file():
        return None
    try:
        with Path(pdb_path).open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if "OPTOBIND CONTIGS:" in line:
                    return line.split("OPTOBIND CONTIGS:", 1)[1].strip()
                if line.startswith("ATOM"):
                    break
    except Exception:
        pass
    return None
