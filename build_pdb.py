#!/usr/bin/env python3
"""
build_pdb.py — CLI de génération automatisée de cibles PDB pour OptoBind-AI.

Exemples :
  python build_pdb.py --inspect 1A1L
  python build_pdb.py --source 1A1L --mode metal_site --metal ZN --reporter inputs/ligands/DFHBI-1T.sdf --out inputs/targets/zn2+_1a1l_auto.pdb
  python build_pdb.py --source 2CBA --mode metal_site --metal ZN --out inputs/targets/ca2_zn_site.pdb
  python build_pdb.py --source P00918 --mode protein_binder --chain A --range 10-120 --out inputs/targets/binder_target.pdb
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from optobind_ai.modules.pdb_builder import build_target_pdb, inspect_structure


def main() -> int:
    parser = argparse.ArgumentParser(description="OptoBind-AI Automated PDB Target Builder")
    parser.add_argument("--inspect", type=str, default=None, help="Inspecter un code PDB / UniProt / fichier local et quitter.")
    parser.add_argument("--source", type=str, default=None, help="Code RCSB PDB (ex: 1A1L), ID UniProt (ex: P00918) ou chemin .pdb/.cif.")
    parser.add_argument("--mode", choices=["metal_site", "ligand_pocket", "protein_binder"], default="metal_site")
    parser.add_argument("--metal", type=str, default=None, help="Élément ou ID du métal cible (ex: ZN ou ZN:A:201).")
    parser.add_argument("--ligand", type=str, default=None, help="Code ou ID du ligand cristallographique (mode ligand_pocket).")
    parser.add_argument("--reporter", type=str, default=None, help="Chemin SDF, nom PubChem ou CID du fluorophore rapporteur (ex: DFHBI-1T).")
    parser.add_argument("--reporter-dist", type=float, default=8.5, help="Distance allostérique cible-rapporteur en Å (défaut: 8.5).")
    parser.add_argument("--chain", type=str, default=None, help="Chaîne cible (mode protein_binder).")
    parser.add_argument("--range", type=str, default=None, help="Plage de résidus ex: 10-120 (mode protein_binder).")
    parser.add_argument("--min-len", type=int, default=70, help="Longueur minimale du squelette généré.")
    parser.add_argument("--max-len", type=int, default=100, help="Longueur maximale du squelette généré.")
    parser.add_argument("--out", type=Path, default=Path("inputs/targets/auto_target.pdb"), help="Chemin du fichier PDB généré.")
    args = parser.parse_args()

    if args.inspect:
        info = inspect_structure(args.inspect)
        print(json.dumps(info, indent=2, ensure_ascii=False))
        return 0

    if not args.source:
        parser.error("Spécifiez --source <PDB/UniProt/fichier> ou --inspect <cible>.")

    res_range = None
    if args.range and "-" in args.range:
        lo_s, hi_s = args.range.split("-", 1)
        res_range = (int(lo_s), int(hi_s))

    result = build_target_pdb(
        source=args.source,
        output_pdb=args.out,
        mode=args.mode,
        metal_id=args.metal,
        ligand_id=args.ligand,
        reporter_ligand=args.reporter,
        reporter_distance=args.reporter_dist,
        target_chain=args.chain,
        residue_range=res_range,
        min_length=args.min_len,
        max_length=args.max_len,
    )
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
