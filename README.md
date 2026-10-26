# OptoBind-AI 🔬

> Pipeline de design de novo de biosenseurs optiques — Zn²⁺ / DFHBI-1T / pH 7.4

## Vue d'ensemble

OptoBind-AI orchestre trois outils d'IA structurale pour concevoir des mini-protéines agissant comme biosenseurs optiques ON/OFF :

```
RFdiffusion All-Atom → ProteinMPNN → AlphaFold 3 / Chai-1
       (Squelette)         (Séquence)       (Validation)
```

## Installation rapide (WSL2 / Linux)

```bash
# 1. Cloner et entrer dans le projet
cd "Projet protéine"

# 2. Installer l'environnement Conda (nécessite Mamba ou Conda)
chmod +x setup_env.sh && ./setup_env.sh

# 3. Activer l'environnement
conda activate optobind-ai

# 4. Tester sans GPU (dry-run)
python orchestrator.py \
  --target inputs/targets/zn2+_site.pdb \
  --ligand inputs/ligands/DFHBI-1T.sdf  \
  --dry-run --verbose
```

## Utilisation

```bash
python orchestrator.py \
  --target  inputs/targets/zn2+_site.pdb \
  --ligand  inputs/ligands/DFHBI-1T.sdf  \
  --config  inputs/configs/default.yaml  \
  --output-dir outputs/
```

### Options principales

| Option | Défaut | Description |
|---|---|---|
| `--target` | — | PDB de la cible (Zn²⁺) |
| `--ligand` | — | SDF/PDB du fluorophore (DFHBI-1T) |
| `--config` | `inputs/configs/default.yaml` | Fichier YAML de configuration |
| `--skip-validation` | False | Ignorer l'étape AlphaFold |
| `--dry-run` | False | Mode test sans GPU |
| `--verbose` | False | Logs DEBUG |

## Structure du projet

```
Projet protéine/
├── orchestrator.py              ← Script principal d'orchestration
├── setup_env.sh                 ← Script d'installation de l'environnement
├── envs/
│   └── environment.yaml         ← Dépendances Conda
├── inputs/
│   ├── targets/                 ← PDB de la cible (Zn²⁺)
│   ├── ligands/                 ← SDF/PDB du fluorophore (DFHBI-1T)
│   └── configs/
│       └── default.yaml         ← Configuration du pipeline
├── outputs/
│   └── <run_id>/
│       ├── pdbs/                ← Squelettes + complexes prédits
│       ├── fasta/               ← Séquences optimisées
│       ├── reports/report.csv   ← Rapport de validation in silico
│       └── inputs/              ← Snapshot des inputs (reproductibilité)
├── optobind_ai/
│   ├── modules/
│   │   ├── scaffolding.py       ← Module 1 : RFdiffusion All-Atom
│   │   ├── sequencing.py        ← Module 2 : ProteinMPNN
│   │   ├── validation.py        ← Module 3 : AlphaFold 3 / Chai-1
│   │   └── reporting.py         ← Génération du rapport CSV
│   └── utils/
│       ├── config.py            ← Chargement/validation YAML
│       ├── logger.py            ← Système de logs (console + fichier)
│       └── io_helpers.py        ← IO, dataclasses (Scaffold, DesignedSequence…)
├── tests/
│   └── test_pipeline.py         ← Tests unitaires (pytest, mode dry_run)
└── weights/                     ← Poids des modèles (à télécharger manuellement)
    ├── RFdiffusion_AllAtom_v1.pt
    ├── ProteinMPNN_v_48_020.pt
    └── chai-1/
```

## Critères de validation in silico

Un variant est qualifié pour une production expérimentale si :

| Critère | Seuil |
|---|---|
| RMSD Cα (RFdiffusion ↔ AlphaFold) | **< 2.0 Å** |
| pLDDT moyen | **> 90** |
| Clashes stériques | **= 0** |

## Rapport de sortie (`report.csv`)

| Colonne | Description |
|---|---|
| `variant_name` | Identifiant unique du variant |
| `plddt_mean` | Score pLDDT global d'AlphaFold |
| `plddt_pocket` | pLDDT de la poche de liaison |
| `rmsd_angstrom` | RMSD Cα en Å |
| `n_steric_clashes` | Nombre de clashes stériques |
| `passed` | True si le variant passe tous les critères |

## Lancer les tests

```bash
pytest tests/ -v
```

## Backends de validation supportés

| Backend | Config | Notes |
|---|---|---|
| **Chai-1** *(défaut)* | `alphafold.backend: "chai-1"` | Open-source, local, complexes ternaires |
| **ESMFold** | `alphafold.backend: "esmfold"` | Rapide, sans ligand |
| **AlphaFold 3** | `alphafold.backend: "alphafold3"` | Meilleure précision, accès restreint |

## Références

- Watson et al. (2023). *De novo design of protein structure and function with RFdiffusion.* Nature.
- Dauparas et al. (2022). *Robust deep learning based protein sequence design.* Science.
- Abramson et al. (2024). *Accurate structure prediction of biomolecular interactions with AlphaFold 3.* Nature.
- Chai Discovery (2024). *Chai-1.* https://github.com/chaidiscovery/chai-lab
