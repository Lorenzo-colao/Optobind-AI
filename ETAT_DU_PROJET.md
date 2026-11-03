# État des Lieux et Architecture Complète du Projet OptoBind-AI

> **Document de référence exhaustif** — Architecture du pipeline, implémentation biophysique, état des fichiers, résultats récents et feuille de route stratégique.  
> **Date de mise à jour :** Novembre 2026  
> **Version du pipeline :** 0.1.0 (Production & Colab GPU)  
> **Auteur :** Équipe OptoBind-AI & Antigravity  

---

## 1. Identité et Vision Scientifique du Projet

### 1.1 Objectif
Concevoir *de novo* une mini-protéine fonctionnelle agissant comme un **biosenseur optogénétique réversible ON/OFF** spécifique du cation zinc divalent ($\text{Zn}^{2+}$) à **pH physiologique (7.4)**, exploitant l'activation de fluorescence du fluorophore synthétique **DFHBI-1T**.

```
       [ État OFF ]                                     [ État ON ]
Protéine apo / déstabilisée                     Poche fermée + Stéréo-verrouillage
  DFHBI-1T libre / mobile   +  Zn²⁺ (1 eq)   ──►  Pont méthine bloqué (80% enfoui)
 (Isomérisation non radiative)                    Phenolate déprotoné à pH 7.4
      NON FLUORESCENT                               FLUORESCENCE VERTE INTENSE
                                                  (λ_ex ≈ 470 nm, λ_em ≈ 505 nm)
```

### 1.2 Principes Biophysiques Fondamentaux
1. **Photophysique du fluorophore DFHBI-1T :**
   * Dans l'eau, le DFHBI-1T dissipe son énergie d'excitation par rotation ultrarapide (picoseconde) autour de son double pont méthine exocyclique ($C2-C3-C11-N2$). Il est **non fluorescent** ($\Phi_F < 0.001$).
   * Pour briller, la protéine doit agir comme un **étau conformationnel rigide** : enfouir le cœur fluorophore ($\ge 75-80\%$ de surface inaccessible au solvant) et bloquer toute liberté rotationnelle interne.
   * L'oxygène phénolique ($O1$) doit rester accessible pour être déprotoné à pH 7.4 sous forme phénolate (forme fluorescente).
2. **Coordination du Zinc ($\text{Zn}^{2+}$) :**
   * Coordination tétraédrique stricte ($T_d$) à **4 liaisons de coordination** : 2 Cystéines (thiolates $S\gamma$) et 2 Histidines ($N\delta1$ ou $N\epsilon2$).
   * Distance métal-ligand native : $\approx 2.30\text{ \AA}$ ($\text{Zn}-S$) et $\approx 2.05\text{ \AA}$ ($\text{Zn}-N$).
3. **Effet Hydrophobe & Stabilité Globulaire :**
   * Le cœur interne doit être compact et composé d'acides aminés apolaires (**Leu, Ile, Val, Phe, Met, Ala**).
   * La surface doit exposer des résidus hydrophiles et chargés pour garantir la solubilité aqueuse (seuil strict : **$\le 35\%$ de SASA hydrophobe en surface**).

---

## 2. Architecture Technique et Flux de Données du Pipeline

Le pipeline est entièrement orchestré par [orchestrator.py](file:///c:/Users/colao/Desktop/Projet%20protéine/orchestrator.py). Il fonctionne nativement sous Linux/WSL2 avec redirection automatique depuis Windows, ainsi que sur Google Colab avec accélération GPU Nvidia (A100 / T4 / L4).

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                   INPUTS DU PROJET                                     │
│  • Cible 3D : inputs/targets/zn2+_site.pdb (Zn²⁺ 4-bonds + DFHBI-1T)                   │
│  • Ligand   : inputs/ligands/DFHBI-1T.sdf (stéréo-chimie)                              │
│  • Config   : inputs/configs/default.yaml                                              │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ MODULE 1 : Scaffolding 3D — RFdiffusion All-Atom (optobind_ai/modules/scaffolding.py) │
│  • Génération du squelette carboné (N-CA-C-O) autour du motif catalytique              │
│  • Paramètres : contigs "A1-4,70-100", diffuser.T=50, noise=0.0                        │
│  • Hotspots automatiques ou manuels sur la poche de liaison                            │
│  • Sortie : Scaffold_XXX.pdb                                                           │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ MODULE 2 : Séquençage Inverse — ProteinMPNN (optobind_ai/modules/sequencing.py)        │
│  • Modèle actif : ProteinMPNN Vanilla (poids weights/ProteinMPNN_v_48_020.pt)          │
│  • Sampling temp : 0.1 | Verrouillage du motif : fix_motif = True                      │
│  • Génère N séquences d'acides aminés optimisées pour la stabilité du squelette        │
│  • Sortie : N séquences au format FASTA                                                │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ FILTRE INTERMÉDIAIRE : Smart Pre-Filter CPU 0 ms (optobind_ai/modules/biophysics.py)   │
│  • 1. Sequence Liabilities (élimination de NG, NS, DG, DS, DP, N-glyco, poly-AA)       │
│  • 2. Parité & Coordination Cys (élimination des Cys orphelines non coordonnées)       │
│  • 3. Biophysique globale : Indice GRAVY, Charge nette pH 7.4, Point Isoélectrique (pI)│
│  • 4. Seuil MPNN Score (max_mpnn_score <= 1.40) & Quota Top-K par squelette            │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ MODULE 3 : Validation Structurale — Chai-1 (optobind_ai/modules/validation.py)         │
│  • Prédiction tout-atomes multi-chaînes (Protéine + DFHBI-1T + Zn²⁺)                   │
│  • Injection automatique des restraints de coordination (liaisons de contact)          │
│  • Évaluation métrique complète : pLDDT, pTM, ipTM, RMSD squelette, RMSD motif         │
│  • Analyse géométrique : Conflits stériques (clashes), angles de Ramachandran          │
│  • Solvatation Shrake-Rupley : SASA totale, ΔSASA ligand, % surface hydrophobe         │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ MODULE 4 : Reporting & Visualisation (optobind_ai/modules/reporting.py & biophysics.py)│
│  • Annotation prédictive de solubilité E. coli via API Protein-Sol (seuil > 0.45)      │
│  • Génération du rapport tabulaire complet CSV (outputs/<run_id>/reports/report.csv)   │
│  • Reconstitution géométrique universelle des CONECT (1.15 Å <= d <= 1.75 Å)          │
│  • Visualisation 3D interactive Mol* avec Sidecar dédié                                │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Détail des Modules et Algorithmes Implémentés

### 3.1 Module 1 — Scaffolding (RFdiffusion All-Atom)
* **Fichier :** [`optobind_ai/modules/scaffolding.py`](file:///c:/Users/colao/Desktop/Projet%20protéine/optobind_ai/modules/scaffolding.py)
* **Rôle :** Génère *de novo* des squelettes de mini-protéines (longueur cible configurable : 70 à 100 résidus) qui intègrent rigidement la structure 3D du motif de coordination $\text{Zn}^{2+}$ et l'empreinte spatiale du DFHBI-1T.
* **Gestion des Contigs :** Construit les masques de diffusion pour geler les coordonnées du motif (ex. segment `A1-4`) et diffuser librement les boucles et éléments secondaires environnants (ex. `70-100` résidus).

### 3.2 Module 2 — Séquençage Inverse (ProteinMPNN Vanilla)
* **Fichier :** [`optobind_ai/modules/sequencing.py`](file:///c:/Users/colao/Desktop/Projet%20protéine/optobind_ai/modules/sequencing.py)
* **État Actuel :** Utilise **ProteinMPNN vanilla** (`ProteinMPNN_v_48_020.pt`).
  * Il est spécialisé dans le repliement thermodynamique du squelette carboné.
  * Il ne "voit" pas directement les atomes non-protéiques du DFHBI-1T ou du zinc, mais préserve les résidus clés grâce à `fix_motif: true`.
* **Échantillonnage :** Température fixée à `0.1` (très conservatrice, favorise la stabilité maximale et les cœurs hydrophobes denses).

### 3.3 Pré-filtrage Analytique CPU Universel
* **Fichier :** [`optobind_ai/modules/biophysics.py`](file:///c:/Users/colao/Desktop/Projet%20protéine/optobind_ai/modules/biophysics.py) (fonction `smart_pre_filter_sequences`)
* **Objectif :** Éliminer instantanément (0 milliseconde de calcul GPU) les séquences inviables avant de solliciter le modèle de repliement 3D lourd (Chai-1).
* **Filtres actifs :**
  1. **Liabilities chimiques :** Déamidation spontanée (`NG`, `NS`), isomérisation de l'aspartate (`DG`, `DS`), clivage acide labile (`DP`), motifs de glycosylation aberrants (`N[^P][ST]`) et homopolymères consécutifs ($\ge 4$ résidus identiques).
  2. **Vérification rigoureuse des Cystéines :** Les cystéines doivent obligatoirement être impliquées dans la coordination du zinc ou former des ponts disulfures appariés. Toute cystéine libre orpheline exposée est rejetée.
  3. **Propriétés physicochimiques (ProtParam) :** Calcul de la charge à pH 7.4, exclusion des protéines à charge nulle ou pI neutre (qui précipitent spontanément en solution).

### 3.4 Module 3 — Validation Structurale Tout-Atomes (Chai-1)
* **Fichier :** [`optobind_ai/modules/validation.py`](file:///c:/Users/colao/Desktop/Projet%20protéine/optobind_ai/modules/validation.py)
* **Modèle :** Chai-1 (avec fallback optionnel sur AlphaFold 3 / ESMFold).
* **Restraints de coordination :** Injecte automatiquement des contraintes de distance pour forcer l'évaluation du complexe ternaire : `Protéine + DFHBI-1T + Zn²⁺`.
* **Métriques calculées :**
  * `pLDDT_mean` : Confiance locale moyenne du repliement (seuil de succès $\ge 80.0$).
  * `RMSD_motif` : Déviation quadratique moyenne du site catalytique par rapport au motif cible cristallographique (seuil $\le 1.50\text{ \AA}$).
  * `SASA` et `hydrophobic_sasa_pct` : Surface accessible au solvant mesurée par Shrake-Rupley. Seuil de sécurité : surface extérieure hydrophobe $\le 35\%$.
  * `Clashes` : Nombre de contacts stériques interatomiques anormaux ($d < 0.6 \times \sum r_{\text{vdW}}$).

### 3.5 Module 4 — Post-Traitement, Solubilité & Visualisation
* **Annotation de solubilité :** Les 3 meilleurs candidats sont automatiquement interrogés sur l'API **Protein-Sol** (`scaled_sol > 0.45` pour garantir une expression soluble dans *E. coli* BL21).
* **Générateur Universel de `CONECT` :** Reconstitue les 23 liaisons covalentes du fluorophore DFHBI-1T et les liaisons métal-ligand directement dans les fichiers PDB de sortie en inspectant les distances géométriques réelles ($1.15 \le d \le 1.75\text{ \AA}$).
* **Visualiseur Mol* :** Intégration interactive directe pour inspecter les distances de coordination, l'enfouissement du fluorophore et les contacts de surface.

---

## 4. Inventaire des Fichiers Clés et Données d'Entrée

| Emplacement | Type | Description | État |
| :--- | :--- | :--- | :--- |
| [`inputs/targets/zn2+_site.pdb`](file:///c:/Users/colao/Desktop/Projet%20protéine/inputs/targets/zn2+_site.pdb) | PDB | Site modèle de $\text{Zn}^{2+}$ tétraédrique (dérivé de PDB 1A1L) avec 2 Cys, 2 His, ion $\text{Zn}^{2+}$, ligand DFHBI-1T et fiches `LINK`/`CONECT`. | Validé (4 liaisons $\text{Zn}^{2+}$ strictes) |
| [`inputs/ligands/DFHBI-1T.sdf`](file:///c:/Users/colao/Desktop/Projet%20protéine/inputs/ligands/DFHBI-1T.sdf) | SDF | Structure 3D tout-atomes et connectivité chimique du fluorophore DFHBI-1T. | Validé |
| [`inputs/configs/default.yaml`](file:///c:/Users/colao/Desktop/Projet%20protéine/inputs/configs/default.yaml) | YAML | Paramètres complets d'exécution pour RFdiffusion, ProteinMPNN, Chai-1 et biophysique. | Opérationnel |
| [`orchestrator.py`](file:///c:/Users/colao/Desktop/Projet%20protéine/orchestrator.py) | Python | Script maître orchestrant l'ensemble du pipeline avec gestion multi-plateformes (Windows/WSL2/Linux). | Opérationnel |
| [`optobind_ai/modules/biophysics.py`](file:///c:/Users/colao/Desktop/Projet%20protéine/optobind_ai/modules/biophysics.py) | Python | Moteur analytique biophysique : pré-filtres, SASA, Cys, calculs de charges, générateur `CONECT`. | Opérationnel |
| [`optobind_ai/modules/validation.py`](file:///c:/Users/colao/Desktop/Projet%20protéine/optobind_ai/modules/validation.py) | Python | Moteur d'inférence Chai-1, génération des restraints, calcul des métriques structurales. | Opérationnel |
| [`OptoBind_AI_Colab.ipynb`](file:///c:/Users/colao/Desktop/Projet%20protéine/OptoBind_AI_Colab.ipynb) | Notebook | Notebook complet pour exécution cloud GPU avec montage Google Drive et dépendances. | Opérationnel |
| [`GUIDE_METRIQUES_VALIDATION.md`](file:///c:/Users/colao/Desktop/Projet%20protéine/GUIDE_METRIQUES_VALIDATION.md) | Markdown | Guide de référence détaillé des 18 métriques de sélection *in silico*. | Documenté |

---

## 5. Analyse des Résultats Récents (*Run 1793495836*)

L'analyse du run GPU sur Colab (`run_1793495836`) a permis d'isoler un candidat phare et de poser un diagnostic précis sur la géométrie :

### 5.1 Profil du Meilleur Candidat (`scaffold_001_seq_04`)
* **Repliement global :** Structure très compacte en tonneau / sandwich $\beta$ amphipathique avec hélices d'ancrage.
* **Confiance structurale :** $\text{pLDDT} = 84.5$ (qualité de repliement élevée).
* **Enfouissement du fluorophore :** Le DFHBI-1T est enfoui à **$\approx 80\%$** dans une poche hydrophobe interne. Le pont méthine ($C2-C3-C11-N2$) a une surface accessible nulle ($\text{SASA} = 0.0\text{ \AA}^2$), ce qui bloque physiquement la rotation non radiative et permet l'activation de la fluorescence.
* **Coordination du Zinc :** **3 liaisons observées sur 4** (2 Cys et 1 His coordonnées ; la seconde His est positionnée en retrait à $\approx 4.2\text{ \AA}$).

### 5.2 Diagnostic de la Liaison Zinc Manquante
* **Cause racine :** La configuration actuelle du contig RFdiffusion utilisait la notation `contigs: "A1-4,70-100"`.
* Cette notation force les 4 résidus du site catalytique (Cys1, Cys2, His3, His4) à être strictement consécutifs dans la séquence primaire ($C\alpha-C\alpha \approx 3.8\text{ \AA}$).
* Or, dans la réalité cristallographique d'un site tétraédrique (PDB 1A1L), les $C\alpha$ de Cys2 et His3 sont distants de **$10.64\text{ \AA}$** dans l'espace 3D, ce qui nécessite une boucle intermédiaire (loop d'au moins 4 à 6 acides aminés) pour que la chaîne peptidique puisse faire demi-tour sans distorsion stérique.

---

## 6. Problèmes Résolus Récemment

1. **Reconstitution des liaisons DFHBI-1T dans Mol* :**
   * *Problème :* L'ancien système réinjectait les lignes `CONECT` avec un décalage d'index rigide (+550), ce qui créait des liaisons géométriquement aberrantes ($d > 3.5\text{ \AA}$) et masquait le ligand dans Mol*.
   * *Solution :* Remplacement par un algorithme géométrique universel dans `biophysics.py` qui recalcule les liaisons réelles à partir des distances euclidiennes interatomiques standard ($1.15 \le d \le 1.75\text{ \AA}$).
2. **Détection des atomes métalliques de Chai-1 :**
   * *Problème :* Chai-1 nommait l'atome de zinc `ZN1_` dans un résidu étiqueté `LIG`, échouant au filtre d'égalité stricte `name == "ZN"`.
   * *Solution :* Prise en compte du symbole de l'élément chimique et recherche par préfixe sur la table périodique des métaux de transition.
3. **PDB de Cible Tétraédrique Strict :**
   * Remplacement du site initial par `inputs/targets/zn2+_site.pdb` calqué sur 1A1L, avec 4 fiches `LINK` explicites et connectivité vérifiée.

---

## 7. Tableau Comparatif : ProteinMPNN vs LigandMPNN

| Paramètre | **ProteinMPNN Vanilla (Actuel)** | **LigandMPNN (Évolution Prévue)** |
| :--- | :--- | :--- |
| **Entrées vues par le réseau** | Coordonnées du squelette protéique ($N, C\alpha, C, O$). | Squelette protéique **+ Atomes du ligand DFHBI-1T + Ion $\text{Zn}^{2+}$**. |
| **Prise en compte du DFHBI-1T** | **Aveugle.** Il remplit la poche d'après la courbure du backbone sans voir les atomes chimiques du ligand. | **Directe et explicite.** Il place des chaînes latérales formant des ponts H et du packing $\pi$-$\pi$ sur les atomes du ligand. |
| **Risques de mutations** | Risque d'insérer des résidus encombrants ou mal orientés dans la poche. | Poche sculptée sur-mesure pour accueillir le ligand sans conflit stérique. |
| **Taux de succès au laboratoire** | ~5 à 15 % de designs actifs *in vitro*. | **> 50 à 70 %** de designs actifs *in vitro* (littérature David Baker 2023). |

---

## 8. Feuille de Route et Prochaines Actions

1. **Correction des Contigs RFdiffusion pour le Site Zinc :**
   * Remplacer le segment contigu `A1-4` par une déclaration segmentée intégrant des boucles d'espacement (ex. `A1-2,4-6,A3-4` ou contigs modulaires) pour respecter l'écart 3D de $10.64\text{ \AA}$ entre Cys2 et His3 et garantir **4/4 liaisons tétraédriques parfaites**.
2. **Intégration de LigandMPNN dans le Module 2 :**
   * Télécharger les poids légers `ligandmpnn_v_32_010_25.pt` (~150 Mo).
   * Implémenter la détection automatique : si la cible contient des `HETATM` (ligand ou métal) $\rightarrow$ routage automatique vers LigandMPNN.
3. **Automatisation de la Génération de Sessions Mol* :**
   * Produire des scripts d'état ou fichiers de session Mol* pour ouvrir en un clic les candidats avec les stick-representations du DFHBI-1T et du zinc déjà colorés et étiquetés.
4. **Validation Expérimentale (Wet-Lab) :**
   * Sélectionner les 3 meilleurs candidats présentant $\text{pLDDT} \ge 85$, $\text{RMSD\_motif} \le 1.2\text{ \AA}$, 4 liaisons $\text{Zn}^{2+}$, et $\text{Protein-Sol} \ge 0.50$ pour synthèse de gène et expression en système bactérien.
