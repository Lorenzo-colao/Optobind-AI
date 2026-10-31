INSTRUCTIONS POUR LES AGENTS IA : Pipeline de Validation De Novo

Contexte : Validation in silico de protéines de novo et de biosenseurs générés par RFdiffusion / ProteinMPNN. Objectif de l'Agent : Évaluer les candidats structuraux et filtrer les séquences en appliquant strictement les seuils d'acceptation ci-dessous.

1. MÉTRIQUES DE CONFIANCE DU REPLIEMENT (FOLDING CONFIDENCE)

Évaluées via AlphaFold 2/3, Chai-1, ESMFold.

1.1. pLDDT (predicted Local Distance Difference Test)

Définition : Mesure la confiance du modèle dans le positionnement local des atomes C-alpha.

Seuils :

threshold_reject : < 70

threshold_acceptable : 70 - 80

threshold_excellent : > 80 (Cible idéale pour le design de novo)

Source : Jumper, J. et al. (2021). Highly accurate protein structure prediction with AlphaFold. Nature.

1.2. pTM (predicted Template Modeling score)

Définition : Mesure la confiance globale de la topologie de la protéine (indépendant de la longueur).

Seuils :

threshold_reject : < 0.5

threshold_accept : > 0.7

Source : Zhang, Y., & Skolnick, J. (2004). Scoring function for automated assessment of protein structure template quality. Proteins.

1.3. PAE (Predicted Aligned Error)

Définition : Erreur attendue sur la position d'un résidu y si les structures sont alignées sur le résidu x. Crucial pour vérifier que les domaines de la protéine ont une position relative fixe.

Seuils :

threshold_accept : PAE bas (< 5 Å) entre les éléments secondaires clés. Un PAE élevé entre deux domaines indique une flexibilité non désirée.

Source : Jumper, J. et al. (2021). AlphaFold2.

2. MÉTRIQUES DE DESIGNABILITÉ (SEQUENCE-STRUCTURE CONSISTENCY)

Évaluées par comparaison entre le backbone généré (RFdiffusion) et le backbone prédit (Chai-1/AF2).

2.1. scRMSD (Self-Consistency Root Mean Square Deviation)

Définition : Écart géométrique (en Ångströms) entre le squelette initial conçu et le squelette replié par l'oracle (AF2/Chai-1) à partir de la séquence ProteinMPNN.

Seuils :

threshold_reject : > 2.5 Å

threshold_accept : ≤ 2.0 Å (Considéré comme un "succès de design")

Source : Watson, J. L. et al. (2023). De novo design of protein structure and function with RFdiffusion. Nature.

3. MÉTRIQUES D'INTERFACE ET DE BINDING (POUR LE BIOSENSEUR)

Spécifiques à l'interaction entre le de novo binder, la cible (Spike) et le ligand (DFHBI-1T).

3.1. ipTM (interface predicted TM-score)

Définition : Version du pTM appliquée spécifiquement aux résidus situés à l'interface entre deux chaînes.

Seuils :

threshold_reject : < 0.6

threshold_accept : > 0.8 (Forte probabilité de liaison in vitro)

Source : Evans, R. et al. (2022). Protein complex prediction with AlphaFold-Multimer. bioRxiv.

3.2. dG_separated (Interface Binding Energy)

Définition : Estimation de l'énergie libre de liaison calculée par des champs de force physiques (ex: Rosetta).

Seuils :

threshold_accept : < -15 Rosetta Energy Units (REU). Plus c'est négatif, plus l'affinité est forte.

Outil : PyRosetta (InterfaceAnalyzer).

Source : Alford, R. F. et al. (2017). The Rosetta All-Atom Energy Function 2015. Journal of Chemical Theory and Computation.

3.3. Sc (Shape Complementarity)

Définition : Mesure à quel point les surfaces des deux protéines s'emboîtent géométriquement à l'interface (de 0 à 1).

Seuils :

threshold_accept : > 0.65 (typique des complexes anticorps-antigène natifs).

Source : Lawrence, M. C., & Colman, P. M. (1993). Shape complementarity at protein/protein interfaces. Journal of Molecular Biology.

4. MÉTRIQUES PHYSICO-CHIMIQUES ET GÉOMÉTRIQUES (FILTRES PHYSIQUES)

Essentielles pour éviter les protéines qui se replient in silico mais s'agrègent ou ne s'expriment pas in vitro.

4.1. Steric Clashes (vdw_clash / clash_score)

Définition : Chevauchement des rayons de van der Waals entre deux atomes non liés. Un modèle d'IA peut générer des atomes qui se superposent physiquement, ce qui est impossible dans la réalité.

Calcul : Somme des répulsions stériques (souvent mesuré via le terme fa_rep dans Rosetta).

Seuils :

threshold_reject : clash_score > 0 (Présence de chevauchements sévères)

threshold_accept : Proche de 0.

Source : Leaver-Fay, A. et al. (2011). ROSETTA3: an object-oriented software suite for the simulation and design of macromolecules. Methods in Enzymology.

4.2. Buns (Buried Unsatisfied Polar Atoms)

Définition : Atomes polaires (Oxygène, Azote) enfouis au cœur de la protéine (ou à l'interface) sans partenaire de liaison hydrogène. Ils déstabilisent massivement la protéine in vitro.

Seuils :

threshold_reject_core : > 2

threshold_reject_interface : > 1

Source : Boyken, S. E. et al. (2016). De novo design of protein homo-oligomers with modular hydrogen-bond network-mediated interfaces. Science.

4.3. PackStat (Packing Density / RosettaHoles)

Définition : Mesure la densité d'empaquetage du cœur hydrophobe. Les protéines naturelles sont très denses (peu de cavités vides).

Seuils :

threshold_accept : > 0.60 (Équivalent aux protéines natives).

Source : Sheffler, W., & Baker, D. (2009). RosettaHoles: rapid assessment of protein core packing for structure prediction. Protein Science.

4.4. SASA (Solvent Accessible Surface Area)

Définition : Surface de la protéine exposée au solvant. Utilisé pour cibler l'épitope (vérifier qu'il est exposé) ou pour vérifier l'enfouissement du cœur hydrophobe.

Utilisation pour le design : L'agent doit maximiser la polarité du SASA global pour assurer la solubilité, et mesurer le delta SASA (dSASA) lors du binding pour s'assurer que l'interface est large.

Seuils (Interface dSASA) : > 1000 Å² pour une interaction de haute affinité.

5. DIRECTIVES D'ÉVALUATION POUR L'AGENT (WORKFLOW LOGIQUE)

FILTRE 1 (Géométrie de base) : Rejeter immédiatement si clash_score est critique ou RMSD == inf.

FILTRE 2 (Confiance du réseau) : Rejeter si pLDDT < 75 ou pTM < 0.7.

FILTRE 3 (Cohérence du Design) : Rejeter si scRMSD > 2.0 Å.

FILTRE 4 (Affinité et Physique) : Valider l'interface via ipTM (> 0.8), Buns (le plus bas possible) et Sc (> 0.65).