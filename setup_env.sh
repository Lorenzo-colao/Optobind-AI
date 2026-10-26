#!/usr/bin/env bash
# =============================================================================
# setup_env.sh — Initialisation de l'environnement Conda OptoBind-AI
# =============================================================================
# Usage (depuis la racine du projet, sous WSL2 ou Linux) :
#   chmod +x setup_env.sh && ./setup_env.sh
# =============================================================================

set -euo pipefail

ENV_NAME="optobind-ai"
ENV_FILE="envs/environment.yaml"
WEIGHTS_DIR="weights"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║       OptoBind-AI — Installation de l'environnement         ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# ── 1. Vérification de conda / mamba ────────────────────────────────────────
# Tenter de sourcer conda si non présent dans le PATH
if ! command -v conda &>/dev/null; then
    for conda_candidate in "$HOME/miniconda3" "$HOME/miniforge3" "$HOME/anaconda3" "/home/colao/miniconda3"; do
        if [ -f "$conda_candidate/etc/profile.d/conda.sh" ]; then
            source "$conda_candidate/etc/profile.d/conda.sh"
            export PATH="$conda_candidate/bin:$PATH"
            break
        fi
    done
fi

if command -v mamba &>/dev/null; then
    CONDA_CMD="mamba"
    echo "  ✓ Mamba détecté"
elif command -v conda &>/dev/null; then
    CONDA_CMD="conda"
    echo "  ⚠ Mamba non trouvé — utilisation de conda"
else
    echo "  ✗ conda/mamba introuvable. Installez Miniforge ou Anaconda."
    exit 1
fi

# ── 2. Cloner les dépôts requis si absents ─────────────────────────────────
echo ""
echo "  [1/5] Vérification des sous-modules RFdiffusion & ProteinMPNN..."
if [ ! -d "RFdiffusion" ]; then
    echo "  -> Clonage de RFdiffusion..."
    git clone https://github.com/RosettaCommons/RFdiffusion.git
fi

if [ ! -d "ProteinMPNN" ]; then
    echo "  -> Clonage de ProteinMPNN..."
    git clone https://github.com/dauparas/ProteinMPNN.git
fi
echo "  ✓ RFdiffusion et ProteinMPNN prêts."

# ── 3. Création de l'environnement ──────────────────────────────────────────
echo ""
echo "  [2/5] Création / mise à jour de l'environnement '$ENV_NAME'..."
$CONDA_CMD env create -f "$ENV_FILE" -n "$ENV_NAME" --yes 2>/dev/null || \
$CONDA_CMD env update -f "$ENV_FILE" -n "$ENV_NAME" --prune

# Installation des dépendances locales pip spécifiques
echo "  [+] Installation des modules RFdiffusion & SE3Transformer..."
if [ -d "RFdiffusion/env/SE3Transformer" ]; then
    (cd RFdiffusion/env/SE3Transformer && pip install --no-cache-dir -r requirements.txt && python setup.py install)
    (cd RFdiffusion && pip install -e .)
fi

echo "  ✓ Environnement '$ENV_NAME' prêt."

# ── 4. Téléchargement des poids des modèles ─────────────────────────────────
echo ""
echo "  [3/5] Vérification des poids dans $WEIGHTS_DIR/..."
mkdir -p "$WEIGHTS_DIR/chai-1"
mkdir -p "$WEIGHTS_DIR/alphafold3"

if [ ! -f "$WEIGHTS_DIR/RFDiffusionAA_paper_weights.pt" ]; then
    echo "  -> Téléchargement de RFDiffusionAA_paper_weights.pt (~1.3 Go)..."
    wget -c -P "$WEIGHTS_DIR/" http://files.ipd.uw.edu/pub/RF-All-Atom/weights/RFDiffusionAA_paper_weights.pt || \
    echo "  ⚠ Téléchargement échoué, téléchargez manuellement avec wget."
else
    echo "  ✓ RFDiffusionAA_paper_weights.pt déjà présent."
fi

if [ ! -f "$WEIGHTS_DIR/ProteinMPNN_v_48_020.pt" ]; then
    echo "  -> Téléchargement de ProteinMPNN_v_48_020.pt..."
    wget -c -O "$WEIGHTS_DIR/ProteinMPNN_v_48_020.pt" https://raw.githubusercontent.com/dauparas/ProteinMPNN/main/vanilla_model_weights/v_48_020.pt || \
    echo "  ⚠ Téléchargement échoué, téléchargez manuellement avec wget."
else
    echo "  ✓ ProteinMPNN_v_48_020.pt déjà présent."
fi

# ── 5. Création du fichier .env local ───────────────────────────────────────
echo ""
echo "  [4/5] Création de .env (variables d'environnement locales)..."
cat > .env << 'EOF'
# OptoBind-AI — Variables d'environnement locales
# Source ce fichier avant de lancer le pipeline :
#   source .env

export OPTOBIND_ENV="optobind-ai"
export OPTOBIND_WEIGHTS="./weights"
export OPTOBIND_CUDA_DEVICE="0"          # ID de la GPU à utiliser
export CHAI_WEIGHTS_DIR="./weights/chai-1"
export PYTHONPATH="$PYTHONPATH:$(pwd)"
EOF
echo "  ✓ .env créé."

# ── 5. Test rapide ───────────────────────────────────────────────────────────
echo ""
echo "  [4/4] Test de l'import du package optobind_ai..."
conda run -n "$ENV_NAME" python -c "import optobind_ai; print('  ✓ optobind_ai importé avec succès.')" \
    || echo "  ⚠ Import échoué — vérifiez le PYTHONPATH."

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Installation terminée !                                     ║"
echo "║  Activez l'environnement :                                   ║"
echo "║    conda activate optobind-ai                                ║"
echo "║  Lancez un test rapide :                                     ║"
echo "║    python orchestrator.py --target inputs/targets/zn.pdb \  ║"
echo "║      --ligand inputs/ligands/DFHBI-1T.sdf --dry-run         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
