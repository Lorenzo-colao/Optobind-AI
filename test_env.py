import torch
import rfdiffusion
import se3_transformer
import chai_lab
import Bio
import rdkit

print("--------------------------------------------------")
print("✅ PyTorch :", torch.__version__)
print("✅ CUDA disponible :", torch.cuda.is_available())
print("✅ RFdiffusion :", rfdiffusion.__name__)
print("✅ SE3Transformer :", se3_transformer.__name__)
print("✅ Chai-1 :", chai_lab.__name__)
print("✅ BioPython :", Bio.__version__)
print("✅ RDKit :", rdkit.__version__)
print("--------------------------------------------------")
print("🎉 TOUS LES COMPOSANTS SONT OPÉRATIONNELS !")
