"""
optobind_ai.utils.patcher — Application automatique des correctifs pour RFdiffusion All-Atom.

Ce module garantit que l'installation locale ou distante (Google Colab, cluster, conteneur)
de RFdiffusion dispose de tous les correctifs critiques requis par le pipeline OptoBind-AI :
1. aa_model.py : Tolérance du parsing PDB pour le ligand DFHBI-1T (filter_het).
2. util.py : Robustesse aux vecteurs colinéaires dans rigid_from_3_points (évite det(R) <= 0).
3. contigs.py : Re-fusion des listes contigs découpées par Hydra.
4. inference/utils.py & diffusion.py : Utilisation de safe_rotation_from_matrix (SVD polar decomposition).
"""

from __future__ import annotations

import glob
import logging
from pathlib import Path

logger = logging.getLogger("optobind.patcher")


def patch_filter_het(rf_dir: Path) -> bool:
    """Applique le patch filter_het dans aa_model.py."""
    aa_model_path = rf_dir / "aa_model.py"
    if not aa_model_path.is_file():
        return False

    try:
        text = aa_model_path.read_text(encoding="utf-8")
        if "Version robuste : tolère les espaces" in text:
            return True  # Déjà patché

        if "def filter_het(" not in text:
            return False

        new_filter_het = '''def filter_het(pdb_lines, ligand):
    """
    Filtre les lignes HETATM et CONECT correspondant au ligand donné.
    Version robuste : tolère les espaces et variantes de nom de résidu.
    """
    lines = []
    hetatm_ids = set()
    ligand_stripped = ligand.strip().upper()

    for l in pdb_lines:
        if not l.startswith('HETATM'):
            continue
        res_name = l[17:20].strip().upper()
        if res_name == ligand_stripped:
            lines.append(l)
            try:
                hetatm_ids.add(int(l[6:11].strip()))
            except (ValueError, IndexError):
                pass

    if not lines:
        for l in pdb_lines:
            if not l.startswith('HETATM'):
                continue
            if ligand_stripped in l[17:27].upper():
                lines.append(l)
                try:
                    hetatm_ids.add(int(l[6:11].strip()))
                except (ValueError, IndexError):
                    pass

    violations = []
    for l in pdb_lines:
        if not l.startswith('CONECT'):
            continue
        try:
            ids = [int(e) for e in l[6:].split() if e.strip().isdigit()]
        except (ValueError, IndexError):
            continue
        if not ids:
            continue
        if all(i in hetatm_ids for i in ids):
            lines.append(l)
            continue
        if any(i in hetatm_ids for i in ids):
            bonded = [i for i in ids if i in hetatm_ids]
            violations.append(
                f'CONECT line references atoms of ligand {ligand}: {bonded} AND other atoms'
            )
    if violations:
        raise Exception('\\n'.join(violations))
    return lines
'''

        parts = text.split("def filter_het(")
        remainder = parts[1]
        next_def_idx = remainder.find("\ndef ")
        func_body_end = remainder[next_def_idx:] if next_def_idx != -1 else ""
        text = parts[0] + new_filter_het + "\n" + func_body_end
        aa_model_path.write_text(text, encoding="utf-8")
        _clear_cache(rf_dir, "aa_model*")
        return True
    except Exception as exc:
        logger.warning(f"Erreur lors du patch filter_het : {exc}")
        return False


def patch_util(rf_dir: Path) -> bool:
    """Applique le patch collinearité et cross dimension dans util.py."""
    util_path = rf_dir / "util.py"
    if not util_path.is_file():
        return False

    try:
        text = util_path.read_text(encoding="utf-8")
        if "Fallback si v1 et v2 sont colin" in text:
            return True  # Déjà patché

        # Corriger make_frame torch.cross(Xn,Yn) -> torch.cross(Xn, Yn, dim=-1)
        text = text.replace("Z = torch.cross(Xn,Yn)", "Z = torch.cross(Xn, Yn, dim=-1)")

        # Remplacer rigid_from_3_points
        new_rigid = '''def rigid_from_3_points(N, Ca, C, non_ideal=False, eps=1e-8):
    #N, Ca, C - [B,L, 3]
    #R - [B,L, 3, 3], det(R)=1, inv(R) = R.T, R is a rotation matrix
    B,L = N.shape[:2]
    
    v1 = C-Ca
    v2 = N-Ca
    norm_v1 = torch.norm(v1, dim=-1, keepdim=True)
    ref_x = torch.tensor([1.0, 0.0, 0.0], device=N.device).expand_as(v1)
    e1 = torch.where(norm_v1 > 1e-4, v1/(norm_v1+eps), ref_x)

    proj = torch.einsum('bli, bli -> bl', e1, v2)[...,None]*e1
    u2 = v2 - proj
    norm_u2 = torch.norm(u2, dim=-1, keepdim=True)

    # Fallback si v1 et v2 sont colinéaires ou quasi-alignés
    ref1 = torch.tensor([1.0, 0.0, 0.0], device=N.device).expand_as(v1)
    ref2 = torch.tensor([0.0, 1.0, 0.0], device=N.device).expand_as(v1)
    ref = torch.where(torch.abs(e1[..., 0:1]) < 0.9, ref1, ref2)
    u2_fallback = ref - torch.einsum('bli, bli -> bl', e1, ref)[..., None] * e1
    u2_fallback = u2_fallback / (torch.norm(u2_fallback, dim=-1, keepdim=True) + eps)

    e2 = torch.where(norm_u2 > 1e-4, u2/(norm_u2+eps), u2_fallback)
    e2 = e2 - torch.einsum('bli, bli -> bl', e1, e2)[..., None] * e1
    e2 = e2 / (torch.norm(e2, dim=-1, keepdim=True) + eps)

    e3 = torch.cross(e1, e2, dim=-1)
    e3 = e3 / (torch.norm(e3, dim=-1, keepdim=True) + eps)

    R = torch.cat([e1[...,None], e2[...,None], e3[...,None]], axis=-1)

    det = torch.linalg.det(R)
    neg_mask = (det <= 0)[..., None, None]
    R = torch.where(neg_mask, torch.cat([e1[..., None], e2[..., None], -e3[..., None]], axis=-1), R)
    
    if non_ideal:
        v2 = v2/(torch.norm(v2, dim=-1, keepdim=True)+eps)
        cosref = torch.sum(e1*v2, dim=-1)
        costgt = cos_ideal_NCAC.item()
        cos2del = torch.clamp( cosref*costgt + torch.sqrt((1-cosref*cosref)*(1-costgt*costgt)+eps), min=-1.0, max=1.0 )
        cosdel = torch.sqrt(0.5*(1+cos2del)+eps)
        sindel = torch.sign(costgt-cosref) * torch.sqrt(1-0.5*(1+cos2del)+eps)
        Rp = torch.eye(3, device=N.device).repeat(B,L,1,1)
        Rp[:,:,0,0] = cosdel
        Rp[:,:,0,1] = -sindel
        Rp[:,:,1,0] = sindel
        Rp[:,:,1,1] = cosdel
    
        R = torch.einsum('blij,bljk->blik', R,Rp)

    return R, Ca
'''

        if "def rigid_from_3_points(" in text:
            parts = text.split("def rigid_from_3_points(")
            remainder = parts[1]
            next_def_idx = remainder.find("\ndef ")
            func_body_end = remainder[next_def_idx:] if next_def_idx != -1 else ""
            text = parts[0] + new_rigid + "\n" + func_body_end
            util_path.write_text(text, encoding="utf-8")
            _clear_cache(rf_dir, "util*")
            return True
        return False
    except Exception as exc:
        logger.warning(f"Erreur lors du patch util.py : {exc}")
        return False


def patch_contigs(rf_dir: Path) -> bool:
    """Applique le patch de re-fusion des contigs dans contigs.py."""
    contigs_path = rf_dir / "contigs.py"
    if not contigs_path.is_file():
        return False

    try:
        text = contigs_path.read_text(encoding="utf-8")
        if "fusionne en un seul bloc contig" in text:
            return True  # Déjà patché

        target = """            if isinstance(contigs, str):
                self.contigs = [contigs]
            else:
                self.contigs = contigs"""

        replacement = """            if isinstance(contigs, str):
                self.contigs = [contigs]
            else:
                # Si Hydra a découpé la chaîne sur les virgules en liste d'éléments,
                # on les re-fusionne en un seul bloc contig continu
                if len(contigs) > 1 and not any(" " in str(c) for c in contigs):
                    self.contigs = [",".join(str(c) for c in contigs)]
                else:
                    self.contigs = contigs"""

        if target in text:
            text = text.replace(target, replacement)
            contigs_path.write_text(text, encoding="utf-8")
            _clear_cache(rf_dir, "contigs*")
            return True
        return False
    except Exception as exc:
        logger.warning(f"Erreur lors du patch contigs.py : {exc}")
        return False


SAFE_ROT_CODE = '''def safe_rotation_from_matrix(R_mat):
    """
    Convertit un tableau ou tenseur (N, 3, 3) ou (3, 3) en objet scipy.spatial.transform.Rotation
    de façon 100% robuste et infaillible via décomposition polaire SVD.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation as scipy_R
    if hasattr(R_mat, "detach"):
        R_np = R_mat.detach().cpu().numpy()
    else:
        R_np = np.asarray(R_mat)

    orig_shape = R_np.shape
    if R_np.ndim == 2:
        R_np = R_np[None]

    R_clean = np.zeros_like(R_np, dtype=np.float64)
    for idx in range(len(R_np)):
        mat = R_np[idx]
        if np.isnan(mat).any() or np.isinf(mat).any() or np.all(mat == 0):
            R_clean[idx] = np.eye(3)
            continue
        try:
            U, S, Vt = np.linalg.svd(mat)
            R_cand = U @ Vt
            if np.linalg.det(R_cand) <= 0:
                U_fixed = U.copy()
                U_fixed[:, -1] *= -1
                R_cand = U_fixed @ Vt
            if np.linalg.det(R_cand) <= 1e-6:
                R_clean[idx] = np.eye(3)
            else:
                R_clean[idx] = R_cand
        except Exception:
            R_clean[idx] = np.eye(3)

    if len(orig_shape) == 2:
        return scipy_R.from_matrix(R_clean[0])
    return scipy_R.from_matrix(R_clean)
'''


def patch_inference_utils(rf_dir: Path) -> bool:
    """Applique le patch safe_rotation_from_matrix dans inference/utils.py."""
    target_path = rf_dir / "inference" / "utils.py"
    if not target_path.is_file():
        return False

    try:
        text = target_path.read_text(encoding="utf-8")
        if "def safe_rotation_from_matrix(" in text:
            return True  # Déjà patché

        text = SAFE_ROT_CODE + "\n" + text
        # Remplacements ciblés
        text = text.replace(
            "key_rots = scipy_R.from_matrix(np.stack([r_t, r_0], axis=0))",
            "key_rots = safe_rotation_from_matrix(np.stack([r_t, r_0], axis=0))",
        )
        text = text.replace(
            "key_rots = scipy_R.from_matrix(np.stack([r_t, r_t], axis=0))",
            "key_rots = safe_rotation_from_matrix(np.stack([r_t, r_t], axis=0))",
        )
        text = text.replace(
            "R_0 = scipy_R.from_matrix(rf2aa.tensor_util.assert_squeeze(R_0).numpy())",
            "R_0 = safe_rotation_from_matrix(rf2aa.tensor_util.assert_squeeze(R_0))",
        )
        text = text.replace(
            "R_t = scipy_R.from_matrix(rf2aa.tensor_util.assert_squeeze(R_t).numpy())",
            "R_t = safe_rotation_from_matrix(rf2aa.tensor_util.assert_squeeze(R_t))",
        )

        target_path.write_text(text, encoding="utf-8")
        _clear_cache(rf_dir / "inference", "utils*")
        return True
    except Exception as exc:
        logger.warning(f"Erreur lors du patch inference/utils.py : {exc}")
        return False


def patch_diffusion(rf_dir: Path) -> bool:
    """Applique le patch safe_rotation_from_matrix dans diffusion.py."""
    target_path = rf_dir / "diffusion.py"
    if not target_path.is_file():
        return False

    try:
        text = target_path.read_text(encoding="utf-8")
        if "def safe_rotation_from_matrix(" in text:
            return True  # Déjà patché

        text = SAFE_ROT_CODE + "\n" + text
        text = text.replace(
            "r_0t_rotvec = torch.tensor(scipy_R.from_matrix(\n            r_0t.cpu().numpy()).as_rotvec()).to(r_0.device)",
            "r_0t_rotvec = torch.tensor(safe_rotation_from_matrix(\n            r_0t.cpu().numpy()).as_rotvec()).to(r_0.device)",
        )
        text = text.replace(
            "R_true = scipy_R.from_matrix(R_true.squeeze())",
            "R_true = safe_rotation_from_matrix(R_true.squeeze())",
        )
        text = text.replace(
            "key_rots = scipy_R.from_matrix(np.stack([r_true, r_rand], axis=0))",
            "key_rots = safe_rotation_from_matrix(np.stack([r_true, r_rand], axis=0))",
        )
        text = text.replace(
            "key_rots = scipy_R.from_matrix(np.stack([r_true, r_true], axis=0))",
            "key_rots = safe_rotation_from_matrix(np.stack([r_true, r_true], axis=0))",
        )
        text = text.replace(
            "key_rots = scipy_R.from_matrix(np.stack([np.eye(3), np.eye(3)], axis=0))",
            "key_rots = safe_rotation_from_matrix(np.stack([np.eye(3), np.eye(3)], axis=0))",
        )

        target_path.write_text(text, encoding="utf-8")
        _clear_cache(rf_dir, "diffusion*")
        return True
    except Exception as exc:
        logger.warning(f"Erreur lors du patch diffusion.py : {exc}")
        return False


def _clear_cache(base_dir: Path, name_pattern: str) -> None:
    for pyc in glob.glob(str(base_dir / "__pycache__" / f"{name_pattern}.pyc")):
        try:
            Path(pyc).unlink()
        except Exception:
            pass


def patch_torchdata_stub() -> bool:
    """
    Injecte un stub pour torchdata.datapipes si le module est absent ou trop récent.
    Nécessaire pour DGL 2.x qui importe torchdata.datapipes.iter.IterDataPipe dans graphbolt.
    Sans ce stub, DGL lève ModuleNotFoundError au démarrage.
    """
    import sys
    import types

    # Vérifier si torchdata.datapipes.iter.IterDataPipe est accessible
    try:
        from torchdata.datapipes.iter import IterDataPipe  # noqa: F401
        return True  # Pas besoin de stub
    except (ImportError, ModuleNotFoundError):
        pass

    logger.info("Injection du stub torchdata.datapipes pour compatibilité DGL...")

    # Créer les modules manquants de façon minimale
    class IterDataPipe:
        """Stub minimal pour torchdata.datapipes.iter.IterDataPipe."""
        pass

    # torchdata
    if "torchdata" not in sys.modules:
        td = types.ModuleType("torchdata")
        sys.modules["torchdata"] = td
    else:
        td = sys.modules["torchdata"]

    # torchdata.datapipes
    tdp_name = "torchdata.datapipes"
    if tdp_name not in sys.modules:
        tdp = types.ModuleType(tdp_name)
        sys.modules[tdp_name] = tdp
        td.datapipes = tdp  # type: ignore[attr-defined]
    else:
        tdp = sys.modules[tdp_name]

    # torchdata.datapipes.iter
    tdpi_name = "torchdata.datapipes.iter"
    if tdpi_name not in sys.modules:
        tdpi = types.ModuleType(tdpi_name)
        sys.modules[tdpi_name] = tdpi
        tdp.iter = tdpi  # type: ignore[attr-defined]
    else:
        tdpi = sys.modules[tdpi_name]

    tdpi.IterDataPipe = IterDataPipe  # type: ignore[attr-defined]

    return True


def patch_fast_inference(rf_dir: Path) -> bool:
    """Active TensorFloat-32 et optimise l'I/O disque dans run_inference.py."""
    inference_path = rf_dir / "run_inference.py"
    if not inference_path.is_file():
        return False
    try:
        text = inference_path.read_text(encoding="utf-8")
        changed = False

        # Si l'ancien monkey-patch inference_mode est présent, le retirer car RoseTTAFold All-Atom
        # utilise torch.autograd.grad dans calc_chiral_grads (Track_module.py)
        if 'if hasattr(torch, "inference_mode"):' in text:
            text = text.replace('if hasattr(torch, "inference_mode"):\n    torch.no_grad = torch.inference_mode\n', '')
            changed = True

        if "torch.backends.cuda.matmul.allow_tf32" not in text and "import torch" in text:
            tf32_code = """import torch
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass
"""
            text = text.replace("import torch", tf32_code, 1)
            changed = True

        if "RFDIFFUSION_SKIP_TRAJ" not in text and "traj_prefix = " in text:
            old_traj = "traj_prefix = os.path.dirname(out_prefix)+'/traj/'+os.path.basename(out_prefix)"
            new_traj = """if os.environ.get("RFDIFFUSION_SKIP_TRAJ", "0") == "1":
        log.info(f"design : {des_path}")
        return
    traj_prefix = os.path.dirname(out_prefix)+'/traj/'+os.path.basename(out_prefix)"""
            if old_traj in text:
                text = text.replace(old_traj, new_traj, 1)
                changed = True

        if changed:
            inference_path.write_text(text, encoding="utf-8")

        # Restaurer torch.no_grad() dans model_runners.py (requis pour calc_chiral_grads)
        mr_path = rf_dir / "inference" / "model_runners.py"
        if mr_path.is_file():
            mr_text = mr_path.read_text(encoding="utf-8")
            if "with (torch.inference_mode() if hasattr(torch, 'inference_mode') else torch.no_grad()):" in mr_text:
                mr_text = mr_text.replace(
                    "with (torch.inference_mode() if hasattr(torch, 'inference_mode') else torch.no_grad()):",
                    "with torch.no_grad():"
                )
                mr_path.write_text(mr_text, encoding="utf-8")

        return True
    except Exception as exc:
        logger.warning(f"Erreur lors du patch fast_inference : {exc}")
        return False


def patch_all_rfdiffusion(rf_dir: Path | str | None = None) -> bool:
    """
    Applique tous les patches requis sur le dossier RFdiffusion spécifié ou détecté.
    """
    # Toujours patcher torchdata en premier (DGL peut en avoir besoin dès l'import)
    patch_torchdata_stub()

    if rf_dir is None:
        candidates = [
            Path("RFdiffusion"),
            Path("/content/Optobind-AI/RFdiffusion"),
            Path("/tmp/optobind_project/RFdiffusion"),
            Path(__file__).resolve().parent.parent.parent / "RFdiffusion",
        ]
        for c in candidates:
            if (c / "run_inference.py").is_file():
                rf_dir = c
                break

    if rf_dir is None:
        return False

    rf_path = Path(rf_dir).resolve()
    if not (rf_path / "run_inference.py").is_file():
        return False

    ok1 = patch_filter_het(rf_path)
    ok2 = patch_util(rf_path)
    ok3 = patch_contigs(rf_path)
    ok4 = patch_inference_utils(rf_path)
    ok5 = patch_diffusion(rf_path)
    ok6 = patch_fast_inference(rf_path)

    return all([ok1, ok2, ok3, ok4, ok5, ok6])
