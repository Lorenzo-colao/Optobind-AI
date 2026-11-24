"""
optobind_ai.web.server
======================
Serveur Web local Starlette + Uvicorn pour **OptoBind Studio** :
  - Constructeur automatisé de fichiers PDB cibles (RCSB PDB, AlphaFold DB, PubChem).
  - Pont d'exécution temps réel vers Google Colab GPU (et exécution locale WSL2 / Dry-Run).
  - Explorateur interactif des runs, métriques biophysiques 3D et optimisation ADN (Twist/IDT).
"""

from __future__ import annotations

import base64
import csv
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
import warnings
import zipfile
from typing import Any

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import requests
import yaml
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from optobind_ai.modules.biophysics import (
    analyze_cysteines,
    analyze_metal_coordination,
    ensure_metal_coordination_conect,
    optimize_dna_sequence,
)
from optobind_ai.modules.pdb_builder import (
    build_target_pdb,
    extract_embedded_contigs,
    fetch_ligand_3d,
    fetch_structure,
    inspect_structure,
)
from optobind_ai.utils.config import DEFAULT_CONFIG, load_config
from optobind_ai.utils.io_helpers import count_pdb_residues, read_fasta

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
BRIDGE_STATE_FILE = PROJECT_ROOT / "outputs" / ".colab_bridge.json"


# ──────────────────────────────────────────────────────────────────────────────
# État persistant du pont Google Colab & des jobs actifs
# ──────────────────────────────────────────────────────────────────────────────

class StudioRuntime:
    def __init__(self) -> None:
        self.colab_url: str = ""
        self.colab_token: str = ""
        self.colab_last_health: dict[str, Any] = {}
        self.jobs: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()
        self._load_bridge_state()

    def _load_bridge_state(self) -> None:
        if BRIDGE_STATE_FILE.is_file():
            try:
                data = json.loads(BRIDGE_STATE_FILE.read_text(encoding="utf-8"))
                self.colab_url = data.get("url", "")
                self.colab_token = data.get("token", "")
            except Exception:
                pass

    def save_bridge_state(self) -> None:
        BRIDGE_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        BRIDGE_STATE_FILE.write_text(
            json.dumps({"url": self.colab_url, "token": self.colab_token}, indent=2),
            encoding="utf-8",
        )

    def colab_headers(self) -> dict[str, str]:
        hdrs = {"Content-Type": "application/json"}
        if self.colab_token:
            hdrs["X-Optobind-Token"] = self.colab_token
        return hdrs

    def check_colab(self, timeout: float = 5.0) -> dict[str, Any]:
        if not self.colab_url:
            return {"connected": False, "url": ""}
        t0 = time.time()
        try:
            resp = requests.get(
                f"{self.colab_url.rstrip('/')}/api/worker/health",
                headers=self.colab_headers(),
                timeout=timeout,
            )
            latency_ms = int(round((time.time() - t0) * 1000))
            if resp.status_code == 200:
                info = resp.json()
                res = {
                    "connected": True,
                    "url": self.colab_url,
                    "latency_ms": latency_ms,
                    **info,
                }
                self.colab_last_health = res
                return res
            return {
                "connected": False,
                "url": self.colab_url,
                "error": f"HTTP {resp.status_code}: {resp.text[:160]}",
            }
        except Exception as exc:
            return {
                "connected": False,
                "url": self.colab_url,
                "error": str(exc),
            }


RUNTIME = StudioRuntime()


# ──────────────────────────────────────────────────────────────────────────────
# Indexation multi-sources des Runs (outputs/ + Output google colab/)
# ──────────────────────────────────────────────────────────────────────────────

def _discover_run_directories() -> dict[str, tuple[Path, str]]:
    """Retourne un mapping {run_id: (chemin_absolu, origine)} trié du plus récent au plus ancien."""
    found: dict[str, tuple[Path, str]] = {}

    colab_import_dir = PROJECT_ROOT / "Output google colab" / "optobind_results" / "outputs"
    if colab_import_dir.is_dir():
        for d in colab_import_dir.iterdir():
            if d.is_dir() and (d / "reports" / "report.csv").is_file():
                found[d.name] = (d, "Colab GPU")

    local_out_dir = PROJECT_ROOT / "outputs"
    if local_out_dir.is_dir():
        for d in local_out_dir.iterdir():
            if d.is_dir() and (d / "reports" / "report.csv").is_file():
                origin = "Colab Sync" if (d / ".synced_from_colab").is_file() else "Local"
                found[d.name] = (d, origin)

    return dict(sorted(found.items(), key=lambda kv: kv[1][0].stat().st_mtime, reverse=True))


def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    s = str(val).strip().rstrip("%")
    if s in ("", "N/A", "inf", "-inf", "None"):
        return default
    try:
        return float(s)
    except ValueError:
        return default


def _compute_fallback_composite(row: dict[str, Any]) -> float:
    """Calcule le score composite à la volée pour les runs antérieurs à la v0.2."""
    existing = _safe_float(row.get("composite_score"), -1.0)
    if existing > 0:
        return existing
    plddt = _safe_float(row.get("plddt_mean"), 0.0)
    if plddt <= 0:
        return 0.0
    pocket = _safe_float(row.get("plddt_pocket"), plddt)
    iptm = _safe_float(row.get("iptm"), _safe_float(row.get("ptm"), 0.0))
    rmsd = _safe_float(row.get("rmsd_angstrom"), 5.0)
    m_rmsd = _safe_float(row.get("motif_rmsd_angstrom"), rmsd)
    hbd = _safe_float(row.get("hbond_density"), 0.0)
    dsasa = _safe_float(row.get("delta_sasa"), 0.0)
    hydro = _safe_float(row.get("hydrophobic_sasa_pct"), 25.0)
    rama = _safe_float(row.get("ramachandran_outliers_pct"), 1.0)
    coil = _safe_float(row.get("coil_pct"), 30.0)
    clashes = int(_safe_float(row.get("n_steric_clashes"), 0.0))
    cys = int(_safe_float(row.get("n_unpaired_cys"), 0.0))
    liab = int(_safe_float(row.get("n_liabilities"), 0.0))

    def _cl(v: float) -> float:
        return max(0.0, min(1.0, v))

    score = (
        25.0 * _cl((plddt - 50.0) / 45.0)
        + 15.0 * _cl((pocket - 50.0) / 45.0)
        + 10.0 * _cl(iptm)
        + 12.0 * _cl(1.0 - m_rmsd / 2.5)
        + 8.0 * _cl(1.0 - rmsd / 3.5)
        + 8.0 * _cl(hbd / 0.40)
        + 7.0 * _cl(dsasa / 600.0)
        + 5.0 * _cl(1.0 - max(0.0, hydro - 20.0) / 25.0)
        + 5.0 * _cl(1.0 - rama / 4.0)
        + 5.0 * _cl(1.0 - max(0.0, coil - 25.0) / 35.0)
        - 5.0 * clashes - 10.0 * cys - 5.0 * liab
    )
    return round(max(0.0, min(100.0, score)), 1)


def _resolve_variant_pdb(run_dir: Path, variant_name: str, kind: str = "predicted") -> Path | None:
    """Localise le fichier PDB/CIF d'un variant (indépendamment des chemins Linux Colab d'origine)."""
    pdbs_dir = run_dir / "pdbs"
    if not pdbs_dir.is_dir():
        return None

    if kind == "predicted":
        v_dir = pdbs_dir / variant_name
        if v_dir.is_dir():
            for pat in ("pred.pdb", "*.pdb", "pred.model_idx_0.cif", "*.cif"):
                matches = sorted(v_dir.glob(pat))
                if matches:
                    return matches[0]
        direct = pdbs_dir / f"{variant_name}_pred.pdb"
        if direct.is_file():
            return direct

    # Fallback scaffold
    m = re.match(r"(.+_scaffold_\d+)_seq_\d+", variant_name)
    scaf_stem = m.group(1) if m else variant_name
    for cand in (
        pdbs_dir / f"{scaf_stem}.pdb",
        *sorted(pdbs_dir.glob(f"{scaf_stem}*.pdb")),
    ):
        if cand.is_file():
            return cand
    return None


def _parse_run_summary(run_id: str, run_dir: Path, origin: str) -> dict[str, Any]:
    csv_path = run_dir / "reports" / "report.csv"
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8-sig", errors="ignore") as fh:
        rows = list(csv.DictReader(fh))

    n_total = len(rows)
    validated = [r for r in rows if r.get("plddt_mean") not in (None, "", "N/A")]
    passed = [r for r in rows if r.get("passed") == "True"]
    best_plddt = max((_safe_float(r.get("plddt_mean")) for r in validated), default=0.0)
    best_score = max((_compute_fallback_composite(r) for r in validated), default=0.0)

    is_dry = False
    if validated:
        first_v = validated[0]["variant_name"]
        pred_p = _resolve_variant_pdb(run_dir, first_v, "predicted")
        if pred_p and pred_p.is_file():
            head = pred_p.read_text(encoding="utf-8", errors="ignore")[:120]
            if "DRY-RUN" in head:
                is_dry = True
        elif all(r.get("predicted_pdb") in ("None", "N/A", "") for r in validated):
            is_dry = True

    tag = "Dry-Run" if is_dry and origin == "Local" else origin
    mtime = int(run_dir.stat().st_mtime)

    return {
        "run_id": run_id,
        "origin": tag,
        "timestamp": mtime,
        "n_total": n_total,
        "n_validated": len(validated),
        "n_passed": len(passed),
        "best_plddt": round(best_plddt, 1),
        "best_score": round(best_score, 1),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints API
# ──────────────────────────────────────────────────────────────────────────────

async def homepage(request: Request) -> Response:
    index_file = STATIC_DIR / "index.html"
    return HTMLResponse(index_file.read_text(encoding="utf-8"))


async def api_workspace(request: Request) -> JSONResponse:
    targets_dir = PROJECT_ROOT / "inputs" / "targets"
    ligands_dir = PROJECT_ROOT / "inputs" / "ligands"
    targets_dir.mkdir(parents=True, exist_ok=True)
    ligands_dir.mkdir(parents=True, exist_ok=True)

    targets = []
    for p in sorted(targets_dir.glob("*.pdb")):
        targets.append({
            "name": p.name,
            "path": f"inputs/targets/{p.name}",
            "n_residues": count_pdb_residues(p),
            "embedded_contigs": extract_embedded_contigs(p),
            "size_kb": round(p.stat().st_size / 1024, 1),
        })

    ligands = []
    for p in sorted(list(ligands_dir.glob("*.sdf")) + list(ligands_dir.glob("*.mol"))):
        ligands.append({
            "name": p.name,
            "path": f"inputs/ligands/{p.name}",
            "size_kb": round(p.stat().st_size / 1024, 1),
        })

    cfg_path = PROJECT_ROOT / "inputs" / "configs" / "default.yaml"
    cfg = load_config(cfg_path) if cfg_path.is_file() else DEFAULT_CONFIG

    colab_status = RUNTIME.check_colab(timeout=3.0) if RUNTIME.colab_url else {"connected": False, "url": ""}

    return JSONResponse({
        "targets": targets,
        "ligands": ligands,
        "default_config": cfg,
        "colab": colab_status,
        "platform": sys.platform,
    })


async def api_pdb_preview(request: Request) -> JSONResponse:
    rel = request.query_params.get("path", "")
    if not rel:
        return JSONResponse({"error": "Paramètre path requis."}, status_code=400)
    cand = (PROJECT_ROOT / rel).resolve()
    if not cand.is_relative_to(PROJECT_ROOT.resolve()) or not cand.is_file():
        return JSONResponse({"error": "Fichier introuvable ou hors espace de travail."}, status_code=404)
    return JSONResponse({
        "filename": cand.name,
        "format": "cif" if cand.suffix.lower() in (".cif", ".mmcif") else "pdb",
        "pdb_text": cand.read_text(encoding="utf-8", errors="ignore"),
        "embedded_contigs": extract_embedded_contigs(cand),
    })


async def api_pdb_inspect(request: Request) -> JSONResponse:
    body = await request.json()
    source = str(body.get("source", "")).strip()
    if not source:
        return JSONResponse({"error": "Veuillez saisir un code PDB, un ID UniProt ou un fichier cible."}, status_code=400)
    try:
        cache_dir = PROJECT_ROOT / "inputs" / "targets" / "_cache"
        src_candidate = PROJECT_ROOT / source
        actual_source: str | Path = src_candidate if src_candidate.is_file() else source
        info = inspect_structure(actual_source, cache_dir=cache_dir)
        pdb_file = Path(info["resolved_pdb"])
        pdb_text = pdb_file.read_text(encoding="utf-8", errors="ignore")
        return JSONResponse({**info, "pdb_text": pdb_text})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def api_pdb_build(request: Request) -> JSONResponse:
    body = await request.json()
    source = str(body.get("source", "")).strip()
    mode = str(body.get("mode", "metal_site")).strip()
    out_name = str(body.get("output_name", "")).strip()

    if not source:
        return JSONResponse({"error": "Source structurale requise."}, status_code=400)

    if not out_name:
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(source).stem).lower()
        out_name = f"{slug}_{mode}.pdb"
    if not out_name.endswith(".pdb"):
        out_name += ".pdb"
    out_name = Path(out_name).name
    out_path = PROJECT_ROOT / "inputs" / "targets" / out_name

    res_range = None
    raw_range = str(body.get("residue_range", "")).strip()
    if raw_range and "-" in raw_range:
        try:
            a_s, b_s = raw_range.split("-", 1)
            res_range = (int(a_s.strip()), int(b_s.strip()))
        except ValueError:
            pass

    reporter = str(body.get("reporter_ligand", "")).strip() or None
    if reporter and (PROJECT_ROOT / reporter).is_file():
        reporter = str(PROJECT_ROOT / reporter)

    src_candidate = PROJECT_ROOT / source
    actual_source: str | Path = src_candidate if src_candidate.is_file() else source

    try:
        result = build_target_pdb(
            source=actual_source,
            output_pdb=out_path,
            mode=mode,
            metal_id=body.get("metal_id") or None,
            ligand_id=body.get("ligand_id") or None,
            reporter_ligand=reporter,
            reporter_distance=float(body.get("reporter_distance", 8.5)),
            target_chain=body.get("target_chain") or None,
            residue_range=res_range,
            min_length=int(body.get("min_length", 70)),
            max_length=int(body.get("max_length", 100)),
            cache_dir=PROJECT_ROOT / "inputs" / "targets" / "_cache",
        )
        rel_out = f"inputs/targets/{out_path.name}"
        rel_sdf = None
        if result.ligand_sdf and result.ligand_sdf.is_file():
            try:
                rel_sdf = result.ligand_sdf.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
            except ValueError:
                dst_sdf = PROJECT_ROOT / "inputs" / "ligands" / result.ligand_sdf.name
                shutil.copy2(result.ligand_sdf, dst_sdf)
                rel_sdf = f"inputs/ligands/{dst_sdf.name}"

        payload = result.to_dict()
        payload["output_pdb"] = rel_out
        payload["ligand_sdf"] = rel_sdf
        payload["pdb_text"] = out_path.read_text(encoding="utf-8", errors="ignore")
        return JSONResponse(payload)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def api_upload_file(request: Request) -> JSONResponse:
    body = await request.json()
    kind = body.get("kind", "target")
    fname = Path(str(body.get("filename", "uploaded.pdb"))).name
    content_b64 = body.get("content_b64", "")
    if not content_b64:
        return JSONResponse({"error": "Contenu de fichier vide."}, status_code=400)

    folder = "targets" if kind == "target" else "ligands"
    dest = PROJECT_ROOT / "inputs" / folder / fname
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(base64.b64decode(content_b64))
    if kind == "target" and dest.suffix.lower() in (".cif", ".mmcif"):
        dest = fetch_structure(dest, cache_dir=PROJECT_ROOT / "inputs" / "targets")
    return JSONResponse({
        "name": dest.name,
        "path": f"inputs/{folder}/{dest.name}",
    })


async def api_fetch_pubchem_ligand(request: Request) -> JSONResponse:
    body = await request.json()
    query = str(body.get("query", "")).strip()
    if not query:
        return JSONResponse({"error": "Nom chimique, CID ou SMILES requis."}, status_code=400)
    try:
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", query)[:32].strip("_") or "ligand"
        out_sdf = PROJECT_ROOT / "inputs" / "ligands" / f"{slug}.sdf"
        sdf_p, atoms = fetch_ligand_3d(query, output_sdf=out_sdf)
        return JSONResponse({
            "name": sdf_p.name,
            "path": f"inputs/ligands/{sdf_p.name}",
            "n_heavy_atoms": len(atoms),
        })
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


# ──────────────────────────────────────────────────────────────────────────────
# Pont Google Colab GPU
# ──────────────────────────────────────────────────────────────────────────────

async def api_colab_connect(request: Request) -> JSONResponse:
    body = await request.json()
    url = str(body.get("url", "")).strip().rstrip("/")
    token = str(body.get("token", "")).strip()
    if not url:
        return JSONResponse({"error": "URL du tunnel Colab requise."}, status_code=400)
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    RUNTIME.colab_url = url
    RUNTIME.colab_token = token
    status = RUNTIME.check_colab(timeout=8.0)
    if status.get("connected"):
        RUNTIME.save_bridge_state()
        return JSONResponse(status)
    return JSONResponse(status, status_code=502)


async def api_colab_disconnect(request: Request) -> JSONResponse:
    RUNTIME.colab_url = ""
    RUNTIME.colab_token = ""
    RUNTIME.save_bridge_state()
    return JSONResponse({"connected": False, "url": ""})


async def api_colab_status(request: Request) -> JSONResponse:
    return JSONResponse(RUNTIME.check_colab(timeout=5.0))


# ──────────────────────────────────────────────────────────────────────────────
# Lancement et Suivi des Jobs Pipeline (Colab GPU / Local WSL2 / Dry-Run)
# ──────────────────────────────────────────────────────────────────────────────

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def _strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s)


def _stage_from_log_line(line: str, cur_stage: str, cur_pct: int) -> tuple[str, int]:
    if "MODULE 1" in line or "Scaffolding" in line:
        return "scaffolding", max(cur_pct, 12)
    if "Scaffold [" in line:
        return cur_stage, min(35, cur_pct + 6)
    if "MODULE 2" in line or "Séquençage" in line:
        return "sequencing", max(cur_pct, 40)
    if "Smart Pre-Filter" in line:
        return "prefilter", max(cur_pct, 55)
    if "MODULE 3" in line or "Validation (" in line:
        return "validation", max(cur_pct, 62)
    if "pLDDT=" in line and ("PASS" in line or "FAIL" in line):
        return cur_stage, min(90, cur_pct + 8)
    if "Solubilité Protein-Sol" in line:
        return "solubility", max(cur_pct, 93)
    if "RAPPORT :" in line or "Rapport sauvegardé" in line:
        return "reporting", max(cur_pct, 97)
    if "terminé avec succès" in line:
        return "completed", 100
    return cur_stage, cur_pct


async def api_job_start(request: Request) -> JSONResponse:
    body = await request.json()
    exec_mode = str(body.get("execution_mode", "colab")).strip()  # colab | local_wsl | dry_run
    cfg_dict = body.get("config") or DEFAULT_CONFIG
    target_rel = str(body.get("target", "inputs/targets/zn2+_site.pdb"))
    ligand_rel = str(body.get("ligand", "inputs/ligands/DFHBI-1T.sdf"))
    skip_val = bool(body.get("skip_validation", False))

    target_path = (PROJECT_ROOT / target_rel).resolve()
    ligand_path = (PROJECT_ROOT / ligand_rel).resolve()
    if not target_path.is_file():
        return JSONResponse({"error": f"Fichier PDB cible introuvable : {target_rel}"}, status_code=400)
    if not ligand_path.is_file():
        return JSONResponse({"error": f"Fichier ligand introuvable : {ligand_rel}"}, status_code=400)

    run_id = body.get("run_id") or f"run_{int(time.time())}"
    run_id = re.sub(r"[^A-Za-z0-9_-]+", "_", str(run_id))

    cfg_yaml_str = yaml.safe_dump(cfg_dict, sort_keys=False, allow_unicode=True)

    # ── MODE 1 : Exécution GPU distante sur Google Colab ─────────────────────
    if exec_mode == "colab":
        if not RUNTIME.colab_url:
            return JSONResponse(
                {"error": "Aucun runtime Google Colab connecté. Connectez l'URL du Worker Colab ou choisissez le mode Local / Dry-Run."},
                status_code=400,
            )

        payload = {
            "run_id": run_id,
            "config_yaml": cfg_yaml_str,
            "target_filename": target_path.name,
            "target_b64": base64.b64encode(target_path.read_bytes()).decode("ascii"),
            "ligand_filename": ligand_path.name,
            "ligand_b64": base64.b64encode(ligand_path.read_bytes()).decode("ascii"),
            "dry_run": False,
            "skip_validation": skip_val,
        }
        try:
            r = requests.post(
                f"{RUNTIME.colab_url.rstrip('/')}/api/worker/jobs",
                headers=RUNTIME.colab_headers(),
                json=payload,
                timeout=15.0,
            )
            if r.status_code != 200:
                err_msg = r.json().get("error", r.text) if r.text else f"HTTP {r.status_code}"
                return JSONResponse({"error": f"Erreur Colab Worker : {err_msg}"}, status_code=r.status_code)
        except Exception as exc:
            return JSONResponse({"error": f"Impossible de joindre Google Colab : {exc}"}, status_code=502)

        job_state: dict[str, Any] = {
            "run_id": run_id,
            "mode": "colab",
            "status": "running",
            "stage": "initializing",
            "progress_pct": 5,
            "started_at": time.time(),
            "finished_at": None,
            "logs": [f"[COLAB BRIDGE] Job {run_id} transmis au GPU distant ({RUNTIME.colab_url})."],
            "proc": None,
        }
        RUNTIME.jobs[run_id] = job_state

        def _poll_colab_job() -> None:
            remote_offset = 0
            while True:
                try:
                    pr = requests.get(
                        f"{RUNTIME.colab_url.rstrip('/')}/api/worker/jobs/{run_id}?offset={remote_offset}",
                        headers=RUNTIME.colab_headers(),
                        timeout=10.0,
                    )
                    if pr.status_code == 200:
                        pdata = pr.json()
                        new_lines = [_strip_ansi(x) for x in pdata.get("logs", [])]
                        if new_lines:
                            job_state["logs"].extend(new_lines)
                        remote_offset = int(pdata.get("next_offset", remote_offset))
                        job_state["stage"] = pdata.get("stage", job_state["stage"])
                        job_state["progress_pct"] = pdata.get("progress_pct", job_state["progress_pct"])

                        r_status = pdata.get("status", "running")
                        if r_status in ("completed", "failed", "cancelled"):
                            if r_status == "completed":
                                job_state["logs"].append("[COLAB BRIDGE] Téléchargement automatique des structures 3D et rapports depuis Colab...")
                                _sync_colab_archive(run_id, job_state)
                            job_state["status"] = r_status
                            job_state["finished_at"] = time.time()
                            break
                except Exception as exc:
                    job_state["logs"].append(f"[COLAB BRIDGE WARN] Interruption momentanée flux : {exc}")
                time.sleep(1.2)

        threading.Thread(target=_poll_colab_job, daemon=True).start()
        return JSONResponse({"run_id": run_id, "status": "running", "mode": "colab"})

    # ── MODE 2 & 3 : Exécution Locale (WSL2 GPU/CPU ou Simulation Dry-Run) ───
    run_cfg_dir = PROJECT_ROOT / "outputs" / "_job_configs"
    run_cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_file = run_cfg_dir / f"{run_id}.yaml"
    cfg_file.write_text(cfg_yaml_str, encoding="utf-8")

    job_state = {
        "run_id": run_id,
        "mode": exec_mode,
        "status": "running",
        "stage": "initializing",
        "progress_pct": 5,
        "started_at": time.time(),
        "finished_at": None,
        "logs": [],
        "proc": None,
    }
    RUNTIME.jobs[run_id] = job_state

    def _run_local() -> None:
        is_dry = (exec_mode == "dry_run")
        rel_cfg_posix = f"outputs/_job_configs/{run_id}.yaml"

        if exec_mode == "local_wsl" and sys.platform == "win32":
            wsl_proj = "/mnt/" + PROJECT_ROOT.drive[0].lower() + PROJECT_ROOT.as_posix()[2:]
            bash_cmd = (
                f"ln -sfn '{wsl_proj}' /tmp/optobind_project && "
                f"cd /tmp/optobind_project && "
                f"PYTHONIOENCODING=utf-8 /home/colao/miniconda3/envs/optobind-ai/bin/python -u orchestrator.py "
                f"--run-id {run_id} --config '{rel_cfg_posix}' "
                f"--target '{target_rel}' --ligand '{ligand_rel}' --output-dir outputs"
            )
            if skip_val:
                bash_cmd += " --skip-validation"
            cmd = ["wsl", "-d", "Ubuntu", "-e", "bash", "-c", bash_cmd]
        else:
            # Chemins POSIX relatifs à PROJECT_ROOT pour compatibilité totale Windows natif ET redirection WSL2
            cmd = [
                sys.executable,
                "-u",
                "orchestrator.py",
                "--run-id", run_id,
                "--config", rel_cfg_posix,
                "--target", target_rel,
                "--ligand", ligand_rel,
                "--output-dir", "outputs",
            ]
            if is_dry:
                cmd.append("--dry-run")
            if skip_val:
                cmd.append("--skip-validation")

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        if is_dry:
            # Exécuter le dry-run directement en Python natif ultra-rapide sans détour WSL2 inutile
            env["OPTOBIND_NO_WSL_REDIRECT"] = "1"

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
            )
            job_state["proc"] = proc
            if proc.stdout:
                for line in iter(proc.stdout.readline, ""):
                    clean = _strip_ansi(line.rstrip("\r\n"))
                    job_state["logs"].append(clean)
                    st, pct = _stage_from_log_line(clean, job_state["stage"], job_state["progress_pct"])
                    job_state["stage"] = st
                    job_state["progress_pct"] = pct
            proc.wait()
            job_state["finished_at"] = time.time()
            if job_state["status"] != "cancelled":
                if proc.returncode == 0:
                    job_state["status"] = "completed"
                    job_state["stage"] = "completed"
                    job_state["progress_pct"] = 100
                else:
                    job_state["status"] = "failed"
                    job_state["stage"] = "failed"
        except Exception as exc:
            job_state["logs"].append(f"[LOCAL RUNNER ERROR] {exc}")
            job_state["status"] = "failed"
            job_state["finished_at"] = time.time()

    threading.Thread(target=_run_local, daemon=True).start()
    return JSONResponse({"run_id": run_id, "status": "running", "mode": exec_mode})


def _sync_colab_archive(run_id: str, job_state: dict[str, Any]) -> None:
    """Télécharge et extrait automatiquement l'archive complète du run Colab dans outputs/<run_id>/."""
    try:
        resp = requests.get(
            f"{RUNTIME.colab_url.rstrip('/')}/api/worker/jobs/{run_id}/archive",
            headers=RUNTIME.colab_headers(),
            timeout=60.0,
        )
        if resp.status_code == 200:
            dest_dir = PROJECT_ROOT / "outputs" / run_id
            dest_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(resp.content), "r") as zf:
                zf.extractall(dest_dir)
            (dest_dir / ".synced_from_colab").write_text(RUNTIME.colab_url, encoding="utf-8")
            job_state["logs"].append(f"[COLAB BRIDGE] Synchronisation terminée → outputs/{run_id}/")
        else:
            job_state["logs"].append(f"[COLAB BRIDGE ERROR] Échec téléchargement archive (HTTP {resp.status_code}).")
    except Exception as exc:
        job_state["logs"].append(f"[COLAB BRIDGE ERROR] {exc}")


async def api_job_status(request: Request) -> JSONResponse:
    run_id = request.path_params["run_id"]
    job = RUNTIME.jobs.get(run_id)
    if not job:
        return JSONResponse({"error": "Job inconnu."}, status_code=404)
    offset = int(request.query_params.get("offset", "0"))
    logs = job["logs"]
    end_t = job["finished_at"] or time.time()
    return JSONResponse({
        "run_id": run_id,
        "mode": job["mode"],
        "status": job["status"],
        "stage": job["stage"],
        "progress_pct": job["progress_pct"],
        "elapsed_sec": round(end_t - job["started_at"], 1),
        "logs": logs[offset:] if offset < len(logs) else [],
        "next_offset": len(logs),
    })


async def api_job_cancel(request: Request) -> JSONResponse:
    run_id = request.path_params["run_id"]
    job = RUNTIME.jobs.get(run_id)
    if not job:
        return JSONResponse({"error": "Job inconnu."}, status_code=404)

    job["status"] = "cancelled"
    job["stage"] = "cancelled"
    job["finished_at"] = time.time()
    if job["mode"] == "colab" and RUNTIME.colab_url:
        try:
            requests.post(
                f"{RUNTIME.colab_url.rstrip('/')}/api/worker/jobs/{run_id}/cancel",
                headers=RUNTIME.colab_headers(),
                timeout=5.0,
            )
        except Exception:
            pass
    elif job.get("proc") and job["proc"].poll() is None:
        job["proc"].terminate()
    return JSONResponse({"run_id": run_id, "status": "cancelled"})


# ──────────────────────────────────────────────────────────────────────────────
# Consultation des Runs, Structures 3D & Synthèse ADN
# ──────────────────────────────────────────────────────────────────────────────

async def api_list_runs(request: Request) -> JSONResponse:
    runs_map = _discover_run_directories()
    summaries = []
    for run_id, (r_dir, origin) in runs_map.items():
        try:
            summaries.append(_parse_run_summary(run_id, r_dir, origin))
        except Exception:
            continue
    return JSONResponse({"runs": summaries})


async def api_run_detail(request: Request) -> JSONResponse:
    run_id = request.path_params["run_id"]
    runs_map = _discover_run_directories()
    if run_id not in runs_map:
        return JSONResponse({"error": f"Run '{run_id}' introuvable."}, status_code=404)

    run_dir, origin = runs_map[run_id]
    csv_path = run_dir / "reports" / "report.csv"
    with csv_path.open("r", encoding="utf-8-sig", errors="ignore") as fh:
        raw_rows = list(csv.DictReader(fh))

    enriched: list[dict[str, Any]] = []
    for r in raw_rows:
        v_name = r.get("variant_name", "")
        seq_str = r.get("sequence", "")
        if not seq_str:
            fa_file = run_dir / "fasta" / f"{v_name}.fasta"
            if fa_file.is_file():
                entries = list(read_fasta(fa_file))
                if entries:
                    seq_str = entries[0][1]

        pred_file = _resolve_variant_pdb(run_dir, v_name, "predicted")
        scaf_file = _resolve_variant_pdb(run_dir, v_name, "scaffold")
        comp = _compute_fallback_composite(r)

        enriched.append({
            **r,
            "sequence": seq_str,
            "composite_score": f"{comp:.1f}",
            "has_predicted_pdb": pred_file is not None,
            "has_scaffold_pdb": scaf_file is not None,
        })

    enriched.sort(
        key=lambda x: (
            x.get("passed") == "True",
            _safe_float(x.get("composite_score"), -1.0),
            -_safe_float(x.get("mpnn_score"), 99.0),
        ),
        reverse=True,
    )

    run_cfg = {}
    for cfg_cand in (run_dir / "inputs" / "default.yaml", run_dir / "inputs" / "config.yaml"):
        if cfg_cand.is_file():
            try:
                run_cfg = yaml.safe_load(cfg_cand.read_text(encoding="utf-8")) or {}
                break
            except Exception:
                pass

    return JSONResponse({
        "summary": _parse_run_summary(run_id, run_dir, origin),
        "variants": enriched,
        "config": run_cfg,
    })


async def api_run_structure(request: Request) -> JSONResponse:
    run_id = request.path_params["run_id"]
    variant_name = request.path_params["variant_name"]
    kind = request.query_params.get("kind", "predicted")

    runs_map = _discover_run_directories()
    if run_id not in runs_map:
        return JSONResponse({"error": "Run introuvable."}, status_code=404)

    run_dir, _ = runs_map[run_id]
    pdb_p = _resolve_variant_pdb(run_dir, variant_name, kind=kind)
    if pdb_p is None and kind == "predicted":
        pdb_p = _resolve_variant_pdb(run_dir, variant_name, kind="scaffold")
        kind = "scaffold"

    if pdb_p is None or not pdb_p.is_file():
        return JSONResponse({"error": f"Aucune structure 3D disponible pour {variant_name}."}, status_code=404)

    if pdb_p.suffix.lower() == ".pdb":
        try:
            tgt_snap = sorted((run_dir / "inputs").glob("*.pdb"))
            ensure_metal_coordination_conect(pdb_p, target_pdb=tgt_snap[0] if tgt_snap else None)
        except Exception:
            pass

    metal_info = analyze_metal_coordination(pdb_p)
    cys_info = analyze_cysteines(pdb_p)

    return JSONResponse({
        "variant_name": variant_name,
        "kind": kind,
        "filename": pdb_p.name,
        "format": "cif" if pdb_p.suffix.lower() in (".cif", ".mmcif") else "pdb",
        "pdb_text": pdb_p.read_text(encoding="utf-8", errors="ignore"),
        "metal_coordination": metal_info,
        "cysteines": cys_info,
    })


async def api_run_download(request: Request) -> Response:
    run_id = request.path_params["run_id"]
    file_kind = request.path_params["file_kind"]
    runs_map = _discover_run_directories()
    if run_id not in runs_map:
        return JSONResponse({"error": "Run introuvable."}, status_code=404)

    run_dir, _ = runs_map[run_id]
    if file_kind == "csv":
        p = run_dir / "reports" / "report.csv"
        if p.is_file():
            return FileResponse(p, filename=f"{run_id}_report.csv")
    elif file_kind == "xlsx":
        p = run_dir / "reports" / "report.xlsx"
        if p.is_file():
            return FileResponse(p, filename=f"{run_id}_report.xlsx")
    elif file_kind == "zip":
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for fp in run_dir.rglob("*"):
                if fp.is_file():
                    zf.write(fp, arcname=fp.relative_to(run_dir).as_posix())
        return Response(
            content=buf.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{run_id}_bundle.zip"'},
        )
    return JSONResponse({"error": "Fichier demandé introuvable."}, status_code=404)


async def api_dna_optimize(request: Request) -> JSONResponse:
    body = await request.json()
    seq = str(body.get("sequence", "")).strip()
    host = str(body.get("host", "ecoli")).strip()
    add_his = bool(body.get("add_his_tev", True))
    variant_name = str(body.get("variant_name", "candidate")).strip()
    try:
        res = optimize_dna_sequence(seq, host=host, add_his_tev=add_his)
        fasta = f">{variant_name}_{res['host']}_CAI_optimized ({res['dna_length_bp']}bp, GC={res['gc_percent']}%)\n{res['dna_sequence']}\n"
        return JSONResponse({**res, "fasta": fasta})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


# ──────────────────────────────────────────────────────────────────────────────
# Application ASGI Starlette
# ──────────────────────────────────────────────────────────────────────────────

STATIC_DIR.mkdir(parents=True, exist_ok=True)

routes = [
    Route("/", homepage, methods=["GET"]),
    Route("/api/workspace", api_workspace, methods=["GET"]),
    Route("/api/pdb/preview", api_pdb_preview, methods=["GET"]),
    Route("/api/pdb/inspect", api_pdb_inspect, methods=["POST"]),
    Route("/api/pdb/build", api_pdb_build, methods=["POST"]),
    Route("/api/pdb/pubchem", api_fetch_pubchem_ligand, methods=["POST"]),
    Route("/api/files/upload", api_upload_file, methods=["POST"]),
    Route("/api/colab/connect", api_colab_connect, methods=["POST"]),
    Route("/api/colab/disconnect", api_colab_disconnect, methods=["POST"]),
    Route("/api/colab/status", api_colab_status, methods=["GET"]),
    Route("/api/jobs/start", api_job_start, methods=["POST"]),
    Route("/api/jobs/{run_id}", api_job_status, methods=["GET"]),
    Route("/api/jobs/{run_id}/cancel", api_job_cancel, methods=["POST"]),
    Route("/api/runs", api_list_runs, methods=["GET"]),
    Route("/api/runs/{run_id}", api_run_detail, methods=["GET"]),
    Route("/api/runs/{run_id}/structure/{variant_name}", api_run_structure, methods=["GET"]),
    Route("/api/runs/{run_id}/download/{file_kind}", api_run_download, methods=["GET"]),
    Route("/api/dna/optimize", api_dna_optimize, methods=["POST"]),
    Mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static"),
]

app = Starlette(debug=False, routes=routes)
