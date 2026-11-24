#!/usr/bin/env python3
"""
optobind_ai.colab_worker
========================
Démon Worker HTTP distant léger (100% bibliothèque standard Python, zéro dépendance tierce)
destiné à tourner sur Google Colab (ou un serveur GPU Linux distant) pour exécuter
le pipeline OptoBind-AI piloté depuis l'interface Web locale.

Lancement sur Google Colab :
  /content/optobind_env/bin/python -m optobind_ai.colab_worker --port 8000 --cloudflare
"""

from __future__ import annotations

import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
from urllib.parse import parse_qs, urlparse
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class JobRecord:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.status = "starting"  # starting | running | completed | failed | cancelled
        self.stage = "initializing"
        self.progress_pct = 2
        self.started_at = time.time()
        self.finished_at: float | None = None
        self.returncode: int | None = None
        self.logs: list[str] = []
        self.proc: subprocess.Popen | None = None
        self.lock = threading.Lock()

    def append_log(self, line: str) -> None:
        clean = line.rstrip("\r\n")
        with self.lock:
            self.logs.append(clean)
            self._update_stage_from_line(clean)

    def _update_stage_from_line(self, line: str) -> None:
        if "MODULE 1" in line or "Scaffolding" in line:
            self.stage = "scaffolding"
            self.progress_pct = max(self.progress_pct, 12)
        elif "Scaffold [" in line:
            self.progress_pct = min(35, self.progress_pct + 6)
        elif "MODULE 2" in line or "Séquençage" in line:
            self.stage = "sequencing"
            self.progress_pct = max(self.progress_pct, 38)
        elif "Smart Pre-Filter" in line:
            self.stage = "prefilter"
            self.progress_pct = max(self.progress_pct, 55)
        elif "MODULE 3" in line or "Validation (" in line:
            self.stage = "validation"
            self.progress_pct = max(self.progress_pct, 60)
        elif "pLDDT=" in line and ("PASS" in line or "FAIL" in line):
            self.progress_pct = min(90, self.progress_pct + 8)
        elif "Solubilité Protein-Sol" in line:
            self.stage = "solubility"
            self.progress_pct = max(self.progress_pct, 92)
        elif "RAPPORT :" in line or "Rapport sauvegardé" in line:
            self.stage = "reporting"
            self.progress_pct = max(self.progress_pct, 96)
        elif "terminé avec succès" in line:
            self.stage = "completed"
            self.progress_pct = 100

    def to_dict(self, offset: int = 0) -> dict:
        with self.lock:
            end_t = self.finished_at or time.time()
            sliced = self.logs[offset:] if offset < len(self.logs) else []
            return {
                "run_id": self.run_id,
                "status": self.status,
                "stage": self.stage,
                "progress_pct": self.progress_pct,
                "elapsed_sec": round(end_t - self.started_at, 1),
                "returncode": self.returncode,
                "logs": sliced,
                "next_offset": len(self.logs),
            }


def probe_gpu_hardware() -> dict:
    """Interroge nvidia-smi pour remonter l'état exact du GPU distant."""
    info = {
        "gpu_available": False,
        "gpu_name": "CPU Only",
        "vram_total_mb": 0,
        "vram_free_mb": 0,
        "driver": "",
    }
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=4,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            first = res.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in first.split(",")]
            if len(parts) >= 3:
                info["gpu_available"] = True
                info["gpu_name"] = parts[0]
                info["vram_total_mb"] = int(float(parts[1]))
                info["vram_free_mb"] = int(float(parts[2]))
                info["driver"] = parts[3] if len(parts) >= 4 else ""
    except Exception:
        pass
    return info


class WorkerState:
    def __init__(self, python_bin: str, token: str = "") -> None:
        self.python_bin = python_bin
        self.token = token
        self.jobs: dict[str, JobRecord] = {}
        self.active_job_id: str | None = None
        self.lock = threading.Lock()

    def start_job(self, payload: dict) -> JobRecord:
        with self.lock:
            if self.active_job_id:
                cur = self.jobs.get(self.active_job_id)
                if cur and cur.status in ("starting", "running"):
                    raise RuntimeError(f"Un job est déjà en cours sur le GPU ({self.active_job_id}).")

            run_id = payload.get("run_id") or f"run_{int(time.time())}"
            run_id = re.sub(r"[^A-Za-z0-9_-]+", "_", run_id)
            job = JobRecord(run_id)
            self.jobs[run_id] = job
            self.active_job_id = run_id

        staging_dir = PROJECT_ROOT / "inputs" / "remote_jobs" / run_id
        staging_dir.mkdir(parents=True, exist_ok=True)

        # Écrire les fichiers d'entrée transmis par le client Web
        tgt_name = Path(payload.get("target_filename", "target.pdb")).name
        tgt_path = staging_dir / tgt_name
        if payload.get("target_b64"):
            tgt_path.write_bytes(base64.b64decode(payload["target_b64"]))
        else:
            default_tgt = PROJECT_ROOT / "inputs" / "targets" / "zn2+_site.pdb"
            shutil.copy2(default_tgt, tgt_path)

        lig_name = Path(payload.get("ligand_filename", "DFHBI-1T.sdf")).name
        lig_path = staging_dir / lig_name
        if payload.get("ligand_b64"):
            lig_path.write_bytes(base64.b64decode(payload["ligand_b64"]))
        else:
            default_lig = PROJECT_ROOT / "inputs" / "ligands" / "DFHBI-1T.sdf"
            shutil.copy2(default_lig, lig_path)

        cfg_path = staging_dir / "config.yaml"
        if payload.get("config_yaml"):
            cfg_path.write_text(payload["config_yaml"], encoding="utf-8")
        else:
            shutil.copy2(PROJECT_ROOT / "inputs" / "configs" / "default.yaml", cfg_path)

        dry_run = bool(payload.get("dry_run", False))
        skip_val = bool(payload.get("skip_validation", False))

        def _runner() -> None:
            cmd = [
                self.python_bin,
                "-u",
                str(PROJECT_ROOT / "orchestrator.py"),
                "--run-id", run_id,
                "--config", str(cfg_path),
                "--target", str(tgt_path),
                "--ligand", str(lig_path),
                "--output-dir", str(PROJECT_ROOT / "outputs"),
            ]
            if dry_run:
                cmd.append("--dry-run")
            if skip_val:
                cmd.append("--skip-validation")

            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            # Exposer automatiquement les bibliothèques CUDA 11.8 packagées avec PyTorch (+cu118) pour libdgl.so
            py_root = Path(self.python_bin).resolve().parent.parent
            nv_libs = sorted(py_root.glob("lib/python*/site-packages/nvidia/*/lib"))
            if nv_libs:
                existing_ld = env.get("LD_LIBRARY_PATH", "")
                ld_parts = [str(p) for p in nv_libs] + ([existing_ld] if existing_ld else [])
                env["LD_LIBRARY_PATH"] = ":".join(ld_parts)

            drive_chai = Path("/content/drive/MyDrive/optobind_weights_cache/chai")
            if drive_chai.exists() and "CHAI_DOWNLOADS_DIR" not in env:
                env["CHAI_DOWNLOADS_DIR"] = str(drive_chai)

            try:
                job.status = "running"
                job.append_log(f"[WORKER] Démarrage GPU : {' '.join(cmd)}")
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
                job.proc = proc
                if proc.stdout:
                    for line in iter(proc.stdout.readline, ""):
                        job.append_log(line)
                proc.wait()
                job.returncode = proc.returncode
                job.finished_at = time.time()
                if job.status != "cancelled":
                    if proc.returncode == 0:
                        job.status = "completed"
                        job.stage = "completed"
                        job.progress_pct = 100
                    else:
                        job.status = "failed"
                        job.stage = "failed"
            except Exception as exc:
                job.append_log(f"[WORKER ERROR] {exc}")
                job.status = "failed"
                job.stage = "failed"
                job.finished_at = time.time()

        t = threading.Thread(target=_runner, daemon=True)
        t.start()
        return job


def make_handler(state: WorkerState):
    class ColabWorkerHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            return  # Silencieux pour garder la console Colab propre

        def _send_json(self, code: int, data: dict) -> None:
            raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Optobind-Token")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()
            self.wfile.write(raw)

        def _check_auth(self) -> bool:
            if not state.token:
                return True
            hdr = self.headers.get("X-Optobind-Token", "")
            if not hdr:
                auth = self.headers.get("Authorization", "")
                if auth.startswith("Bearer "):
                    hdr = auth[7:].strip()
            if hdr != state.token:
                self._send_json(401, {"error": "Jeton d'authentification Worker invalide."})
                return False
            return True

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Optobind-Token")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()

        def do_GET(self) -> None:
            if not self._check_auth():
                return
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/")
            qs = parse_qs(parsed.query)

            if path in ("", "/api/worker/health"):
                gpu = probe_gpu_hardware()
                w_dir = PROJECT_ROOT / "weights"
                weights = {
                    "rfdiffusion": (w_dir / "RFDiffusionAA_paper_weights.pt").is_file(),
                    "ligandmpnn": (w_dir / "ligandmpnn_v_32_010_25.pt").is_file(),
                    "proteinmpnn": (w_dir / "ProteinMPNN_v_48_020.pt").is_file() or (w_dir / "v_48_020.pt").is_file(),
                }
                active = None
                if state.active_job_id and state.active_job_id in state.jobs:
                    j = state.jobs[state.active_job_id]
                    if j.status in ("starting", "running"):
                        active = j.run_id
                self._send_json(200, {
                    "status": "online",
                    "service": "optobind-colab-worker",
                    "project_root": str(PROJECT_ROOT),
                    "python_bin": state.python_bin,
                    "active_job_id": active,
                    "weights": weights,
                    **gpu,
                })
                return

            m_arch = re.fullmatch(r"/api/worker/jobs/([^/]+)/archive", path)
            if m_arch:
                run_id = m_arch.group(1)
                run_dir = PROJECT_ROOT / "outputs" / run_id
                if not run_dir.is_dir():
                    self._send_json(404, {"error": f"Répertoire de sortie introuvable pour {run_id}."})
                    return
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    for file_p in run_dir.rglob("*"):
                        if file_p.is_file():
                            rel = file_p.relative_to(run_dir).as_posix()
                            zf.write(file_p, arcname=rel)
                raw_zip = buf.getvalue()
                self.send_response(200)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Length", str(len(raw_zip)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(raw_zip)
                return

            m_job = re.fullmatch(r"/api/worker/jobs/([^/]+)", path)
            if m_job:
                run_id = m_job.group(1)
                job = state.jobs.get(run_id)
                if not job:
                    self._send_json(404, {"error": f"Job '{run_id}' inconnu."})
                    return
                offset = int(qs.get("offset", ["0"])[0])
                self._send_json(200, job.to_dict(offset=offset))
                return

            self._send_json(404, {"error": f"Route inconnue : {path}"})

        def do_POST(self) -> None:
            if not self._check_auth():
                return
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/")

            length = int(self.headers.get("Content-Length", "0"))
            body_bytes = self.rfile.read(length) if length > 0 else b"{}"
            try:
                payload = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
            except Exception as exc:
                self._send_json(400, {"error": f"JSON invalide : {exc}"})
                return

            if path == "/api/worker/jobs":
                try:
                    job = state.start_job(payload)
                    self._send_json(200, job.to_dict())
                except Exception as exc:
                    self._send_json(409, {"error": str(exc)})
                return

            m_cancel = re.fullmatch(r"/api/worker/jobs/([^/]+)/cancel", path)
            if m_cancel:
                run_id = m_cancel.group(1)
                job = state.jobs.get(run_id)
                if not job:
                    self._send_json(404, {"error": "Job introuvable."})
                    return
                if job.proc and job.proc.poll() is None:
                    job.status = "cancelled"
                    job.stage = "cancelled"
                    job.proc.terminate()
                    job.append_log("[WORKER] Exécution interrompue par l'utilisateur.")
                self._send_json(200, job.to_dict())
                return

            self._send_json(404, {"error": f"Route POST inconnue : {path}"})

    return ColabWorkerHandler


def _start_cloudflare_tunnel(port: int) -> str | None:
    """Télécharge cloudflared si nécessaire sous Linux et ouvre un tunnel HTTPS gratuit."""
    bin_path = shutil.which("cloudflared") or "/tmp/cloudflared"
    if not Path(bin_path).is_file():
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
        subprocess.run(["wget", "-q", url, "-O", "/tmp/cloudflared"], check=False)
        subprocess.run(["chmod", "+x", "/tmp/cloudflared"], check=False)
        bin_path = "/tmp/cloudflared"

    if not Path(bin_path).is_file():
        return None

    proc = subprocess.Popen(
        [bin_path, "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    public_url = None
    deadline = time.time() + 20.0
    if proc.stdout:
        while time.time() < deadline:
            line = proc.stdout.readline()
            if not line:
                break
            m = re.search(r"(https://[a-zA-Z0-9-]+\.trycloudflare\.com)", line)
            if m:
                public_url = m.group(1)
                break
    return public_url


def main() -> int:
    parser = argparse.ArgumentParser(description="OptoBind-AI Remote GPU Worker (Google Colab Bridge)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--python-bin", default=None, help="Interpréteur Python du pipeline (ex: /content/optobind_env/bin/python).")
    parser.add_argument("--token", default=os.environ.get("OPTOBIND_TOKEN", ""), help="Jeton de sécurité optionnel.")
    parser.add_argument("--cloudflare", action="store_true", help="Ouvrir automatiquement un tunnel HTTPS Cloudflare gratuit.")
    args = parser.parse_args()

    py_bin = args.python_bin
    if not py_bin:
        colab_env = Path("/content/optobind_env/bin/python")
        py_bin = str(colab_env) if colab_env.is_file() else sys.executable

    state = WorkerState(python_bin=py_bin, token=args.token)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))

    gpu = probe_gpu_hardware()
    tunnel_url = _start_cloudflare_tunnel(args.port) if args.cloudflare else None

    print("=" * 68, flush=True)
    print("  OPTOBIND-AI — REMOTE GPU WORKER ONLINE", flush=True)
    print("-" * 68, flush=True)
    print(f"  Local Endpoint : http://{args.host}:{args.port}", flush=True)
    if tunnel_url:
        print(f"  Public URL     : {tunnel_url}", flush=True)
    print(f"  GPU Hardware   : {gpu['gpu_name']} ({gpu['vram_total_mb']} MB VRAM)", flush=True)
    print(f"  Python Runner  : {py_bin}", flush=True)
    if args.token:
        print(f"  Auth Token     : {args.token}", flush=True)
    print("=" * 68, flush=True)
    if tunnel_url:
        print("\nCopiez la 'Public URL' ci-dessus dans l'interface Web OptoBind-AI.\n", flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nArrêt du Worker GPU.", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
