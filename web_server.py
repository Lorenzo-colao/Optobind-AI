#!/usr/bin/env python3
"""
web_server.py — Lanceur de l'application Web OptoBind Studio.

Usage :
  python web_server.py --port 8080
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
import threading
import time

import uvicorn


def main() -> int:
    parser = argparse.ArgumentParser(description="OptoBind Studio Web Interface")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--open-browser", action="store_true", help="Ouvrir automatiquement dans le navigateur.")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"
    print("=" * 64)
    print("  OptoBind Studio — Interface de Conception Protéique De Novo")
    print(f"  Serveur actif : {url}")
    print("=" * 64)

    if args.open_browser:
        def _open():
            time.sleep(1.0)
            webbrowser.open(url)
        threading.Thread(target=_open, daemon=True).start()

    uvicorn.run("optobind_ai.web.server:app", host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
