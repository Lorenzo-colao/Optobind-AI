"""
optobind_ai.utils.logger
========================
Configuration centralisée du système de logs (console + fichier rotatif).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


# Palette ANSI pour la console
_COLORS = {
    "DEBUG"    : "\033[36m",   # cyan
    "INFO"     : "\033[32m",   # vert
    "WARNING"  : "\033[33m",   # jaune
    "ERROR"    : "\033[31m",   # rouge
    "CRITICAL" : "\033[35m",   # magenta
    "RESET"    : "\033[0m",
}


class _ColorFormatter(logging.Formatter):
    """Formateur console avec coloration ANSI selon le niveau."""

    FMT = "%(asctime)s  %(levelname)-8s  %(message)s"
    DATE = "%H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        color = _COLORS.get(record.levelname, "")
        reset = _COLORS["RESET"]
        formatter = logging.Formatter(
            f"{color}{self.FMT}{reset}", datefmt=self.DATE
        )
        return formatter.format(record)


def setup_logger(
    name    : str,
    log_dir : Path,
    run_id  : str,
    verbose : bool = False,
) -> logging.Logger:
    """
    Initialise et retourne un logger nommé.

    Deux handlers sont ajoutés :
    - **Console** (stdout) : niveau INFO (ou DEBUG si ``verbose``).
    - **Fichier** : niveau DEBUG, sauvegardé sous ``log_dir/<run_id>.log``.

    Parameters
    ----------
    name    : Nom du logger (racine de la hiérarchie logging).
    log_dir : Répertoire dans lequel sauvegarder le fichier de log.
    run_id  : Identifiant du run, utilisé pour nommer le fichier.
    verbose : Si True, le handler console descend au niveau DEBUG.

    Returns
    -------
    logging.Logger
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{run_id}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Éviter les doublons si le logger est appelé plusieurs fois
    if logger.handlers:
        logger.handlers.clear()

    # ── Handler console ───────────────────────────────────────────────────────
    # Sur Windows (cp1252), forcer l'encodage UTF-8 pour éviter les UnicodeEncodeError
    # liés aux caractères spéciaux (┌─ ▸ ✓ ✗ etc.)
    import sys as _sys
    console_stream = _sys.stdout
    if _sys.platform == "win32":
        try:
            import io
            console_stream = io.TextIOWrapper(
                _sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
            )
        except AttributeError:
            # Si stdout n'a pas de .buffer (ex: dans un IDE), on continue sans wrapper
            pass

    ch = logging.StreamHandler(console_stream)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(_ColorFormatter())
    logger.addHandler(ch)

    # ── Handler fichier ───────────────────────────────────────────────────────
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(fh)

    logger.info("Logger initialisé — fichier de log : %s", log_file)
    return logger
