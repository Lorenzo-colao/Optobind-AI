"""
optobind_ai.utils.runner
========================
Lanceur universel avec patchs de rétrocompatibilité Python 3.10/3.11/3.12/3.13.
Rétablit collections.Mapping et collections.Iterable pour DGL et les vieux paquets.
"""

import sys
import collections
import collections.abc

# Rétablir les alias supprimés en Python 3.10+
for attr in ("Mapping", "Iterable", "Sequence", "MutableMapping", "Callable"):
    if not hasattr(collections, attr) and hasattr(collections.abc, attr):
        setattr(collections, attr, getattr(collections.abc, attr))

# Rétablir les alias DGL (AvgPooling & MaxPooling déplacés dans dgl.nn)
try:
    import dgl.nn
    import dgl.nn.pytorch as dgl_pytorch
    if not hasattr(dgl_pytorch, "AvgPooling"):
        dgl_pytorch.AvgPooling = getattr(dgl.nn, "AvgPooling", None)
    if not hasattr(dgl_pytorch, "MaxPooling"):
        dgl_pytorch.MaxPooling = getattr(dgl.nn, "MaxPooling", None)
except Exception:
    pass

from pathlib import Path

if len(sys.argv) > 1:
    target_script = sys.argv.pop(1)
    target_path = Path(target_script).resolve()
    sys.argv[0] = str(target_path)

    # Ajouter automatiquement la racine du repo et ses sous-modules au sys.path
    script_dir = target_path.parent
    repo_dir = script_dir.parent
    se3_dir = repo_dir / "env" / "SE3Transformer"

    for candidate in (repo_dir, se3_dir, script_dir):
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))

    with open(target_path, "rb") as fh:
        code = compile(fh.read(), str(target_path), "exec")
    globs = {
        "__name__": "__main__",
        "__file__": str(target_path),
        "__doc__": None,
    }
    exec(code, globs)
