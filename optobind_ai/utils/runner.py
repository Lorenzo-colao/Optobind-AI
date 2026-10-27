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

if len(sys.argv) > 1:
    target_script = sys.argv.pop(1)
    sys.argv[0] = target_script
    with open(target_script, "rb") as fh:
        code = compile(fh.read(), target_script, "exec")
    globs = {
        "__name__": "__main__",
        "__file__": target_script,
        "__doc__": None,
    }
    exec(code, globs)
