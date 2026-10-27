"""
optobind_ai.utils.compat
========================
Patch de rétrocompatibilité pour Python 3.10+ (pour dgl et dépendances plus anciennes).
"""

import collections
import collections.abc

for _attr in ("Mapping", "Iterable", "Sequence", "MutableMapping", "Callable"):
    if not hasattr(collections, _attr) and hasattr(collections.abc, _attr):
        setattr(collections, _attr, getattr(collections.abc, _attr))

# Patch NVTX profiling pour permettre l'exécution sur CPU sans erreur NVTX
try:
    import contextlib
    import torch
    import torch.cuda.nvtx

    @contextlib.contextmanager
    def _dummy_nvtx_range(*args, **kwargs):
        yield

    torch.cuda.nvtx.range = _dummy_nvtx_range
except Exception:
    pass
