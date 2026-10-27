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
