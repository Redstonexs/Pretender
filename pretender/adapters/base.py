"""Adapter base: the Adapter Protocol re-export plus capability helpers.

The protocol itself lives in seams.py (the whole extension surface in one
file); this module is where adapter implementations import from.
"""

from __future__ import annotations

from pretender.seams import Adapter

__all__ = ["Adapter", "capability_set"]


def capability_set(*names: str) -> frozenset[str]:
    """Build a ``capabilities`` frozenset from feature names.

    The set is open (platforms invent features); this helper exists only
    for readability at adapter definition sites.
    """
    return frozenset(names)