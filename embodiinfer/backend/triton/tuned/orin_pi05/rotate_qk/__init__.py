"""Composite ``rotate_qk`` for the orin_pi05 experiment profile.

Built from two separately tuned and verified exports with disjoint layouts:
``prefix`` (run 20261015T134002Z-6a72e1be9e) and ``decode`` (run 20261015T150031Z-03be0c2142). Each call goes to
the variant whose generated ``applies`` accepts it; otherwise ``applies`` is false
and the caller falls back to its preferred implementation.
"""

from . import decode as _decode
from . import prefix as _prefix

_VARIANTS = (_prefix, _decode)
if _prefix._VERIFIED & _decode._VERIFIED:
    raise ImportError("rotate_qk variants overlap; refuse an ambiguous composite")


def applies(*args, **kwargs):
    """Whether any variant verified this exact argument layout."""
    return any(variant.applies(*args, **kwargs) for variant in _VARIANTS)


def call(*args, **kwargs):
    """Run the variant that verified this layout."""
    for variant in _VARIANTS:
        if variant.applies(*args, **kwargs):
            return variant.call(*args, **kwargs)
    raise ValueError("rotate_qk composite called with an unverified layout")
