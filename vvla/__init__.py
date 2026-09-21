"""Deprecated import shim: ``vvla`` is now ``embodiinfer``.

The Python package was renamed to ``embodiinfer`` (matching the distribution and
product name). This shim keeps ``import vvla`` working for existing deployments;
it resolves every ``vvla``/``vvla.*`` import to the corresponding ``embodiinfer``
module. It will be removed in a future release — please switch to
``import embodiinfer``.
"""

from __future__ import annotations

import sys

import embodiinfer

sys.modules[__name__] = embodiinfer  # vvla.* resolves through embodiinfer's path
