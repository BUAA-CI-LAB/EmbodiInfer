"""Deprecated import shim: ``vvla`` is now ``embodiinfer``.

The Python package was renamed to ``embodiinfer`` (matching the distribution and
product name). This shim keeps ``import vvla`` working for existing deployments;
it resolves every ``vvla``/``vvla.*`` import to the corresponding ``embodiinfer``
module. Notably, the RLinf rollout integration (BUAA-CI-LAB/RLinf,
``vvla-rollout-backend`` branch) still imports ``vvla`` and is migrated to the
new name in a downstream release; this shim is removed only after that lands.
Please switch to ``import embodiinfer``.
"""

from __future__ import annotations

import sys

import embodiinfer

sys.modules[__name__] = embodiinfer  # vvla.* resolves through embodiinfer's path
