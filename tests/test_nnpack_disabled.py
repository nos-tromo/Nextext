"""Regression test: NNPACK is disabled at package import.

The Nextext package init disables PyTorch's NNPACK backend so that the
"Could not initialize NNPACK! Reason: Unsupported hardware." stderr warning
is never emitted on hosts where NNPACK init fails (Docker Desktop on Apple
Silicon via Rosetta, older x86 CPUs, virtualized hosts). This test guards
against accidental removal or replacement of that disable in future
refactors.
"""

from __future__ import annotations

import torch

import nextext  # noqa: F401  # side-effect: disables NNPACK


def test_nnpack_disabled_at_import() -> None:
    """Ensure ``torch._C._get_nnpack_enabled()`` is ``False`` after import.

    The Python attribute ``torch.backends.nnpack.enabled`` does not exist on
    torch >= 2.5; the C++ context flag must be set via
    ``torch.backends.nnpack.set_flags(_enabled=False)`` (which is what
    ``nextext/__init__.py`` calls). We assert the C++-side flag directly
    because that is what the conv dispatcher actually reads.
    """
    assert not torch._C._get_nnpack_enabled()
