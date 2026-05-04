"""Nextext package init.

Disables PyTorch's NNPACK backend for CPU 2D convolutions. NNPACK requires
AVX2 and a 3-level cache hierarchy and fails to initialize on Docker Desktop
on Apple Silicon (Rosetta), older x86 CPUs, and some virtualized hosts,
producing a stderr warning on the first CPU conv2d. Setting the flag here
short-circuits the dispatcher before init is attempted; oneDNN handles the
fallback. See https://github.com/pytorch/pytorch/issues/143646.

Doing this in the top-level package init guarantees the flag is set before
any nextext submodule (and any third-party model code it pulls in) reaches a
Conv2d call, since Python always runs ``__init__.py`` before submodule
imports under ``nextext.*``.
"""

from __future__ import annotations

import torch

torch.backends.nnpack.set_flags(_enabled=False)
