"""Centralized GPU-model residency manager for Nextext.

Models that can live on GPU (Whisper, diarization, GLiNER, ...) register a
:class:`ModelSpec` with the process-wide :data:`REGISTRY`. Callers wrap use of
the model in ``with REGISTRY.acquire(name) as model:`` to ensure the model is
on the target device during use and released (either moved back to CPU or
evicted) after the block exits. Between batches, call :func:`flush_gpu` to
collect Python garbage and reclaim PyTorch allocator reservations.

Strategy resolution per release:
  1. ``MODEL_RESIDENCY_<NAME>`` environment variable (per-model override).
  2. ``MODEL_RESIDENCY_STRATEGY`` (global override).
  3. ``spec.default_strategy``.
"""

from __future__ import annotations

import gc as gc  # re-exported for tests
import os
import threading
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from enum import StrEnum
from types import TracebackType
from typing import Any

import torch as torch  # re-exported for tests
from loguru import logger


class Strategy(StrEnum):
    """Residency strategy for a model after a caller releases it."""

    OFFLOAD = "offload"
    EVICT = "evict"


VALID_STRATEGIES: frozenset[str] = frozenset(s.value for s in Strategy)


@dataclass(frozen=True)
class ModelSpec:
    """Loader / mover configuration for a managed model.

    Attributes:
        name: Registry key, e.g. ``"gliner"``, ``"whisper_turbo"``.
        loader: Zero-arg callable that builds the model on CPU.
        mover: Callable ``(model, device) -> model`` that moves the model to
            the requested device and returns it (possibly the same object).
        default_strategy: Fallback when no env override is set.
        gpu_capable: Set ``False`` for CPU-only caches (e.g. spaCy); disables
            device moves and forces an effective strategy of ``OFFLOAD``.
        mps_compatible: Set ``False`` for models whose ops are not supported
            on the Apple Silicon MPS backend (notably Whisper, which relies
            on sparse-tensor constructors that raise
            ``NotImplementedError`` on SparseMPS even with
            ``PYTORCH_ENABLE_MPS_FALLBACK=1``). When ``False`` the default
            device picker skips MPS and falls back to CPU. Ignored when
            ``gpu_capable`` is ``False``.
    """

    name: str
    loader: Callable[[], Any]
    mover: Callable[[Any, str], Any]
    default_strategy: Strategy = Strategy.OFFLOAD
    gpu_capable: bool = True
    mps_compatible: bool = True


@dataclass
class _HandleState:
    """Mutable state tracked per registered spec."""

    instance: Any = None
    device: str = "cpu"
    lock: threading.RLock = field(default_factory=threading.RLock)


def _default_device(*, allow_mps: bool = True) -> str:
    """Return the preferred GPU device name or ``"cpu"``.

    Args:
        allow_mps: When ``False``, the Apple Silicon MPS backend is skipped
            and CPU is returned instead. Set by the registry for specs that
            declare ``mps_compatible=False`` (e.g. Whisper), whose
            sparse-tensor paths crash on SparseMPS.

    Returns:
        str: ``"cuda"`` if a CUDA GPU is available, ``"mps"`` if an Apple
            Silicon GPU is available and ``allow_mps`` is ``True``, or
            ``"cpu"`` otherwise.
    """
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        return "cuda"
    if allow_mps:
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    return "cpu"


def _is_oom_error(exc: BaseException) -> bool:
    """Return True for torch OOM exceptions across torch versions.

    Args:
        exc (BaseException): The exception to inspect.

    Returns:
        bool: ``True`` if ``exc`` is a CUDA out-of-memory error (either the
            dedicated ``torch.cuda.OutOfMemoryError`` subclass or a
            ``RuntimeError`` whose message contains ``"out of memory"`` or
            ``"cuda oom"``), ``False`` otherwise.
    """
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_cls is not None and isinstance(exc, oom_cls):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        return "out of memory" in msg or "cuda oom" in msg
    return False


class ModelHandle(AbstractContextManager[Any]):
    """Context manager returned by :meth:`ModelRegistry.acquire`."""

    def __init__(
        self,
        registry: ModelRegistry,
        spec: ModelSpec,
        target_device: str,
    ) -> None:
        """Store registry, spec, and target device; mark as not yet released.

        Args:
            registry (ModelRegistry): The owning registry instance.
            spec (ModelSpec): The model specification being managed.
            target_device (str): Preferred device to promote the model to on
                :meth:`__enter__`.
        """
        self._registry = registry
        self._spec = spec
        self._target_device = target_device
        self._released = False

    @property
    def device(self) -> str:
        """Return the actual device of the underlying model.

        Returns:
            str: The live device string (``"cuda"``, ``"mps"``, or
                ``"cpu"``), reflecting any OOM fallback that occurred.
        """
        return self._registry._state(self._spec.name).device

    def __enter__(self) -> Any:
        """Acquire the model and move it to the target device.

        Returns:
            Any: The model instance on the target device.  May be on CPU
                if an OOM fallback occurred during the GPU move.
        """
        return self._registry._acquire(self._spec, self._target_device)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Release the model according to its resolved residency strategy.

        Args:
            exc_type (type[BaseException] | None): Exception type, if any.
            exc (BaseException | None): Exception instance, if any.
            tb (TracebackType | None): Traceback, if any.
        """
        if self._released:
            return
        self._released = True
        self._registry._release(self._spec)


class ModelRegistry:
    """Process-wide manager for GPU-resident models."""

    def __init__(self) -> None:
        """Initialize empty spec and state dictionaries."""
        self._specs: dict[str, ModelSpec] = {}
        self._states: dict[str, _HandleState] = {}
        self._registry_lock = threading.RLock()

    def register(self, spec: ModelSpec) -> None:
        """Register a model spec, silently overwriting an existing entry.

        Args:
            spec (ModelSpec): Model specification to register under
                ``spec.name``.
        """
        with self._registry_lock:
            self._specs[spec.name] = spec
            self._states.setdefault(spec.name, _HandleState())

    def is_registered(self, name: str) -> bool:
        """Return ``True`` if a spec is registered under ``name``.

        Args:
            name (str): Registry key to look up.

        Returns:
            bool: ``True`` when a spec with ``name`` has been registered.
        """
        return name in self._specs

    def acquire(self, name: str, *, device: str | None = None) -> ModelHandle:
        """Return a context manager that yields the moved model.

        Args:
            name (str): Registered spec name.
            device (str | None): Optional device override. Defaults to the
                preferred GPU device when ``spec.gpu_capable``; ``"cpu"``
                otherwise.

        Returns:
            ModelHandle: Context manager that moves the model to ``device``
                on entry and releases it on exit.

        Raises:
            KeyError: If ``name`` is not registered.
        """
        spec = self._specs.get(name)
        if spec is None:
            raise KeyError(f"No registered model named '{name}'.")
        if device is not None:
            target = device
        elif spec.gpu_capable:
            target = _default_device(allow_mps=spec.mps_compatible)
        else:
            target = "cpu"
        return ModelHandle(self, spec, target)

    def evict(self, name: str) -> None:
        """Drop the cached instance for a single model and flush GPU memory.

        Args:
            name (str): Registry key whose cached instance should be
                dropped. No-op when ``name`` is not registered.
        """
        state = self._states.get(name)
        if state is None:
            return
        with state.lock:
            state.instance = None
            state.device = "cpu"
        self.flush_gpu()

    def flush_gpu(self) -> None:
        """Run ``gc.collect()`` and ``torch.cuda.empty_cache()``.

        Safe to call between files in the Streamlit / CLI loops even when no
        model was acquired during the prior iteration.
        """
        gc.collect()
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("GPU memory flushed (gc.collect + cuda.empty_cache).")
        else:
            logger.debug("GPU memory flushed (gc.collect only, no CUDA available).")

    def shutdown(self) -> None:
        """Evict every cached instance and flush. Use on process teardown."""
        with self._registry_lock:
            for state in self._states.values():
                with state.lock:
                    state.instance = None
                    state.device = "cpu"
        self.flush_gpu()

    # ------------------------------------------------------------------
    # Internal API consumed by ModelHandle
    # ------------------------------------------------------------------

    def _state(self, name: str) -> _HandleState:
        """Return the mutable state record for the named spec.

        Args:
            name (str): Registered spec name.

        Returns:
            _HandleState: The live state object for ``name``.
        """
        return self._states[name]

    def _resolve_strategy(self, spec: ModelSpec) -> Strategy:
        """Determine the effective residency strategy for ``spec``.

        Resolution order:

        1. ``MODEL_RESIDENCY_<NAME>`` env var (per-model override).
        2. ``MODEL_RESIDENCY_STRATEGY`` env var (global override).
        3. ``spec.default_strategy``.

        Args:
            spec (ModelSpec): The model specification whose strategy to resolve.

        Returns:
            Strategy: The resolved :class:`Strategy` enum member.
        """
        per_model_raw = os.getenv(f"MODEL_RESIDENCY_{spec.name.upper()}", "").strip().lower()
        if per_model_raw in VALID_STRATEGIES:
            return Strategy(per_model_raw)
        if per_model_raw:
            logger.warning(
                "Unknown MODEL_RESIDENCY_{}='{}'. Falling back.",
                spec.name.upper(),
                per_model_raw,
            )

        global_raw = os.getenv("MODEL_RESIDENCY_STRATEGY", "").strip().lower()
        if global_raw in VALID_STRATEGIES:
            return Strategy(global_raw)
        if global_raw:
            logger.warning(
                "Unknown MODEL_RESIDENCY_STRATEGY='{}'. Using spec default.",
                global_raw,
            )

        return spec.default_strategy

    def _acquire(self, spec: ModelSpec, target_device: str) -> Any:
        """Load the model if necessary and move it to ``target_device``.

        Args:
            spec (ModelSpec): The model specification to acquire.
            target_device (str): Device to move the model to.  Ignored when
                ``spec.gpu_capable`` is ``False`` (always stays on CPU).

        Returns:
            Any: The live model instance on the requested device, or on CPU
                after an OOM fallback.
        """
        state = self._states[spec.name]
        with state.lock:
            if state.instance is None:
                logger.info("Loading model '{}' on CPU for first use.", spec.name)
                state.instance = spec.loader()
                state.device = "cpu"
            desired = target_device if spec.gpu_capable else "cpu"
            if spec.gpu_capable and state.device != desired:
                state.instance, state.device = self._safe_move(spec, state.instance, desired)
            return state.instance

    def _release(self, spec: ModelSpec) -> None:
        """Apply the resolved residency strategy after a caller releases the model.

        Under :attr:`Strategy.EVICT` the cached instance is dropped and GPU
        memory is flushed immediately.  Under :attr:`Strategy.OFFLOAD` the
        model is moved back to CPU and memory is flushed.

        Args:
            spec (ModelSpec): The model specification to release.
        """
        state = self._states[spec.name]
        strategy = self._resolve_strategy(spec)
        with state.lock:
            if state.instance is None:
                return
            if strategy is Strategy.EVICT:
                logger.debug("Evicting model '{}' per strategy.", spec.name)
                state.instance = None
                state.device = "cpu"
                self.flush_gpu()
                return
            if spec.gpu_capable and state.device != "cpu":
                logger.debug("Offloading model '{}' to CPU.", spec.name)
                state.instance, state.device = self._safe_move(spec, state.instance, "cpu")
                self.flush_gpu()

    def _safe_move(self, spec: ModelSpec, model: Any, target: str) -> tuple[Any, str]:
        """Move ``model`` to ``target`` with a one-shot OOM retry.

        On OOM while promoting to GPU: flush and retry once; if that also
        fails, log a warning and fall back to CPU rather than crash.

        Args:
            spec (ModelSpec): The model specification (used for log messages).
            model (Any): The model instance to move.
            target (str): Destination device string, e.g. ``"cuda"`` or
                ``"cpu"``.

        Returns:
            tuple[Any, str]: ``(moved_model, actual_device)`` where
                ``actual_device`` may be ``"cpu"`` after an OOM fallback.

        Raises:
            Exception: Re-raises any non-OOM exception from the mover, and
                re-raises OOM exceptions that occur while trying to fall back
                to CPU.
        """
        try:
            moved = spec.mover(model, target)
            return moved, target
        except Exception as exc:
            if target == "cpu" or not _is_oom_error(exc):
                raise
            logger.warning(
                "OOM moving '{}' to '{}'. Flushing and retrying.",
                spec.name,
                target,
            )
            self.flush_gpu()
            try:
                moved = spec.mover(model, target)
                return moved, target
            except Exception as exc2:
                if not _is_oom_error(exc2):
                    raise
                logger.warning(
                    "Still OOM moving '{}' to '{}'. Falling back to CPU.",
                    spec.name,
                    target,
                )
                try:
                    moved = spec.mover(model, "cpu")
                    return moved, "cpu"
                except Exception:
                    logger.warning(
                        "Could not move '{}' to CPU after OOM. Instance may be in an inconsistent state.",
                        spec.name,
                    )
                    return model, "cpu"


REGISTRY = ModelRegistry()


def flush_gpu() -> None:
    """Run ``gc.collect()`` and flush GPU allocator reservations.

    Convenience wrapper for :meth:`ModelRegistry.flush_gpu` on the
    process-wide :data:`REGISTRY`.
    """
    REGISTRY.flush_gpu()
