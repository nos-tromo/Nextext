"""Tests for the GPU-model residency manager."""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import MagicMock

import pytest

from nextext.utils import model_registry
from nextext.utils.env_cfg import MemoryConfig, load_memory_env
from nextext.utils.model_registry import (
    ModelRegistry,
    ModelSpec,
    Strategy,
    _is_oom_error,
    flush_gpu,
)


class _FakeModel:
    """Lightweight stand-in for a torch module."""

    def __init__(self) -> None:
        self.device: str = "cpu"


def _make_spec(
    name: str = "fake",
    *,
    default_strategy: Strategy = Strategy.OFFLOAD,
    gpu_capable: bool = True,
    move_fn: Any = None,
    loaded_models: list[_FakeModel] | None = None,
) -> ModelSpec:
    """Build a ModelSpec backed by ``_FakeModel`` with recording hooks."""
    loaded = loaded_models if loaded_models is not None else []

    def loader() -> _FakeModel:
        model = _FakeModel()
        loaded.append(model)
        return model

    def default_mover(model: _FakeModel, device: str) -> _FakeModel:
        model.device = device
        return model

    return ModelSpec(
        name=name,
        loader=loader,
        mover=move_fn or default_mover,
        default_strategy=default_strategy,
        gpu_capable=gpu_capable,
    )


# ---------------------------------------------------------------------------
# Basic lifecycle
# ---------------------------------------------------------------------------


def test_acquire_loads_and_moves_to_target_device() -> None:
    registry = ModelRegistry()
    loaded: list[_FakeModel] = []
    registry.register(_make_spec(loaded_models=loaded))

    with registry.acquire("fake", device="cuda") as model:
        assert model.device == "cuda"

    assert len(loaded) == 1


def test_offload_returns_model_to_cpu_on_exit() -> None:
    registry = ModelRegistry()
    registry.register(_make_spec(default_strategy=Strategy.OFFLOAD))

    with registry.acquire("fake", device="cuda") as model:
        assert model.device == "cuda"

    # Model instance is reused across acquires under OFFLOAD
    handle = registry.acquire("fake", device="cuda")
    # Before entering, instance is back on cpu
    state = registry._state("fake")
    assert state.instance is model
    assert state.device == "cpu"
    handle.__exit__(None, None, None)


def test_evict_drops_instance_on_exit() -> None:
    registry = ModelRegistry()
    loaded: list[_FakeModel] = []
    registry.register(_make_spec(default_strategy=Strategy.EVICT, loaded_models=loaded))

    with registry.acquire("fake", device="cuda"):
        pass

    # Next acquire must build a fresh instance
    with registry.acquire("fake", device="cuda"):
        pass

    assert len(loaded) == 2


def test_acquire_raises_for_unknown_spec() -> None:
    registry = ModelRegistry()
    with pytest.raises(KeyError, match="not_registered"):
        registry.acquire("not_registered")


def test_is_registered_reflects_register() -> None:
    registry = ModelRegistry()
    assert not registry.is_registered("fake")
    registry.register(_make_spec())
    assert registry.is_registered("fake")


def test_gpu_capable_false_skips_device_moves() -> None:
    moves: list[tuple[Any, str]] = []

    def recording_mover(model: _FakeModel, device: str) -> _FakeModel:
        moves.append((model, device))
        model.device = device
        return model

    registry = ModelRegistry()
    registry.register(_make_spec(gpu_capable=False, move_fn=recording_mover))

    with registry.acquire("fake", device="cuda") as model:
        assert model.device == "cpu"

    assert moves == []


# ---------------------------------------------------------------------------
# Strategy resolution via env
# ---------------------------------------------------------------------------


def test_global_env_override_flips_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_RESIDENCY_STRATEGY", "evict")
    monkeypatch.delenv("MODEL_RESIDENCY_FAKE", raising=False)

    registry = ModelRegistry()
    loaded: list[_FakeModel] = []
    registry.register(
        _make_spec(default_strategy=Strategy.OFFLOAD, loaded_models=loaded)
    )

    with registry.acquire("fake", device="cuda"):
        pass
    with registry.acquire("fake", device="cuda"):
        pass

    assert len(loaded) == 2  # evict forced reload between acquires


def test_per_model_override_beats_global(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_RESIDENCY_STRATEGY", "evict")
    monkeypatch.setenv("MODEL_RESIDENCY_FAKE", "offload")

    registry = ModelRegistry()
    loaded: list[_FakeModel] = []
    registry.register(_make_spec(default_strategy=Strategy.EVICT, loaded_models=loaded))

    with registry.acquire("fake", device="cuda"):
        pass
    with registry.acquire("fake", device="cuda"):
        pass

    assert len(loaded) == 1  # offload kept the instance alive


def test_unknown_strategy_falls_back_to_spec_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODEL_RESIDENCY_STRATEGY", "garbage")
    monkeypatch.delenv("MODEL_RESIDENCY_FAKE", raising=False)

    registry = ModelRegistry()
    loaded: list[_FakeModel] = []
    registry.register(_make_spec(default_strategy=Strategy.EVICT, loaded_models=loaded))

    with registry.acquire("fake", device="cuda"):
        pass
    with registry.acquire("fake", device="cuda"):
        pass

    assert len(loaded) == 2  # spec default (EVICT) applied


# ---------------------------------------------------------------------------
# Thread-safety
# ---------------------------------------------------------------------------


def test_concurrent_acquires_do_not_double_load() -> None:
    registry = ModelRegistry()
    loaded: list[_FakeModel] = []

    load_gate = threading.Event()

    def slow_loader() -> _FakeModel:
        load_gate.wait(timeout=2.0)
        model = _FakeModel()
        loaded.append(model)
        return model

    def mover(model: _FakeModel, device: str) -> _FakeModel:
        model.device = device
        return model

    registry.register(
        ModelSpec(
            name="fake",
            loader=slow_loader,
            mover=mover,
            default_strategy=Strategy.OFFLOAD,
        )
    )

    results: list[_FakeModel] = []

    def worker() -> None:
        with registry.acquire("fake", device="cuda") as model:
            results.append(model)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    load_gate.set()
    for t in threads:
        t.join()

    assert len(loaded) == 1
    assert all(m is loaded[0] for m in results)


# ---------------------------------------------------------------------------
# OOM fallback
# ---------------------------------------------------------------------------


def test_oom_on_gpu_move_falls_back_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_devices: list[str] = []

    def flaky_mover(model: _FakeModel, device: str) -> _FakeModel:
        call_devices.append(device)
        if device == "cuda":
            raise RuntimeError("CUDA out of memory")
        model.device = device
        return model

    registry = ModelRegistry()
    registry.register(_make_spec(move_fn=flaky_mover))

    # Silence empty_cache during flush_gpu
    monkeypatch.setattr(model_registry.torch.cuda, "is_available", lambda: False)

    handle = registry.acquire("fake", device="cuda")
    model = handle.__enter__()
    assert model.device == "cpu"
    assert handle.device == "cpu"
    # Attempted cuda at least once, plus a retry, plus the final cpu fallback
    assert call_devices.count("cuda") == 2
    assert call_devices.count("cpu") == 1
    handle.__exit__(None, None, None)


def test_non_oom_error_propagates() -> None:
    def angry_mover(model: _FakeModel, device: str) -> _FakeModel:
        raise RuntimeError("something totally unrelated")

    registry = ModelRegistry()
    registry.register(_make_spec(move_fn=angry_mover))

    with pytest.raises(RuntimeError, match="totally unrelated"):
        with registry.acquire("fake", device="cuda"):
            pass


def test_is_oom_error_detects_common_messages() -> None:
    assert _is_oom_error(RuntimeError("CUDA out of memory"))
    assert _is_oom_error(RuntimeError("cuda oom triggered"))
    assert not _is_oom_error(RuntimeError("device-side assert"))
    assert not _is_oom_error(ValueError("not a runtime"))


# ---------------------------------------------------------------------------
# flush_gpu + shutdown + evict
# ---------------------------------------------------------------------------


def test_flush_gpu_runs_gc_and_empty_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gc_mock = MagicMock()
    empty_cache_mock = MagicMock()

    monkeypatch.setattr(model_registry.gc, "collect", gc_mock)
    monkeypatch.setattr(model_registry.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(model_registry.torch.cuda, "empty_cache", empty_cache_mock)

    flush_gpu()

    gc_mock.assert_called_once()
    empty_cache_mock.assert_called_once()


def test_flush_gpu_without_cuda_still_runs_gc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gc_mock = MagicMock()
    empty_cache_mock = MagicMock()

    monkeypatch.setattr(model_registry.gc, "collect", gc_mock)
    monkeypatch.setattr(model_registry.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(model_registry.torch.cuda, "empty_cache", empty_cache_mock)

    flush_gpu()

    gc_mock.assert_called_once()
    empty_cache_mock.assert_not_called()


def test_shutdown_drops_all_instances(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_registry.torch.cuda, "is_available", lambda: False)

    registry = ModelRegistry()
    loaded_a: list[_FakeModel] = []
    loaded_b: list[_FakeModel] = []
    registry.register(_make_spec(name="a", loaded_models=loaded_a))
    registry.register(_make_spec(name="b", loaded_models=loaded_b))

    with registry.acquire("a", device="cpu"):
        pass
    with registry.acquire("b", device="cpu"):
        pass

    registry.shutdown()

    with registry.acquire("a", device="cpu"):
        pass
    with registry.acquire("b", device="cpu"):
        pass

    assert len(loaded_a) == 2
    assert len(loaded_b) == 2


def test_evict_drops_single_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_registry.torch.cuda, "is_available", lambda: False)

    registry = ModelRegistry()
    loaded_a: list[_FakeModel] = []
    loaded_b: list[_FakeModel] = []
    registry.register(_make_spec(name="a", loaded_models=loaded_a))
    registry.register(_make_spec(name="b", loaded_models=loaded_b))

    with registry.acquire("a", device="cpu"):
        pass
    with registry.acquire("b", device="cpu"):
        pass
    assert len(loaded_a) == 1
    assert len(loaded_b) == 1

    registry.evict("a")

    with registry.acquire("a", device="cpu"):
        pass
    with registry.acquire("b", device="cpu"):
        pass
    assert len(loaded_a) == 2  # reloaded
    assert len(loaded_b) == 1  # untouched


def test_evict_ignores_unknown_name() -> None:
    registry = ModelRegistry()
    registry.evict("never-registered")  # must not raise


# ---------------------------------------------------------------------------
# load_memory_env
# ---------------------------------------------------------------------------


def test_load_memory_env_defaults_to_offload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MODEL_RESIDENCY_STRATEGY", raising=False)
    cfg = load_memory_env()
    assert isinstance(cfg, MemoryConfig)
    assert cfg.default_strategy == "offload"


def test_load_memory_env_accepts_evict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_RESIDENCY_STRATEGY", "EVICT")
    cfg = load_memory_env()
    assert cfg.default_strategy == "evict"


def test_load_memory_env_rejects_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_RESIDENCY_STRATEGY", "garbage")
    cfg = load_memory_env()
    assert cfg.default_strategy == "offload"
