"""Tests for idle model unloading and the swap-path collect.

No weights are loaded — a sentinel object stands in for the model, which is
all these need: what is under test is when the engine drops its reference,
and what it does before sweeping the allocator cache.
"""

from __future__ import annotations

import threading
import time

import pytest

from app import engine as engine_mod
from app.config import settings
from app.engine import Engine


@pytest.fixture
def eng(monkeypatch):
    """A bare Engine with a fake resident model, no reaper running."""
    e = Engine.__new__(Engine)
    e._model = object()
    e._processor = object()
    e._model_id = "fake/model"
    e._kind = "tts"
    e._loading_id = None
    e._lock = threading.RLock()
    e._last_used = time.monotonic()
    e._reaper = None
    e._stop_reaper = threading.Event()
    e.device = "cpu"
    e.dtype = "float32"
    yield e
    e.stop_reaper()


# --- idle accounting ---------------------------------------------------------


def test_idle_seconds_is_none_when_nothing_resident(eng):
    eng._model = None
    assert eng.idle_seconds is None


def test_idle_seconds_grows_from_last_use(eng):
    eng._last_used = time.monotonic() - 42
    assert eng.idle_seconds == pytest.approx(42, abs=1)


def test_touch_resets_the_clock(eng):
    eng._last_used = time.monotonic() - 42
    eng._touch()
    assert eng.idle_seconds < 1


def test_idle_clock_is_monotonic_not_wall_clock(monkeypatch, eng):
    """A wall-clock jump (NTP step, laptop sleep) must not fake up an idle
    model. Pinning time.time() to an implausible value would poison
    _last_used if the clock came from there."""
    monkeypatch.setattr(engine_mod.time, "time", lambda: 1_000_000_000.0)
    eng._touch()
    assert eng._last_used != 1_000_000_000.0
    assert eng._last_used == pytest.approx(time.monotonic(), abs=1)
    assert eng.idle_seconds < 1


# --- the reap decision -------------------------------------------------------


def test_unloads_once_idle_exceeds_the_timeout(monkeypatch, eng):
    monkeypatch.setattr(settings, "idle_unload_seconds", 10)
    eng._last_used = time.monotonic() - 11
    assert eng.unload_if_idle() is True
    assert eng._model is None
    assert eng.loaded_model is None


def test_keeps_a_recently_used_model(monkeypatch, eng):
    monkeypatch.setattr(settings, "idle_unload_seconds", 10)
    eng._last_used = time.monotonic() - 3
    assert eng.unload_if_idle() is False
    assert eng._model is not None


def test_zero_disables_unloading_entirely(monkeypatch, eng):
    """The historical behaviour has to stay reachable."""
    monkeypatch.setattr(settings, "idle_unload_seconds", 0)
    eng._last_used = time.monotonic() - 86400
    assert eng.unload_if_idle() is False
    assert eng._model is not None


def test_nothing_resident_is_a_no_op(monkeypatch, eng):
    monkeypatch.setattr(settings, "idle_unload_seconds", 10)
    eng._model = None
    eng._last_used = None
    assert eng.unload_if_idle() is False


def test_a_generate_finishing_during_the_wait_saves_the_model(monkeypatch, eng):
    """The reaper waits on the lock; by the time it gets in, the clock may
    have been reset by the generate it was waiting for. It must re-check."""
    monkeypatch.setattr(settings, "idle_unload_seconds", 10)
    eng._last_used = time.monotonic() - 11

    holder_in = threading.Event()
    release = threading.Event()

    def busy_generate():
        with eng._lock:
            holder_in.set()
            release.wait(5)
            eng._touch()  # generate completed, model is in active use

    t = threading.Thread(target=busy_generate)
    t.start()
    holder_in.wait(5)

    result: list[bool] = []
    reaper = threading.Thread(target=lambda: result.append(eng.unload_if_idle()))
    reaper.start()
    time.sleep(0.1)  # let the reaper block on the lock
    release.set()
    t.join(5)
    reaper.join(5)

    assert result == [False]
    assert eng._model is not None


# --- the swap-path collect ---------------------------------------------------


def test_unload_collects_before_emptying_the_cache(monkeypatch, eng):
    """gc.collect() must run before empty_cache(), or the cache sweep finds
    the weights still reachable through their reference cycles and frees
    nothing — the next load then stacks on top of the old model."""
    eng.device = "cuda"
    order: list[str] = []
    monkeypatch.setattr(engine_mod.gc, "collect", lambda *a: order.append("collect"))
    monkeypatch.setattr(
        engine_mod.torch.cuda, "empty_cache", lambda: order.append("empty_cache")
    )
    eng._unload_resident()
    assert order == ["collect", "empty_cache"]


def test_unload_drops_every_reference(eng):
    eng._unload_resident()
    assert (eng._model, eng._processor, eng._model_id, eng._kind) == (
        None, None, None, None
    )
    assert eng._last_used is None


def test_unload_of_nothing_does_not_collect(monkeypatch, eng):
    """No resident model means no work — don't pay for a full gc pass."""
    eng._model = None
    collected = []
    monkeypatch.setattr(engine_mod.gc, "collect", lambda *a: collected.append(1))
    eng._unload_resident()
    assert collected == []


# --- reaper thread lifecycle -------------------------------------------------


def test_reaper_actually_unloads_in_the_background(monkeypatch, eng):
    monkeypatch.setattr(settings, "idle_unload_seconds", 1)
    eng._last_used = time.monotonic() - 5
    eng._ensure_reaper()
    deadline = time.monotonic() + 5
    while eng._model is not None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert eng._model is None


def test_ensure_reaper_starts_only_one_thread(monkeypatch, eng):
    monkeypatch.setattr(settings, "idle_unload_seconds", 60)
    eng._ensure_reaper()
    first = eng._reaper
    eng._ensure_reaper()
    assert eng._reaper is first


def test_no_reaper_thread_when_disabled(monkeypatch, eng):
    monkeypatch.setattr(settings, "idle_unload_seconds", 0)
    eng._ensure_reaper()
    assert eng._reaper is None


def test_reaper_survives_an_unload_failure(monkeypatch, eng):
    """One bad sweep must not silently kill the reaper for the process's life."""
    monkeypatch.setattr(settings, "idle_unload_seconds", 1)
    calls = []

    def boom():
        calls.append(1)
        raise RuntimeError("allocator exploded")

    monkeypatch.setattr(eng, "unload_if_idle", boom)
    eng._ensure_reaper()
    deadline = time.monotonic() + 5
    while len(calls) < 2 and time.monotonic() < deadline:
        time.sleep(0.05)
    assert len(calls) >= 2
    assert eng._reaper.is_alive()


def test_tick_stays_within_bounds(monkeypatch, eng):
    for timeout, expected in [(1, 1.0), (900, 30.0), (40, 10.0), (0, 30.0)]:
        monkeypatch.setattr(settings, "idle_unload_seconds", timeout)
        assert eng._reaper_tick() == expected


# --- preload deduplication ---------------------------------------------------


def test_preload_skips_a_model_already_loading(eng):
    eng._loading_id = "fake/other"
    eng._model_id = None
    assert eng.preload_async("fake/other") is False


def test_preload_skips_a_model_already_resident(eng):
    assert eng.preload_async("fake/model") is False


def test_preload_starts_a_load_for_a_new_model(monkeypatch, eng):
    started = threading.Event()
    monkeypatch.setattr(
        Engine, "ensure_loaded", lambda self, *a, **k: started.set()
    )
    assert eng.preload_async("fake/new") is True
    assert started.wait(5)


# --- settings ----------------------------------------------------------------


def test_default_timeout_is_fifteen_minutes():
    from app.config import Settings

    assert Settings(_env_file=None).idle_unload_seconds == 900


def test_negative_timeout_is_rejected():
    import pydantic

    from app.config import Settings

    with pytest.raises(pydantic.ValidationError):
        Settings(_env_file=None, idle_unload_seconds=-1)


def test_env_var_overrides_the_default(monkeypatch):
    from app.config import Settings

    monkeypatch.setenv("IDLE_UNLOAD_SECONDS", "60")
    assert Settings(_env_file=None).idle_unload_seconds == 60


# --- rss reporting -----------------------------------------------------------


def test_rss_bytes_reports_something_plausible():
    rss = engine_mod.rss_bytes()
    assert rss is None or rss > 1024 * 1024  # any live interpreter clears 1MB


def test_rss_bytes_never_raises(monkeypatch):
    """/health must not 500 because a stat read failed."""
    monkeypatch.setattr(
        engine_mod.Path, "exists", lambda self: (_ for _ in ()).throw(OSError("nope"))
    )
    assert engine_mod.rss_bytes() is None


# --- the health endpoint -----------------------------------------------------


def _health_client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app import routes as routes_mod

    app = FastAPI()
    app.include_router(routes_mod.router)
    return TestClient(app)


def test_health_exposes_the_memory_fields():
    body = _health_client().get("/health").json()
    assert body["status"] == "ok"
    assert body["rss_mb"] is None or body["rss_mb"] > 1
    assert body["idle_unload_seconds"] == settings.idle_unload_seconds
    # Nothing resident in a test process, so there is no idle clock running.
    assert body["idle_seconds"] is None


def test_health_survives_an_unreadable_rss(monkeypatch):
    from app import routes as routes_mod

    monkeypatch.setattr(routes_mod, "rss_bytes", lambda: None)
    resp = _health_client().get("/health")
    assert resp.status_code == 200
    assert resp.json()["rss_mb"] is None
