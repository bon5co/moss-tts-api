"""Tests for the process shutdown path.

The bug these cover is a log-legibility one, not a crash: a worker pool that
is force-killed instead of exiting on its own leaks a semaphore that
multiprocessing's resource_tracker reclaims with a warning that reads like a
fault. Nothing here starts a real pool — what is under test is that the
shutdown path tries a clean exit first, only kills when that hangs, and
survives every one of these steps failing.
"""

from __future__ import annotations

import sys
import threading
import types

import pytest

from app import shutdown as shutdown_mod


class FakeEngine:
    def __init__(self, boom: bool = False):
        self.stopped = False
        self._boom = boom

    def stop_reaper(self):
        if self._boom:
            raise RuntimeError("reaper wedged")
        self.stopped = True


def _install_executor(monkeypatch, executor):
    loky = types.ModuleType("joblib.externals.loky")
    loky.get_reusable_executor = lambda: executor
    externals = types.ModuleType("joblib.externals")
    externals.loky = loky
    joblib = types.ModuleType("joblib")
    joblib.externals = externals

    monkeypatch.setitem(sys.modules, "joblib", joblib)
    monkeypatch.setitem(sys.modules, "joblib.externals", externals)
    monkeypatch.setitem(sys.modules, "joblib.externals.loky", loky)


@pytest.fixture
def fake_joblib(monkeypatch):
    """Install a fake loky executor that exits on its own instantly."""
    calls = []

    class FakeExecutor:
        def shutdown(self, wait, kill_workers):
            calls.append({"wait": wait, "kill_workers": kill_workers})

    _install_executor(monkeypatch, FakeExecutor())
    return calls


@pytest.fixture
def wedged_joblib(monkeypatch):
    """A loky executor whose graceful shutdown never returns on its own."""
    calls = []

    class WedgedExecutor:
        def shutdown(self, wait, kill_workers):
            calls.append({"wait": wait, "kill_workers": kill_workers})
            if not kill_workers:
                threading.Event().wait()  # never set: simulates a hang

    _install_executor(monkeypatch, WedgedExecutor())
    monkeypatch.setattr(shutdown_mod, "GRACE_SECONDS", 0.05)
    return calls


# --- the pool ----------------------------------------------------------------


def test_pool_exits_gracefully_when_it_can(fake_joblib):
    assert shutdown_mod.shutdown_worker_pool() is True
    # A worker that exits on its own unregisters its own semaphore -- no
    # force-kill needed, and no resource_tracker warning.
    assert fake_joblib == [{"wait": True, "kill_workers": False}]


def test_wedged_pool_falls_back_to_kill(wedged_joblib):
    assert shutdown_mod.shutdown_worker_pool() is True
    assert wedged_joblib == [
        {"wait": True, "kill_workers": False},
        {"wait": True, "kill_workers": True},
    ]


def test_absent_joblib_is_not_an_error(monkeypatch):
    """joblib is transitive, so a deploy without it must still exit clean."""
    monkeypatch.setitem(sys.modules, "joblib.externals.loky", None)
    assert shutdown_mod.shutdown_worker_pool() is False


def test_failing_pool_shutdown_is_swallowed(monkeypatch):
    loky = types.ModuleType("joblib.externals.loky")

    def explode():
        raise RuntimeError("pool already dead")

    loky.get_reusable_executor = explode
    monkeypatch.setitem(sys.modules, "joblib.externals.loky", loky)
    assert shutdown_mod.shutdown_worker_pool() is False


# --- the whole path ----------------------------------------------------------


def test_shutdown_stops_reaper_then_pool(fake_joblib):
    eng = FakeEngine()
    shutdown_mod.shutdown(eng)
    assert eng.stopped
    assert fake_joblib == [{"wait": True, "kill_workers": False}]


def test_pool_still_closed_when_reaper_raises(fake_joblib):
    """A wedged reaper must not cost us the pool cleanup that follows it."""
    eng = FakeEngine(boom=True)
    shutdown_mod.shutdown(eng)
    assert fake_joblib == [{"wait": True, "kill_workers": False}]
