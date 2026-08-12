"""Tests for the process shutdown path.

The bug these cover is a log-legibility one, not a crash: a process that exits
without closing joblib's loky pool prints a leaked-semaphore warning that looks
like a fault. Nothing here starts a real pool — what is under test is that the
shutdown path calls the right things, in the right order, and survives every
one of them failing.
"""

from __future__ import annotations

import sys
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


@pytest.fixture
def fake_joblib(monkeypatch):
    """Install a fake joblib.externals.loky exposing a recording executor."""
    calls = {}

    class FakeExecutor:
        def shutdown(self, wait, kill_workers):
            calls["wait"] = wait
            calls["kill_workers"] = kill_workers

    loky = types.ModuleType("joblib.externals.loky")
    loky.get_reusable_executor = lambda: FakeExecutor()
    externals = types.ModuleType("joblib.externals")
    externals.loky = loky
    joblib = types.ModuleType("joblib")
    joblib.externals = externals

    monkeypatch.setitem(sys.modules, "joblib", joblib)
    monkeypatch.setitem(sys.modules, "joblib.externals", externals)
    monkeypatch.setitem(sys.modules, "joblib.externals.loky", loky)
    return calls


# --- the pool ----------------------------------------------------------------


def test_pool_is_shut_down_and_workers_killed(fake_joblib):
    assert shutdown_mod.shutdown_worker_pool() is True
    # Waiting matters: returning before the workers are gone is what leaves
    # the semaphore for the resource_tracker to complain about.
    assert fake_joblib == {"wait": True, "kill_workers": True}


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
    assert fake_joblib["kill_workers"] is True


def test_pool_still_closed_when_reaper_raises(fake_joblib):
    """A wedged reaper must not cost us the pool cleanup that follows it."""
    eng = FakeEngine(boom=True)
    shutdown_mod.shutdown(eng)
    assert fake_joblib["kill_workers"] is True
