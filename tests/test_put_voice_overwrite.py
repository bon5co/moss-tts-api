"""Tests for the overwrite semantics of PUT /v1/voices/{name}.

ffmpeg and soundfile are stubbed out — no real transcoding happens, so these
run anywhere. What is under test is which file ends up at
<voices_dir>/<name>.wav, and what the response says about it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import routes as routes_mod
from app.config import settings


class _Info:
    """Stand-in for soundfile.info — an 8s clip, comfortably in range."""

    frames = 8 * 16000
    samplerate = 16000


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A client whose voices_dir is a temp dir and whose ffmpeg is a stub."""
    monkeypatch.setattr(settings, "voices_dir", str(tmp_path))

    real_run = subprocess.run

    def fake_run(cmd, *args, **kwargs):
        # The route calls ffmpeg with `-i pipe:0 ... <tmp_path>`; write the
        # uploaded bytes straight to that destination instead of transcoding.
        if cmd and cmd[0] == "ffmpeg":
            Path(cmd[-1]).write_bytes(kwargs.get("input", b""))
            return subprocess.CompletedProcess(cmd, 0, b"", b"")
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(routes_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(routes_mod.sf, "info", lambda path: _Info())

    app = FastAPI()
    app.include_router(routes_mod.router)
    with TestClient(app) as c:
        c.voices_dir = tmp_path
        yield c


def _put(client, name, content=b"AUDIO", **params):
    return client.put(
        f"/v1/voices/{name}",
        files={"file": ("ref.wav", content, "audio/wav")},
        params=params,
    )


# --- the reported outcome ----------------------------------------------------


def test_first_upload_reports_not_replaced(client):
    resp = _put(client, "alice")
    assert resp.status_code == 200
    assert resp.json()["replaced"] is False
    assert (client.voices_dir / "alice.wav").read_bytes() == b"AUDIO"


def test_same_name_again_overwrites_and_reports_replaced(client):
    _put(client, "alice", b"FIRST")
    resp = _put(client, "alice", b"SECOND")
    assert resp.status_code == 200
    assert resp.json()["replaced"] is True
    # The old clip is gone — this endpoint keeps no history.
    assert (client.voices_dir / "alice.wav").read_bytes() == b"SECOND"


def test_names_are_case_sensitive(client):
    """`Alice` and `alice` are two voices, not one."""
    _put(client, "alice", b"LOWER")
    resp = _put(client, "Alice", b"UPPER")
    assert resp.json()["replaced"] is False
    assert (client.voices_dir / "alice.wav").read_bytes() == b"LOWER"
    assert (client.voices_dir / "Alice.wav").read_bytes() == b"UPPER"


# --- the guard ---------------------------------------------------------------


def test_overwrite_false_on_a_new_name_still_creates_it(client):
    resp = _put(client, "alice", overwrite="false")
    assert resp.status_code == 200
    assert resp.json()["replaced"] is False


def test_overwrite_false_refuses_an_existing_name(client):
    _put(client, "alice", b"FIRST")
    resp = _put(client, "alice", b"SECOND", overwrite="false")
    assert resp.status_code == 409
    assert "already exists" in resp.json()["detail"]
    assert (client.voices_dir / "alice.wav").read_bytes() == b"FIRST"


def test_overwrite_defaults_to_true(client):
    """Omitting the param keeps the historical replace-without-asking behaviour."""
    _put(client, "alice", b"FIRST")
    assert _put(client, "alice", b"SECOND").status_code == 200


# --- a rejected upload must not damage what is already there ------------------


def test_undecodable_upload_leaves_the_existing_clip_alone(client, monkeypatch):
    _put(client, "alice", b"GOOD")

    def failing_ffmpeg(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, b"", b"Invalid data found")

    monkeypatch.setattr(routes_mod.subprocess, "run", failing_ffmpeg)
    resp = _put(client, "alice", b"GARBAGE")
    assert resp.status_code == 400
    assert (client.voices_dir / "alice.wav").read_bytes() == b"GOOD"


def test_out_of_range_duration_leaves_the_existing_clip_alone(client, monkeypatch):
    _put(client, "alice", b"GOOD")

    class _TooShort:
        frames = 100
        samplerate = 16000

    monkeypatch.setattr(routes_mod.sf, "info", lambda path: _TooShort())
    resp = _put(client, "alice", b"BLIP")
    assert resp.status_code == 400
    assert (client.voices_dir / "alice.wav").read_bytes() == b"GOOD"


def test_failed_upload_leaves_no_temp_files_behind(client, monkeypatch):
    _put(client, "alice", b"GOOD")

    monkeypatch.setattr(
        routes_mod.subprocess,
        "run",
        lambda cmd, *a, **k: subprocess.CompletedProcess(cmd, 1, b"", b"boom"),
    )
    _put(client, "alice", b"GARBAGE")
    assert [p.name for p in client.voices_dir.iterdir()] == ["alice.wav"]


# --- unchanged rejections ----------------------------------------------------


def test_reserved_name_still_rejected(client):
    assert _put(client, "default").status_code == 400


def test_invalid_name_still_rejected(client):
    assert _put(client, "bad!name").status_code == 400
