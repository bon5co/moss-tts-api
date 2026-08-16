#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["requests>=2.31"]
# ///
"""End-to-end check against a running moss-tts-api server.

The pytest suite mocks the processor, so it never downloads weights and never
generates audio: it proves the routing is right and says nothing about whether
this machine can actually load a model. That gap is not theoretical — a server
has been observed answering /health, /v1/models and /v1/voices correctly while
every model load threw two seconds in, which no unit test could have caught.

This script is the other half. It talks to a real server over HTTP, makes it
load real weights and speak real audio, times it, and says plainly which stage
failed. It needs no checkout and no virtualenv:

    uv run scripts/e2e.py --base http://127.0.0.1:8766
    uv run https://raw.githubusercontent.com/bon5co/moss-tts-api/main/scripts/e2e.py

It doubles as the benchmark: the wall-clock cost per second of generated audio
is the number that decides whether a host can carry a daily narration workload,
and it varies by an order of magnitude between machines.

Exit code is 0 only if every stage that ran passed.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import requests

# Roughly five seconds of speech: long enough that generation cost is not lost
# in request overhead, short enough that a slow host still finishes the check.
SAMPLE_TEXT = (
    "This is an end to end check of the speech server. "
    "If you can hear this sentence, the model loaded and generation works."
)

PASS, FAIL, INFO, WARN = "PASS", "FAIL", "info", "warn"


class Report:
    """Stage results, printed as they happen and summarised at the end."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []
        self.failed = False

    def add(self, status: str, stage: str, detail: str = "") -> None:
        self.rows.append((status, stage, detail))
        if status == FAIL:
            self.failed = True
        mark = {PASS: "  ok  ", FAIL: " FAIL ", INFO: "      ", WARN: " warn "}[status]
        print(f"[{mark}] {stage}" + (f" — {detail}" if detail else ""), flush=True)


def wav_duration_seconds(data: bytes) -> float | None:
    """Duration of a RIFF/WAVE payload, or None if it is not one.

    Parsed rather than trusted: a server that returns an HTML error page with a
    200 would otherwise look like a pass, and "we got bytes back" is not the
    same claim as "we got audio back".
    """
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return None
    pos, rate, channels, bits, frames = 12, None, None, None, None
    while pos + 8 <= len(data):
        cid = data[pos : pos + 4]
        size = struct.unpack("<I", data[pos + 4 : pos + 8])[0]
        body = data[pos + 8 : pos + 8 + size]
        if cid == b"fmt " and len(body) >= 16:
            channels, rate = struct.unpack("<HI", body[2:8])
            bits = struct.unpack("<H", body[14:16])[0]
        elif cid == b"data":
            frames = len(body)
        pos += 8 + size + (size & 1)
    if not (rate and channels and bits and frames):
        return None
    return frames / (rate * channels * (bits // 8))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="http://127.0.0.1:8766", help="server base URL")
    ap.add_argument("--api-key", default="", help="bearer token, if the server sets API_KEY")
    ap.add_argument("--model", default="", help="model id to test (default: the server's default)")
    ap.add_argument("--ref", type=Path, help="reference wav; adds a voice-clone stage")
    ap.add_argument("--sfx", action="store_true", help="also test /v1/audio/sound-effect (slow: minutes)")
    ap.add_argument("--out", type=Path, help="write the generated wav here for a listen")
    ap.add_argument(
        "--load-timeout",
        type=float,
        default=600.0,
        help="seconds to wait for a cold model load (default 600; 16GB from a cold cache is slow)",
    )
    ap.add_argument(
        "--gen-timeout",
        type=float,
        default=1800.0,
        help="seconds to wait for generation (default 1800; CPU hosts are extremely slow)",
    )
    args = ap.parse_args()

    base = args.base.rstrip("/")
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    r = Report()
    print(f"moss-tts-api end-to-end check against {base}\n")

    # 1 — reachable, and what hardware it thinks it has.
    try:
        health = requests.get(f"{base}/health", headers=headers, timeout=15).json()
    except Exception as exc:
        r.add(FAIL, "health", f"{type(exc).__name__}: {exc}")
        print("\nserver unreachable — nothing else can be checked.")
        return 1
    device = health.get("device")
    r.add(PASS, "health", f"device={device} dtype={health.get('dtype')} rss={health.get('rss_mb')}MB")
    r.add(
        INFO,
        "idle policy",
        f"idle_unload_seconds={health.get('idle_unload_seconds')} "
        f"(0 = model pinned forever)",
    )
    if device == "cpu":
        r.add(WARN, "device", "running on CPU — generation will be minutes per sentence")

    # 2 — the registry, and which model this run will exercise.
    try:
        models = requests.get(f"{base}/v1/models", headers=headers, timeout=15).json()["data"]
    except Exception as exc:
        r.add(FAIL, "models", f"{type(exc).__name__}: {exc}")
        return 1
    default = next((m["id"] for m in models if m.get("default")), None)
    target = args.model or default
    r.add(PASS, "models", f"{len(models)} registered, default={default}")

    # 3 — the stage that unit tests cannot reach: real weights into real memory.
    #
    # Deliberately via preload rather than by letting the first generate do it,
    # because preload returns immediately and lets us watch /health. A load that
    # *aborts* is otherwise indistinguishable from a load that is still running,
    # and the difference decides whether waiting longer is worth anything.
    started = time.time()
    try:
        pre = requests.post(
            f"{base}/v1/models/preload",
            headers=headers,
            json={"model": target} if args.model else {},
            timeout=30,
        )
        pre.raise_for_status()
    except Exception as exc:
        r.add(FAIL, "preload", f"{type(exc).__name__}: {exc}")
        return 1

    load_seconds = None
    saw_loading = False
    while time.time() - started < args.load_timeout:
        try:
            h = requests.get(f"{base}/health", headers=headers, timeout=15).json()
        except Exception:
            time.sleep(2)
            continue
        if h.get("loading_model"):
            saw_loading = True
        if h.get("loaded_model"):
            load_seconds = time.time() - started
            r.add(
                PASS,
                "model load",
                f"{h['loaded_model']} resident in {load_seconds:.0f}s "
                f"(device_allocated={h.get('device_allocated_mb')}MB)",
            )
            break
        if saw_loading and not h.get("loading_model"):
            # Started, then stopped, with nothing resident: the load raised.
            r.add(
                FAIL,
                "model load",
                f"aborted after {time.time() - started:.0f}s — loading started then stopped with "
                f"nothing resident (device_allocated={h.get('device_allocated_mb')}MB). "
                "The traceback is in the SERVER's stdout; usual causes are a missing or "
                "half-written HuggingFace cache with no network to refetch, a full disk, "
                "or a broken torch/accelerate install. Waiting longer will not help.",
            )
            return 1
        time.sleep(2)
    else:
        r.add(FAIL, "model load", f"still not resident after {args.load_timeout:.0f}s")
        return 1

    # 4 — generation, timed. This is the number that sizes every caller's timeout.
    t0 = time.time()
    try:
        resp = requests.post(
            f"{base}/v1/audio/speech",
            headers=headers,
            json={"input": SAMPLE_TEXT, "voice": "default", "response_format": "wav", "model": target},
            timeout=args.gen_timeout,
        )
    except Exception as exc:
        r.add(FAIL, "speech", f"{type(exc).__name__} after {time.time() - t0:.0f}s: {exc}")
        return 1
    gen_seconds = time.time() - t0
    if resp.status_code != 200:
        r.add(FAIL, "speech", f"HTTP {resp.status_code}: {resp.text[:200]}")
        return 1
    audio_seconds = wav_duration_seconds(resp.content)
    if audio_seconds is None:
        r.add(FAIL, "speech", f"200 OK but the body is not a WAV ({len(resp.content)} bytes)")
        return 1
    ratio = gen_seconds / audio_seconds if audio_seconds else float("inf")
    r.add(
        PASS,
        "speech",
        f"{audio_seconds:.1f}s of audio in {gen_seconds:.0f}s — {ratio:.1f}x wall per audio-second",
    )
    if args.out:
        args.out.write_bytes(resp.content)
        r.add(INFO, "saved", str(args.out))

    # 5 — cloning, only when a reference clip is offered. It is a different code
    # path from /speech (multipart, reference encoder) and fails separately.
    if args.ref:
        if not args.ref.is_file():
            r.add(FAIL, "clone", f"no such reference clip: {args.ref}")
        else:
            t0 = time.time()
            try:
                with args.ref.open("rb") as fh:
                    cl = requests.post(
                        f"{base}/v1/audio/clone",
                        headers=headers,
                        data={"input": SAMPLE_TEXT, "response_format": "wav"},
                        files={"file": (args.ref.name, fh, "audio/wav")},
                        timeout=args.gen_timeout,
                    )
                if cl.status_code != 200:
                    r.add(FAIL, "clone", f"HTTP {cl.status_code}: {cl.text[:200]}")
                else:
                    dur = wav_duration_seconds(cl.content)
                    r.add(
                        PASS if dur else FAIL,
                        "clone",
                        f"{dur:.1f}s of audio in {time.time() - t0:.0f}s" if dur else "body is not a WAV",
                    )
            except Exception as exc:
                r.add(FAIL, "clone", f"{type(exc).__name__}: {exc}")

    # 6 — sound effects: a different model entirely, and slow enough that it is
    # opt-in. Note it evicts the TTS model, so run it last.
    if args.sfx:
        t0 = time.time()
        try:
            fx = requests.post(
                f"{base}/v1/audio/sound-effect",
                headers=headers,
                json={"input": "a single soft chime", "seconds": 2, "num_inference_steps": 20},
                timeout=args.gen_timeout,
            )
            dur = wav_duration_seconds(fx.content) if fx.status_code == 200 else None
            r.add(
                PASS if dur else FAIL,
                "sound-effect",
                f"{dur:.1f}s in {time.time() - t0:.0f}s" if dur else f"HTTP {fx.status_code}: {fx.text[:160]}",
            )
        except Exception as exc:
            r.add(FAIL, "sound-effect", f"{type(exc).__name__}: {exc}")

    print()
    if r.failed:
        print("RESULT: FAIL — see the stages marked FAIL above.")
        return 1
    print(
        "RESULT: PASS — "
        f"load {load_seconds:.0f}s, {ratio:.1f}x wall per audio-second on {device}."
    )
    print(
        "Size client timeouts off that ratio, not off a median: throughput on a shared "
        "machine varies several-fold with whatever else it is doing."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
