"""OpenAI-compatible audio API.

POST /v1/audio/speech mirrors https://platform.openai.com/docs/api-reference/audio/createSpeech
— any OpenAI SDK pointed at this server with `base_url` works, but `model`
must be a MOSS name (or empty for the server default) — anything else is
rejected with 422 listing what's available. `voice` resolves to a reference
clip in the voices dir for zero-shot cloning, unknown names fall back to
the model's default voice.
"""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import soundfile as sf
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel, Field

from .auth import require_auth
from .config import settings
from .engine import DEFAULT_MODEL, MODELS, encode_wav, engine, resolve_model_id

router = APIRouter()

MEDIA_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "flac": "audio/flac",
    "pcm": "audio/pcm",
}


class SpeechRequest(BaseModel):
    # Empty = the server's MODEL_ID default. A concrete short name here
    # would silently swap in that model (the 8B flagship OOMs small hosts)
    # whenever a client omits the field.
    model: str = ""
    input: str = Field(min_length=1, max_length=4096)
    voice: str = "default"
    response_format: str = "wav"
    speed: float = 1.0


def _resolve_model_or_422(requested: str) -> str:
    model_id = resolve_model_id(requested)
    if model_id is None:
        raise HTTPException(
            422,
            f"unknown model '{requested}' — this server only serves MOSS-TTS. "
            f"Available: {', '.join(MODELS)} (or full ids "
            f"{', '.join(MODELS.values())}); omit or send \"\" for the "
            f"server default ({DEFAULT_MODEL}).",
        )
    return model_id


def _resolve_voice(name: str) -> Path | None:
    if name in ("", "default"):
        return None
    path = Path(settings.voices_dir) / f"{name}.wav"
    if not path.is_file():
        return None
    return path


def _to_mp3(wav_bytes: bytes) -> bytes:
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-i", "pipe:0", "-f", "mp3", "-b:a", "128k", "pipe:1"],
        input=wav_bytes, capture_output=True,
    )
    if proc.returncode != 0:
        raise HTTPException(500, f"ffmpeg mp3 encode failed: {proc.stderr.decode()[:200]}")
    return proc.stdout


@router.post("/v1/audio/speech", dependencies=[Depends(require_auth)])
async def create_speech(req: SpeechRequest) -> Response:
    if req.response_format not in MEDIA_TYPES:
        raise HTTPException(
            400,
            f"response_format '{req.response_format}' not supported; "
            f"use one of {sorted(MEDIA_TYPES)}",
        )
    if req.speed != 1.0:
        raise HTTPException(400, "speed is not supported by this backend; use 1.0")

    reference = _resolve_voice(req.voice)
    model_id = _resolve_model_or_422(req.model)
    wav, sr = await asyncio.to_thread(engine.synthesize, req.input, reference, model_id)
    return _audio_response(wav, sr, req.response_format)


def _audio_response(wav, sr: int, fmt: str) -> Response:
    if fmt == "mp3":
        body = _to_mp3(encode_wav(wav, sr, "wav"))
    else:
        body = encode_wav(wav, sr, fmt)
    return Response(content=body, media_type=MEDIA_TYPES[fmt])


@router.post("/v1/audio/clone", dependencies=[Depends(require_auth)])
async def clone_speech(
    input: str = Form(min_length=1, max_length=4096),
    file: UploadFile = File(description="reference clip (wav/mp3/flac), 5-15s"),
    response_format: str = Form("wav"),
    model: str = Form(""),
) -> Response:
    """One-shot voice clone: multipart reference clip + text, no registration."""
    if response_format not in MEDIA_TYPES:
        raise HTTPException(
            400,
            f"response_format '{response_format}' not supported; "
            f"use one of {sorted(MEDIA_TYPES)}",
        )
    suffix = Path(file.filename or "ref.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        ref_path = Path(tmp.name)
    try:
        model_id = _resolve_model_or_422(model)
        wav, sr = await asyncio.to_thread(engine.synthesize, input, ref_path, model_id)
    finally:
        ref_path.unlink(missing_ok=True)
    return _audio_response(wav, sr, response_format)


class PreloadRequest(BaseModel):
    model: str = ""


@router.post("/v1/models/preload", dependencies=[Depends(require_auth)])
async def preload_model(req: PreloadRequest) -> dict:
    """Order a model into memory ahead of the first inference call."""
    model_id = _resolve_model_or_422(req.model)
    started = engine.preload_async(model_id)
    return {
        "model": model_id,
        "status": "loading" if started else "already-loaded",
    }


@router.get("/v1/models", dependencies=[Depends(require_auth)])
async def list_models() -> dict:
    ids = list(MODELS.values())
    if DEFAULT_MODEL not in ids:
        ids.insert(0, DEFAULT_MODEL)
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "openmoss",
                "default": model_id == DEFAULT_MODEL,
                "loaded": engine.loaded_model == model_id,
                "loading": engine.loading_model == model_id,
            }
            for model_id in ids
        ],
    }


@router.get("/v1/voices", dependencies=[Depends(require_auth)])
async def list_voices() -> dict:
    """Not an OpenAI route — lists reference clips available for cloning."""
    voices_dir = Path(settings.voices_dir)
    names = sorted(p.stem for p in voices_dir.glob("*.wav")) if voices_dir.is_dir() else []
    return {"voices": ["default", *names]}


VOICE_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
MAX_VOICE_UPLOAD_BYTES = 20 * 1024 * 1024
MIN_VOICE_SECONDS = 1.0
MAX_VOICE_SECONDS = 60.0


@router.put("/v1/voices/{name}", dependencies=[Depends(require_auth)])
async def put_voice(
    name: str,
    file: UploadFile = File(description="reference clip (wav/mp3/flac), ideally 5-15s"),
) -> dict:
    """Not an OpenAI route — register or replace a named reference clip.

    The clip is transcoded to wav and stored at <voices_dir>/<name>.wav, so
    it survives restarts when voices_dir sits on a persistent volume and is
    immediately usable as `"voice": "<name>"` in /v1/audio/speech.
    """
    if not VOICE_NAME_RE.fullmatch(name):
        raise HTTPException(400, "voice name must match [A-Za-z0-9][A-Za-z0-9_-]{0,63}")
    if name == "default":
        raise HTTPException(400, "'default' is reserved for the model's built-in voice")
    raw = await file.read()
    if len(raw) > MAX_VOICE_UPLOAD_BYTES:
        raise HTTPException(413, "reference clip too large (max 20MB)")

    voices_dir = Path(settings.voices_dir)
    voices_dir.mkdir(parents=True, exist_ok=True)
    # Transcode to wav in a temp file, then atomically replace the target so
    # a concurrent synthesis never reads a half-written clip.
    with tempfile.NamedTemporaryFile(
        suffix=".wav", dir=voices_dir, delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-i", "pipe:0", "-f", "wav", str(tmp_path)],
            input=raw, capture_output=True,
        )
        if proc.returncode != 0:
            raise HTTPException(
                400, f"could not decode reference clip: {proc.stderr.decode()[:200]}"
            )
        info = sf.info(str(tmp_path))
        seconds = info.frames / info.samplerate
        if not MIN_VOICE_SECONDS <= seconds <= MAX_VOICE_SECONDS:
            raise HTTPException(
                400,
                f"reference clip is {seconds:.1f}s; must be "
                f"{MIN_VOICE_SECONDS:g}-{MAX_VOICE_SECONDS:g}s (5-15s works best)",
            )
        os.replace(tmp_path, voices_dir / f"{name}.wav")
    finally:
        tmp_path.unlink(missing_ok=True)
    return {"voice": name, "seconds": round(seconds, 2)}


USAGE_GUIDE = """\
# MOSS-TTS API — usage guide

Local OpenAI-compatible text-to-speech server. One MOSS-TTS model held in
memory; generation is serialized, so send one request at a time and expect
roughly 5-10s of wall time per second of generated audio.

## Synthesize speech

POST /v1/audio/speech            (Content-Type: application/json)

    {
      "input": "Text to speak.",        // required, 1-4096 chars
      "voice": "default",               // or a name from GET /v1/voices
      "response_format": "wav",         // wav | mp3 | flac | pcm
      "model": "",                      // omit/empty = server default; see Models
      "speed": 1.0                      // only 1.0 supported; else 400
    }

Response body = raw audio bytes (24kHz mono; pcm = s16le). Save to file:

    curl -s -X POST http://localhost:8766/v1/audio/speech \\
      -H "Content-Type: application/json" \\
      -d '{"input": "Hello.", "response_format": "wav"}' -o out.wav

OpenAI SDKs work by overriding base_url to http://localhost:8766/v1.

## Voice cloning

Two ways:

1. Named voice: GET /v1/voices lists reference clips; pass one as "voice"
   in /v1/audio/speech. Register a new voice by uploading a 5-15s clean
   clip (wav/mp3/flac) — stored persistently, usable immediately:

    curl -s -X PUT http://localhost:8766/v1/voices/alice \\
      -F file=@ref.wav

   (Or drop a WAV into the server's voices/ directory by hand.)
2. One-shot: POST /v1/audio/clone (multipart/form-data) with fields
   `input` (text), `file` (reference clip), optional `response_format`
   and `model`. Returns audio bytes, nothing is stored.

    curl -s -X POST http://localhost:8766/v1/audio/clone \\
      -F input="Text in the cloned voice." -F file=@ref.wav -o out.wav

## Models

Only MOSS-TTS models are served. Available:

    moss-tts-local        OpenMOSS-Team/MOSS-TTS-Local-Transformer       1.7B
    moss-tts-local-v1.5   OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5  4B (48kHz stereo)
    moss-tts              OpenMOSS-Team/MOSS-TTS                         8B (v1.0)
    moss-tts-v1.5         OpenMOSS-Team/MOSS-TTS-v1.5                    8B

`model` accepts a short name, a full HF id, or empty/omitted for the
server's configured default. Anything else (e.g. "tts-1") = 422 listing
the available models. One model is resident in memory at a time;
requesting a non-resident model on inference loads it first (slow —
swap + load; first ever use also downloads weights).

GET  /v1/models          — registry with default/loaded/loading flags
POST /v1/models/preload  — {"model": "..."} (empty = default); starts a
                           background load, returns immediately with
                           {"status": "loading" | "already-loaded"}

## Other routes

GET /health — {status, default_model, loaded_model, loading_model,
               device, dtype}

## Errors

Plain JSON: {"detail": "reason"}. 400 = bad parameter (unsupported format
or speed), 401 = missing/wrong bearer token (only when auth is enabled),
422 = schema violation (empty input, non-MOSS model name), 500 = engine
failure.
"""


@router.get("/", response_class=PlainTextResponse)
async def usage_guide() -> str:
    return USAGE_GUIDE


@router.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "default_model": DEFAULT_MODEL,
        "loaded_model": engine.loaded_model,
        "loading_model": engine.loading_model,
        "device": engine.device,
        "dtype": str(engine.dtype),
    }
