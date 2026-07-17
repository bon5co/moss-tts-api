"""OpenAI-compatible audio API.

POST /v1/audio/speech mirrors https://platform.openai.com/docs/api-reference/audio/createSpeech
— any OpenAI SDK pointed at this server with `base_url` works. The `model`
field is accepted but ignored (one model per process); `voice` resolves to a
reference clip in the voices dir for zero-shot cloning, unknown names fall
back to the model's default voice.
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel, Field

from .auth import require_auth
from .config import settings
from .engine import encode_wav, engine

router = APIRouter()

MEDIA_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "flac": "audio/flac",
    "pcm": "audio/pcm",
}


class SpeechRequest(BaseModel):
    model: str = "moss-tts-v1.5"
    input: str = Field(min_length=1, max_length=4096)
    voice: str = "default"
    response_format: str = "wav"
    speed: float = 1.0


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
    wav, sr = await asyncio.to_thread(engine.synthesize, req.input, reference)

    if req.response_format == "mp3":
        body = _to_mp3(encode_wav(wav, sr, "wav"))
    else:
        body = encode_wav(wav, sr, req.response_format)
    return Response(content=body, media_type=MEDIA_TYPES[req.response_format])


@router.get("/v1/models", dependencies=[Depends(require_auth)])
async def list_models() -> dict:
    return {
        "object": "list",
        "data": [
            {
                "id": settings.model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "openmoss",
            }
        ],
    }


@router.get("/v1/voices", dependencies=[Depends(require_auth)])
async def list_voices() -> dict:
    """Not an OpenAI route — lists reference clips available for cloning."""
    names = sorted(p.stem for p in Path(settings.voices_dir).glob("*.wav"))
    return {"voices": ["default", *names]}


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
      "model": "anything",              // accepted, ignored (single model)
      "speed": 1.0                      // only 1.0 supported; else 400
    }

Response body = raw audio bytes (24kHz mono; pcm = s16le). Save to file:

    curl -s -X POST http://localhost:8766/v1/audio/speech \\
      -H "Content-Type: application/json" \\
      -d '{"input": "Hello.", "response_format": "wav"}' -o out.wav

OpenAI SDKs work by overriding base_url to http://localhost:8766/v1.

## Voice cloning

GET /v1/voices lists available voice names. Any name other than "default"
uses that reference clip for zero-shot cloning. New voices = drop a 5-15s
clean WAV into the server's voices/ directory; it is picked up immediately.

## Other routes

GET /v1/models  — OpenAI list-models shape
GET /health     — {status, model, loaded, device, dtype}; first request
                  after startup also loads the model (adds ~25s)

## Errors

Plain JSON: {"detail": "reason"}. 400 = bad parameter (unsupported format
or speed), 401 = missing/wrong bearer token (only when auth is enabled),
422 = schema violation (e.g. empty input), 500 = engine failure.
"""


@router.get("/", response_class=PlainTextResponse)
async def usage_guide() -> str:
    return USAGE_GUIDE


@router.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "model": settings.model_id,
        "loaded": engine.loaded,
        "device": engine.device,
        "dtype": str(engine.dtype),
    }
