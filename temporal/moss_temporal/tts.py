"""HTTP client for the MOSS-TTS server.

The server is `moss-tts-api` (this repo's `app/`), an OpenAI-compatible
`POST /v1/audio/speech` that returns raw audio bytes. Nothing here reaches into
the model: the worker is a client, which is what lets it run on a different
machine from the GPU and lets the existing HTTP API stay untouched.
"""

from __future__ import annotations

import io
import wave

import httpx

from .config import settings
from .shared import CONTENT_TYPES, InvalidRequest

# 24 kHz mono signed 16-bit, per the server's `pcm` response format.
PCM_RATE = 24_000
PCM_BYTES_PER_FRAME = 2


class TTSUnavailable(RuntimeError):
    """The server did not answer, or answered 5xx. Worth retrying."""


def _headers() -> dict[str, str]:
    if settings.tts_api_key:
        return {"Authorization": f"Bearer {settings.tts_api_key}"}
    return {}


def health(timeout: float = 10.0) -> dict:
    """`GET /health`: liveness, device, resident model, RSS, idle clock."""
    try:
        response = httpx.get(f"{settings.tts_url}/health", headers=_headers(), timeout=timeout)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise TTSUnavailable(f"{settings.tts_url}/health: {exc}") from exc
    return response.json()


def synthesize(
    text: str,
    *,
    voice: str,
    model: str,
    response_format: str,
    language: str | None,
    timeout: float,
) -> bytes:
    """One `POST /v1/audio/speech`, returning the audio bytes.

    Blocking on purpose: it runs on the worker's single activity thread, which
    is what serialises access to the one TTS server. The read timeout is the
    caller's clip timeout, so an HTTP client that gave up early can never be
    mistaken for a server that hung.
    """
    payload: dict[str, object] = {
        "input": text,
        "voice": voice,
        "response_format": response_format,
        "model": model,
    }
    if language:
        payload["language"] = language

    try:
        response = httpx.post(
            f"{settings.tts_url}/v1/audio/speech",
            json=payload,
            headers=_headers(),
            # Generous read, short connect: an unreachable host should fail in
            # seconds, a slow generation should not fail at all.
            timeout=httpx.Timeout(timeout, connect=10.0),
        )
    except httpx.HTTPError as exc:
        raise TTSUnavailable(f"{settings.tts_url}/v1/audio/speech: {exc}") from exc

    if response.status_code in (400, 401, 403, 404, 413, 422):
        # The server rejected the request itself. An identical retry gets an
        # identical rejection, so this must not be retried.
        raise InvalidRequest(f"TTS server {response.status_code}: {response.text[:400]}")
    if response.status_code >= 500:
        raise TTSUnavailable(f"TTS server {response.status_code}: {response.text[:400]}")
    response.raise_for_status()

    if not response.content:
        raise TTSUnavailable("TTS server returned 200 with an empty body")
    return response.content


def duration_seconds(data: bytes, response_format: str) -> float | None:
    """Audio length, when it can be read without decoding.

    WAV and PCM are readable from the header (or from arithmetic). MP3 and FLAC
    would need a decoder, and pulling one in to label a number nobody blocks on
    is not worth the dependency — the field is None and says so.
    """
    if response_format == "pcm":
        return round(len(data) / (PCM_RATE * PCM_BYTES_PER_FRAME), 3)
    if response_format == "wav":
        try:
            with wave.open(io.BytesIO(data)) as handle:
                return round(handle.getnframes() / float(handle.getframerate()), 3)
        except (wave.Error, EOFError, ZeroDivisionError):
            return None
    return None


def content_type(response_format: str) -> str:
    return CONTENT_TYPES.get(response_format, "application/octet-stream")
