"""Environment-driven settings.

Read once at import. Every value has a default that works for the intended
layout: Temporal and MinIO in Docker on the `raphael` host, the MOSS-TTS server
on a Mac reachable over the tailnet, and this worker anywhere that can reach
both.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv() -> None:
    """Read `.env` into the environment, without overriding what is already set.

    Nothing else does this. `uv run` does not load `.env`, so a worker started
    with a `.env` full of correct values would still sign its S3 requests with
    the empty default below and fail with `SignatureDoesNotMatch` — an error
    that points at the key rather than at the file that was never read. This is
    lifted from zimage-temporal, where omitting it cost a debugging cycle.

    Real environment variables win, so a launchd unit or a `FOO=bar moss ...`
    prefix still overrides the file.
    """
    for candidate in (Path.cwd() / ".env", Path(__file__).resolve().parent.parent / ".env"):
        if not candidate.is_file():
            continue
        for line in candidate.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            # Strip an inline comment, then matching quotes.
            value = value.split("#", 1)[0].strip() if not value.strip().startswith(("'", '"')) else value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            if key and key not in os.environ:
                os.environ[key] = value
        return


_load_dotenv()


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def _env_any(names: tuple[str, ...], default: str) -> str:
    """First of `names` that is set and non-empty, else `default`.

    Exists so the S3 settings can fall back to `ZIMAGE_S3_*`. It is the same
    MinIO on the same host with the same credentials — only the bucket differs
    — and `~/.env` already exports the zimage names machine-wide. Requiring a
    second copy of the same secret under a new name would be ceremony that
    invites the two to drift.
    """
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return default


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, "1" if default else "0").lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # --- Temporal ---
    temporal_address: str = field(default_factory=lambda: _env("MOSS_TEMPORAL_ADDRESS", "raphael.local:7233"))
    temporal_namespace: str = field(default_factory=lambda: _env("MOSS_TEMPORAL_NAMESPACE", "moss"))
    task_queue: str = field(default_factory=lambda: _env("MOSS_TASK_QUEUE", "moss-tts"))

    # --- the TTS server ---
    tts_url: str = field(
        default_factory=lambda: _env("MOSS_TTS_URL", "http://calicoagent.tail200737.ts.net:8766").rstrip("/")
    )
    tts_api_key: str = field(default_factory=lambda: _env("MOSS_TTS_API_KEY", ""))
    tts_startup_check: bool = field(default_factory=lambda: _env_bool("MOSS_TTS_STARTUP_CHECK", True))

    # --- object storage ---
    # Endpoint the *worker* writes through.
    s3_endpoint: str = field(
        default_factory=lambda: _env_any(("MOSS_S3_ENDPOINT", "ZIMAGE_S3_ENDPOINT"), "http://raphael.local:9100")
    )
    # Endpoint baked into returned URLs. Usually identical; differs when the
    # worker writes over the tailnet but consumers read over the LAN.
    s3_public_endpoint: str = field(default_factory=lambda: _env("MOSS_S3_PUBLIC_ENDPOINT", ""))
    s3_bucket: str = field(default_factory=lambda: _env("MOSS_S3_BUCKET", "moss"))
    s3_region: str = field(default_factory=lambda: _env_any(("MOSS_S3_REGION", "ZIMAGE_S3_REGION"), "us-east-1"))
    s3_access_key: str = field(
        default_factory=lambda: _env_any(("MOSS_S3_ACCESS_KEY", "ZIMAGE_S3_ACCESS_KEY"), "zimage")
    )
    s3_secret_key: str = field(default_factory=lambda: _env_any(("MOSS_S3_SECRET_KEY", "ZIMAGE_S3_SECRET_KEY"), ""))
    # The bucket is anonymous-read by default, so URLs are plain and permanent.
    # Set to 1 to hand back presigned URLs instead (bucket must then be private).
    s3_presign: bool = field(default_factory=lambda: _env_bool("MOSS_S3_PRESIGN", False))
    s3_presign_ttl: int = field(default_factory=lambda: _env_int("MOSS_S3_PRESIGN_TTL", 7 * 24 * 3600))

    # --- timeouts ---
    # One clip. Sized off the worst case, not the median: measured generations
    # of one identical 33-character line ran 14.9s / 17.7s / 34s / 54s / 78s,
    # and a cold model load adds ~127s on top of that. See the README.
    clip_timeout_seconds: int = field(default_factory=lambda: _env_int("MOSS_CLIP_TIMEOUT_SECONDS", 1200))
    # How often the activity heartbeats while the HTTP call is outstanding.
    heartbeat_seconds: int = field(default_factory=lambda: _env_int("MOSS_HEARTBEAT_SECONDS", 5))

    # --- limits (rejected as non-retryable errors, not clamped silently) ---
    # 4096 is the server's own `input` field cap; matching it here turns a
    # 422 into a message that names the line number.
    max_chars: int = field(default_factory=lambda: _env_int("MOSS_MAX_CHARS", 4096))
    max_lines: int = field(default_factory=lambda: _env_int("MOSS_MAX_LINES", 200))

    @property
    def public_endpoint(self) -> str:
        return self.s3_public_endpoint or self.s3_endpoint


settings = Settings()
