"""Types crossing the workflow/activity boundary.

Deliberately free of heavy imports: the workflow sandbox imports this module,
and so does the CLI, which should not need boto3 or httpx loaded to print a
workflow id.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Formats the server will encode to. `pcm` is headerless 24 kHz mono s16le.
FORMATS = ("wav", "mp3", "flac", "pcm")

CONTENT_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "flac": "audio/flac",
    "pcm": "audio/pcm",
}


@dataclass
class SynthesizeRequest:
    """One synthesis job: N lines of text spoken in one voice.

    `lines` is a list because the unit of work an agent actually has is a
    script, not a sentence — see the README on why a 31-line story is one
    workflow rather than 31.
    """

    lines: list[str] = field(default_factory=list)
    voice: str = "default"
    # Empty = the server's own MODEL_ID default. Naming a model here would swap
    # the resident one, which costs a reload for everything else using it.
    model: str = ""
    response_format: str = "wav"
    # Plain language name ("Japanese", "Thai"). None lets the model infer it.
    language: str | None = None
    # Key prefix inside the bucket, e.g. "bedtime/ep014".
    prefix: str = ""
    # Free-form, echoed back on the result. For agent-side correlation.
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class ClipJob:
    """What one activity attempt needs. Derived from the request by the workflow."""

    index: int
    num_clips: int
    text: str
    voice: str
    model: str
    response_format: str
    language: str | None
    key: str


@dataclass
class ClipOut:
    index: int
    key: str
    url: str
    text: str
    voice: str
    bytes: int
    content_type: str
    # Length of the audio itself. None when the format carries no frame count
    # this client can read without decoding (mp3, flac).
    audio_seconds: float | None = None
    # Wall time the server took. Recorded because it is the number that decides
    # whether a batch is worth running at all, and it varies by 5x.
    generate_seconds: float = 0.0


@dataclass
class SynthesizeResult:
    clips: list[ClipOut]
    voice: str
    model: str
    response_format: str
    seconds: float
    bytes: int
    audio_seconds: float | None = None
    language: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class Progress:
    """Heartbeat payload. Read by `moss status` while a job is running."""

    clip_index: int = 0
    num_clips: int = 1
    chars: int = 0
    # waiting | synthesizing | uploading
    phase: str = "waiting"
    # Seconds this clip has been outstanding. The only signal that separates
    # "slow, as usual" from "the server wedged", since generation reports no
    # intermediate progress of its own.
    elapsed: float = 0.0


class InvalidRequest(Exception):
    """Bad parameters. Registered non-retryable — retrying cannot fix it."""


def validate(req: SynthesizeRequest, *, max_chars: int, max_lines: int) -> None:
    """Reject what the server would reject, before anything is queued.

    Runs in the workflow rather than the activity, unlike zimage-temporal. With
    a batch the difference matters: a 31-line job whose line 30 is empty should
    fail in the first second, not after 29 clips have been generated and paid
    for.
    """
    if not req.lines:
        raise InvalidRequest("no lines to speak")
    if len(req.lines) > max_lines:
        raise InvalidRequest(f"{len(req.lines)} lines exceeds MOSS_MAX_LINES={max_lines}")
    if req.response_format not in FORMATS:
        raise InvalidRequest(f"response_format {req.response_format!r} not in {list(FORMATS)}")
    for i, line in enumerate(req.lines):
        if not line.strip():
            raise InvalidRequest(f"line {i} is empty")
        if len(line) > max_chars:
            raise InvalidRequest(f"line {i} is {len(line)} chars, over MOSS_MAX_CHARS={max_chars}")
