"""The one activity that owns the TTS server."""

from __future__ import annotations

import logging
import threading
import time

from temporalio import activity
from temporalio.exceptions import ApplicationError, CancelledError

from . import storage, tts
from .config import settings
from .shared import ClipJob, ClipOut, InvalidRequest, Progress

log = logging.getLogger(__name__)


@activity.defn(name="synthesize_clip")
def synthesize_clip(job: ClipJob) -> ClipOut:
    """Speak one line, upload the audio, return where it landed.

    Synchronous on purpose: it runs in a single-thread pool, which is what
    serialises access to the one TTS server. The server generates serially
    anyway — a second concurrent request only makes both slower.

    The HTTP call is issued on a helper thread so this one can heartbeat while
    it is outstanding. Generation reports no intermediate progress, so the
    heartbeat carries elapsed time: the only signal that separates "slow, as
    usual" from "wedged", given a measured 5x spread on identical input.
    """
    info = activity.info()
    # Identifies the run, not just the workflow id: an id whose previous run
    # has closed can be reused, and Temporal's default policy starts a fresh
    # run under it. Those two runs must not share an object.
    owner = f"{info.workflow_id}:{info.workflow_run_id}"

    # Before generating, not after. The key check costs one HEAD; discovering
    # the collision after synthesis costs the 15-80s the clip took, thrown away.
    existing = storage.owner_of(job.key)
    if existing is not None and existing != owner:
        raise ApplicationError(
            f"{job.key} already exists, written by {existing or 'an untagged writer'}. "
            f"Submit with a fresh "
            f"--id, or a --prefix that separates the two.",
            type="KeyCollision",
            non_retryable=True,
        )

    started = time.perf_counter()
    result: dict[str, object] = {}

    def call() -> None:
        try:
            result["audio"] = tts.synthesize(
                job.text,
                voice=job.voice,
                model=job.model,
                response_format=job.response_format,
                language=job.language,
                timeout=float(settings.clip_timeout_seconds),
            )
        except BaseException as exc:  # noqa: BLE001 - re-raised on the activity thread
            result["error"] = exc

    thread = threading.Thread(target=call, name=f"moss-tts-{job.index}", daemon=True)
    thread.start()

    while thread.is_alive():
        elapsed = time.perf_counter() - started
        activity.heartbeat(
            Progress(job.index, job.num_clips, len(job.text), "synthesizing", round(elapsed, 1))
        )
        if activity.is_cancelled():
            # Cooperative, and honest about its limit: the server has no cancel
            # endpoint, so the clip already in flight finishes there and is
            # discarded here. What cancellation actually buys is the rest of
            # the batch, which is the expensive part.
            log.info("cancelled while clip %s was in flight; abandoning it", job.index)
            raise CancelledError(f"cancelled during clip {job.index}")
        thread.join(timeout=settings.heartbeat_seconds)

    error = result.get("error")
    if isinstance(error, InvalidRequest):
        # Retrying identical bad input can only fail identically.
        raise ApplicationError(str(error), type="InvalidRequest", non_retryable=True) from error
    if isinstance(error, BaseException):
        raise error

    audio = result["audio"]
    assert isinstance(audio, bytes)
    generate_seconds = round(time.perf_counter() - started, 3)

    if activity.is_cancelled():
        # The clip finished generating in the same moment the cancel landed.
        # Uploading it now would leave an object in the bucket that no result
        # ever references, so drop it. A cancel arriving during the PUT itself
        # still loses this race; the object is harmless, just unreferenced.
        log.info("cancelled after clip %s generated; discarding it", job.index)
        raise CancelledError(f"cancelled after clip {job.index}")

    activity.heartbeat(
        Progress(job.index, job.num_clips, len(job.text), "uploading", generate_seconds)
    )
    content_type = tts.content_type(job.response_format)
    try:
        url = storage.put(job.key, audio, content_type, owner=owner)
    except storage.KeyCollision as exc:
        # Lost the race with another run between the check above and here.
        raise ApplicationError(str(exc), type="KeyCollision", non_retryable=True) from exc
    audio_seconds = tts.duration_seconds(audio, job.response_format)

    log.info(
        "clip %s/%s -> %s (%.1f kB, %.1fs wall, %s audio)",
        job.index + 1,
        job.num_clips,
        url,
        len(audio) / 1024,
        generate_seconds,
        f"{audio_seconds:.1f}s" if audio_seconds is not None else "unknown",
    )

    return ClipOut(
        index=job.index,
        key=job.key,
        url=url,
        text=job.text,
        voice=job.voice,
        bytes=len(audio),
        content_type=content_type,
        audio_seconds=audio_seconds,
        generate_seconds=generate_seconds,
    )
