"""Workflow definitions.

Kept thin. Temporal supplies the queue, the retry, the cancellation and the
history; the workflow validates the request, walks the lines, and exposes a
query so an agent can ask where a job is without waiting for it.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError, is_cancelled_exception

with workflow.unsafe.imports_passed_through():
    from .config import settings
    from .shared import ClipJob, ClipOut, InvalidRequest, SynthesizeRequest, SynthesizeResult, validate

# Sized off the worst case, not the median. One 33-character line was measured
# at 14.9s / 17.7s / 34s / 54s / 78s back to back, and a cold model load adds
# ~127s from idle (the server unloads after IDLE_UNLOAD_SECONDS, default 900).
# A long line can be minutes. The heartbeat below is what actually detects a
# hung job; this timeout only needs to be too big to fire on a healthy one.
START_TO_CLOSE = timedelta(seconds=settings.clip_timeout_seconds + 60)
# The activity heartbeats every few seconds while the HTTP call is outstanding,
# so a dead worker is noticed in a minute rather than at the timeout above.
# Kept short deliberately. Temporal throttles how often heartbeat *details*
# reach the server to a fraction of this timeout, so a large value would
# mean the elapsed counter below barely moves on a clip that finishes in
# under a minute -- which is most of them.
HEARTBEAT_TIMEOUT = timedelta(seconds=max(30, settings.heartbeat_seconds * 6))


@workflow.defn(name="SynthesizeSpeech")
class SynthesizeSpeechWorkflow:
    def __init__(self) -> None:
        self._state = "queued"
        self._clips: list[ClipOut] = []
        self._num_clips = 0
        self._error: str | None = None

    @workflow.run
    async def run(self, req: SynthesizeRequest) -> SynthesizeResult:
        try:
            validate(req, max_chars=settings.max_chars, max_lines=settings.max_lines)
        except InvalidRequest as exc:
            self._state = "failed"
            self._error = str(exc)
            # Failing here rather than in the activity means a 31-line job with
            # a bad line 30 fails in the first second, not after 29 clips have
            # been generated and paid for.
            raise ApplicationError(str(exc), type="InvalidRequest", non_retryable=True) from exc

        self._num_clips = len(req.lines)
        self._state = "running"
        started = workflow.now()
        prefix = req.prefix.strip("/")
        extension = "raw" if req.response_format == "pcm" else req.response_format

        try:
            for index, line in enumerate(req.lines):
                stem = f"{workflow.info().workflow_id}-{index:03d}.{extension}"
                job = ClipJob(
                    index=index,
                    num_clips=self._num_clips,
                    text=line,
                    voice=req.voice,
                    model=req.model,
                    response_format=req.response_format,
                    language=req.language,
                    key=f"{prefix}/{stem}" if prefix else stem,
                )
                # Sequential, not gathered. The server generates one at a time
                # regardless, so fanning out would only trade a readable
                # progress line for a queue inside someone else's process.
                clip = await workflow.execute_activity(
                    "synthesize_clip",
                    job,
                    # Without this the payload comes back as a plain dict: the
                    # activity is referenced by name, so there is no signature
                    # for the converter to read the return type from.
                    result_type=ClipOut,
                    start_to_close_timeout=START_TO_CLOSE,
                    heartbeat_timeout=HEARTBEAT_TIMEOUT,
                    retry_policy=RetryPolicy(
                        maximum_attempts=3,
                        initial_interval=timedelta(seconds=5),
                        non_retryable_error_types=["InvalidRequest"],
                    ),
                )
                self._clips.append(clip)
        except asyncio.CancelledError:
            # Cooperative: clips already uploaded stay in the bucket, and the
            # status query still answers on a closed workflow, so the caller
            # can see exactly how far it got.
            self._state = "cancelled"
            raise
        except Exception as exc:
            # A cancel that reaches the activity first comes back wrapped in an
            # ActivityError, which is an ordinary Exception -- catching only
            # asyncio.CancelledError above reported a cancelled batch as failed.
            if is_cancelled_exception(exc):
                self._state = "cancelled"
                self._error = None
            else:
                self._state = "failed"
                self._error = str(exc)
            raise

        self._state = "succeeded"
        durations = [c.audio_seconds for c in self._clips]
        return SynthesizeResult(
            clips=self._clips,
            voice=req.voice,
            model=req.model,
            response_format=req.response_format,
            seconds=round((workflow.now() - started).total_seconds(), 3),
            bytes=sum(c.bytes for c in self._clips),
            audio_seconds=(
                round(sum(d for d in durations if d is not None), 3)
                if all(d is not None for d in durations)
                else None
            ),
            language=req.language,
            metadata=req.metadata,
        )

    @workflow.query(name="status")
    def status(self) -> dict:
        """Coarse state plus finished clips. In-clip progress comes from the
        activity heartbeat, which `moss status` reads separately."""
        return {
            "state": self._state,
            "clips_done": len(self._clips),
            "clips_total": self._num_clips,
            "error": self._error,
            "urls": [clip.url for clip in self._clips],
        }
