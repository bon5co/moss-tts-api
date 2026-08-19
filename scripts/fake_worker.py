"""A worker that runs the real workflow with fake audio.

Exists so the plumbing -- task queue, payload serialisation, the status query,
heartbeat progress, the upload, the returned URL, cancellation -- can be
exercised on a machine that cannot reach the TTS server. It writes a silent WAV
with the standard library and nothing else.

    uv run python scripts/fake_worker.py

Never register this against a queue a real worker is also polling: whichever
picks the task up wins, and half your clips would be silence.
"""

from __future__ import annotations

import asyncio
import io
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

from temporalio import activity
from temporalio.client import Client
from temporalio.exceptions import CancelledError
from temporalio.worker import Worker

from moss_temporal import storage, tts
from moss_temporal.config import settings
from moss_temporal.shared import ClipJob, ClipOut, Progress
from moss_temporal.workflows import SynthesizeSpeechWorkflow

# Roughly the real thing's pace: a syllable of audio per few characters.
SECONDS_PER_CHAR = 0.08
# How long the fake "generation" takes per character of input.
WALL_PER_CHAR = 0.05


def silent_wav(seconds: float, rate: int = 24_000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * int(rate * seconds))
    return buffer.getvalue()


@activity.defn(name="synthesize_clip")
def synthesize_clip(job: ClipJob) -> ClipOut:
    started = time.perf_counter()
    wall = max(1.0, len(job.text) * WALL_PER_CHAR)
    while time.perf_counter() - started < wall:
        elapsed = time.perf_counter() - started
        activity.heartbeat(Progress(job.index, job.num_clips, len(job.text), "synthesizing", round(elapsed, 1)))
        if activity.is_cancelled():
            raise CancelledError(f"cancelled during clip {job.index}")
        time.sleep(0.5)

    data = silent_wav(max(0.2, len(job.text) * SECONDS_PER_CHAR))
    activity.heartbeat(Progress(job.index, job.num_clips, len(job.text), "uploading", wall))
    content_type = tts.content_type(job.response_format)
    info = activity.info()
    url = storage.put(
        job.key, data, content_type, owner=f"{info.workflow_id}:{info.workflow_run_id}"
    )
    return ClipOut(
        index=job.index,
        key=job.key,
        url=url,
        text=job.text,
        voice=job.voice,
        bytes=len(data),
        content_type=content_type,
        audio_seconds=tts.duration_seconds(data, job.response_format),
        generate_seconds=round(time.perf_counter() - started, 3),
    )


async def main() -> None:
    storage.ensure_bucket()
    client = await Client.connect(settings.temporal_address, namespace=settings.temporal_namespace)
    with ThreadPoolExecutor(max_workers=1) as executor:
        async with Worker(
            client,
            task_queue=settings.task_queue,
            workflows=[SynthesizeSpeechWorkflow],
            activities=[synthesize_clip],
            activity_executor=executor,
            max_concurrent_activities=1,
            default_heartbeat_throttle_interval=timedelta(seconds=settings.heartbeat_seconds),
            max_heartbeat_throttle_interval=timedelta(seconds=settings.heartbeat_seconds),
        ):
            print(f"fake worker polling {settings.task_queue}; ctrl-c to stop", flush=True)
            await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
