"""The worker. Runs wherever the MOSS-TTS server is reachable.

It makes only outbound connections (Temporal long-poll, HTTP to the TTS server,
S3 PUT), so the machine it runs on needs no open ports and no static address.
It does not need torch: the model lives in the TTS server, on the Mac. Moving
synthesis to a different server means changing MOSS_TTS_URL, not moving this.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

from temporalio.client import Client
from temporalio.worker import Worker

from . import storage, tts
from .activities import synthesize_clip
from .config import settings
from .workflows import SynthesizeSpeechWorkflow

log = logging.getLogger("moss.worker")


async def run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    log.info("storage %s bucket=%s", settings.s3_endpoint, settings.s3_bucket)
    try:
        storage.ensure_bucket()
    except storage.StorageAuthError as exc:
        # A stack trace here says nothing the message does not, and buries it.
        log.error("%s", exc)
        raise SystemExit(2) from None

    if settings.tts_startup_check:
        # Fail before advertising on the queue. A worker that has taken a task
        # and only then discovers the TTS server is down burns the retry budget
        # of a job that was never going to run; one that never started leaves
        # the job queued, which is the durable state we wanted.
        try:
            info = tts.health()
        except tts.TTSUnavailable as exc:
            log.error("%s", exc)
            log.error("set MOSS_TTS_URL, or MOSS_TTS_STARTUP_CHECK=0 to start anyway")
            raise SystemExit(2) from None
        log.info(
            "tts %s: %s on %s, loaded=%s idle_unload=%ss",
            settings.tts_url,
            info.get("status"),
            info.get("device"),
            info.get("loaded_model"),
            info.get("idle_unload_seconds"),
        )

    client = await Client.connect(settings.temporal_address, namespace=settings.temporal_namespace)
    log.info(
        "connected to %s ns=%s queue=%s",
        settings.temporal_address,
        settings.temporal_namespace,
        settings.task_queue,
    )

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    # One TTS server, one slot. It generates serially, so a second concurrent
    # request would only make both slower; everything past the one slot is
    # Temporal's backlog, which is durable and survives this process moving.
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="moss") as executor:
        worker = Worker(
            client,
            task_queue=settings.task_queue,
            workflows=[SynthesizeSpeechWorkflow],
            activities=[synthesize_clip],
            activity_executor=executor,
            max_concurrent_activities=1,
            # Heartbeat details are throttled before they reach the server --
            # by default to 30s, capped at 60s. Two things ride on them: the
            # elapsed-seconds progress an agent reads, and the cancellation
            # signal, which is only delivered in a heartbeat *response*. Left
            # at the default, a cancel took longer to arrive than a whole clip
            # takes to generate, so the clip uploaded anyway. One small
            # heartbeat every few seconds is a cheap price for both.
            default_heartbeat_throttle_interval=timedelta(seconds=settings.heartbeat_seconds),
            max_heartbeat_throttle_interval=timedelta(seconds=settings.heartbeat_seconds),
        )
        async with worker:
            log.info("worker running; ctrl-c to stop")
            await stop.wait()
            log.info("draining")


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
