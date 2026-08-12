"""What has to happen before this process is allowed to exit.

Two things outlive a plain `return` from uvicorn: the idle reaper thread, and
the loky worker pool that joblib keeps warm on behalf of whatever called it
during model load. The pool is the noisier of the two — it holds a POSIX
semaphore per worker, and when a worker is SIGKILLed instead of exiting on its
own, it never gets to unregister that semaphore. multiprocessing's
resource_tracker reclaims it anyway, but announces the reclaim with

    UserWarning: resource_tracker: There appear to be 1 leaked semaphore
    objects to clean up at shutdown: {'/loky-6379-l20y2i09'}

which reads like a crash report and is not one. It is, however, avoidable: a
worker that is asked to exit and does so on its own closes its own semaphore
cleanly, no reclaim, no warning. So shutdown asks nicely first and only kills
if that hangs — the pool is idle by the time this runs (nothing in this app
holds it open across a request), so the graceful path is the common case and
the grace period exists only to bound the wedged one.

Nothing here may raise. A shutdown path that throws turns an orderly stop into
a stack trace, and every one of these steps is best-effort by nature.
"""

from __future__ import annotations

import logging
import threading

log = logging.getLogger(__name__)

# How long to let workers exit on their own before giving up and killing
# them. Idle workers close in well under a second; anything past a few
# seconds means one is wedged, and waiting longer just delays an exit that
# force-killing was already going to have to finish.
GRACE_SECONDS = 3.0


def shutdown_worker_pool() -> bool:
    """Close joblib's reusable loky pool. True if there was one to close.

    joblib is a transitive dependency (librosa pulls it, and the model's
    remote code reaches for it during load), so it may not be importable and
    the pool may never have been created. Both are normal, and neither is
    worth a log line above debug.
    """
    try:
        from joblib.externals.loky import get_reusable_executor
    except Exception:  # joblib absent, or a layout change upstream
        log.debug("no joblib loky pool to shut down")
        return False

    try:
        executor = get_reusable_executor()
    except Exception:
        log.exception("loky pool shutdown failed; exiting anyway")
        return False

    try:
        # A polite drain first: workers that exit on their own unregister
        # their own semaphore, which a SIGKILL never gives them the chance
        # to do. Run it on a thread so a wedged worker can't hang the whole
        # process down here forever -- past the grace period, kill_workers
        # is the fallback that guarantees this returns.
        graceful = threading.Thread(
            target=executor.shutdown,
            kwargs={"wait": True, "kill_workers": False},
            daemon=True,
        )
        graceful.start()
        graceful.join(GRACE_SECONDS)
        if graceful.is_alive():
            log.warning("loky pool did not exit within %ss; killing it", GRACE_SECONDS)
            executor.shutdown(wait=True, kill_workers=True)
    except Exception:
        log.exception("loky pool shutdown failed; exiting anyway")
        return False
    return True


def shutdown(engine) -> None:
    """Stop everything this process started, in the order that is safe."""
    # Reaper first. It can call into the engine, and unloading weights
    # underneath a worker pool that is still running is the one ordering that
    # can deadlock.
    try:
        engine.stop_reaper()
    except Exception:
        log.exception("reaper shutdown failed; continuing")
    shutdown_worker_pool()
    log.info("shutdown complete")
