"""What has to happen before this process is allowed to exit.

Two things outlive a plain `return` from uvicorn: the idle reaper thread, and
the loky worker pool that joblib keeps warm on behalf of whatever called it
during model load. The pool is the noisier of the two — it holds a POSIX
semaphore per worker, and when the interpreter tears down without closing it,
multiprocessing's resource_tracker prints

    UserWarning: resource_tracker: There appear to be 1 leaked semaphore
    objects to clean up at shutdown: {'/loky-6379-l20y2i09'}

which reads like a crash report and is not one. The semaphore is reclaimed by
the tracker either way; what is lost is any way to tell this line apart from a
real fault in the logs, which is the actual cost.

Nothing here may raise. A shutdown path that throws turns an orderly stop into
a stack trace, and every one of these steps is best-effort by nature.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


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
        # kill_workers, not a polite drain: this runs after the server has
        # stopped serving, so any work still queued belongs to a request
        # nobody is waiting on.
        get_reusable_executor().shutdown(wait=True, kill_workers=True)
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
