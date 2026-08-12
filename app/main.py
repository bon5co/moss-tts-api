import logging
import os
import signal
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from .config import settings
from .engine import engine
from .routes import router
from .shutdown import shutdown

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Say who we are at startup. A server that has been idle for hours logs
    # nothing, so the previous line in the journal is whatever happened last
    # -- usually a model load -- and a later shutdown message reads as though
    # it followed the load immediately. Stamping the pid and port makes the
    # two ends of a run identifiable as one run.
    log.info(
        "moss-tts-api up: pid=%s listening on %s:%s", os.getpid(), settings.host, settings.port
    )
    try:
        yield
    finally:
        log.info("shutting down (pid=%s)", os.getpid())
        shutdown(engine)


app = FastAPI(
    title="MOSS-TTS API",
    description="OpenAI-compatible TTS server backed by a local MOSS-TTS model.",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(router)


def _log_signal(signum, _frame):
    """Name the signal that is stopping us, then let uvicorn handle it.

    Without this a SIGHUP from a closed terminal and a SIGTERM from a real
    stop produce byte-identical logs, so "why did the TTS server die" is
    unanswerable after the fact.

    It re-raises as SIGTERM rather than raising KeyboardInterrupt, because
    uvicorn installs a SIGTERM handler that drains connections and runs the
    lifespan shutdown; an exception thrown from a signal handler into the
    event loop can escape past both.
    """
    try:
        name = signal.Signals(signum).name
    except ValueError:
        name = str(signum)
    log.info("received %s, stopping", name)
    signal.raise_signal(signal.SIGTERM)


if __name__ == "__main__":
    # SIGHUP is the one that matters here: the server is usually started from
    # a terminal on a laptop, and closing that terminal (or letting the ssh
    # session that owns it drop) sends HUP, which Python's default handling
    # kills the process for -- silently, with no shutdown hooks run.
    signal.signal(signal.SIGHUP, _log_signal)
    uvicorn.run(app, host=settings.host, port=settings.port)
