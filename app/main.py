import logging

import uvicorn
from fastapi import FastAPI

from .config import settings
from .routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = FastAPI(
    title="MOSS-TTS API",
    description="OpenAI-compatible TTS server backed by a local MOSS-TTS model.",
    version="0.1.0",
)
app.include_router(router)


if __name__ == "__main__":
    uvicorn.run(app, host=settings.host, port=settings.port)
