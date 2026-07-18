# ============================================================================
# CPU-ONLY DEPLOYMENT IMAGE
#
# Built for GPU-less hosts (Railway, Fly, plain VPS): installs CPU torch
# wheels (~600MB image instead of ~6GB CUDA) and defaults to the smallest
# MOSS model (1.7B, bfloat16) so it fits an 8GB-RAM instance.
#
# There is deliberately no CUDA in this image. On a GPU machine, run the
# server directly instead: `uv sync && uv run python -m app.main`.
# ============================================================================
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /bin/uv

WORKDIR /app

# CPU torch wheels: uv checks extra indexes before PyPI, so the torch family
# resolves from the cpu index (~200MB) instead of PyPI's CUDA builds (~6GB).
COPY pyproject.toml ./
# MOSS remote code is written against transformers 5.0.0 (per the model
# cards); 5.14 breaks the 1.7B variant's config class. Pin it here — the
# non-Docker path keeps whatever uv.lock says.
RUN uv pip install --system \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r pyproject.toml \
    "transformers==5.0.0"

COPY app ./app
COPY voices ./voices

# /data is the persistent volume mount point on Railway: HF model cache and
# uploaded voices both survive redeploys. Works without a volume too.
# CPU-sized model defaults (this image has no GPU torch): 1.7B in bfloat16
# fits an 8GB instance; MAX_NEW_TOKENS caps runaway CPU generations (~40s
# audio). Override any of these to go bigger on larger instances.
ENV HF_HOME=/data/hf \
    VOICES_DIR=/data/voices \
    MODEL_ID=OpenMOSS-Team/MOSS-TTS-Local-Transformer \
    DTYPE=bfloat16 \
    MAX_NEW_TOKENS=512 \
    PYTHONUNBUFFERED=1

EXPOSE 8766

CMD ["python", "-m", "app.main"]
