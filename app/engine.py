"""Singleton MOSS-TTS engine.

One model instance per process, lazily loaded on first request and reused
for every call after that. Generation is serialized with a lock — a single
MPS/CUDA device gains nothing from concurrent generate calls, and
interleaving them thrashes memory.
"""

from __future__ import annotations

import io
import logging
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .config import settings

log = logging.getLogger("moss-tts.engine")

# Known MOSS-TTS variants, largest first. Keys are the short names accepted
# in the `model` request field (full HF ids work too). One model is resident
# at a time; requesting a different one swaps it in.
MODELS: dict[str, str] = {
    "moss-tts-v1.5": "OpenMOSS-Team/MOSS-TTS-v1.5",  # 8B — default
    "moss-tts-local-v1.5": "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",  # 4B
    "moss-tts-local": "OpenMOSS-Team/MOSS-TTS-Local-Transformer",  # 1.7B
}
# Env MODEL_ID overrides; ships as the 8B flagship (largest).
DEFAULT_MODEL = settings.model_id


def resolve_model_id(requested: str | None) -> str:
    """Map a request's `model` field to an HF id.

    Accepts short names and full HF ids; anything unknown (e.g. an OpenAI
    client sending "tts-1") falls back to the default largest model so
    stock clients work unmodified.
    """
    if not requested:
        return DEFAULT_MODEL
    if requested in MODELS:
        return MODELS[requested]
    if requested in MODELS.values():
        return requested
    return DEFAULT_MODEL


def _resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(name: str, device: str) -> torch.dtype:
    if name == "auto":
        return torch.bfloat16 if device in ("cuda", "mps") else torch.float32
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


class Engine:
    """One resident model; requesting a different variant swaps it in."""

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._model_id: str | None = None
        self._loading_id: str | None = None
        # One lock for load AND generate: a swap must never run while a
        # generate is in flight (the resident model would be freed mid-use).
        self._lock = threading.RLock()
        self.device = _resolve_device(settings.device)
        self.dtype = _resolve_dtype(settings.dtype, self.device)

    @property
    def loaded_model(self) -> str | None:
        return self._model_id

    @property
    def loading_model(self) -> str | None:
        return self._loading_id

    def ensure_loaded(self, model_id: str) -> None:
        """Blocking. Loads model_id, swapping out any other resident model."""
        if self._model_id == model_id:
            return
        with self._lock:
            if self._model_id == model_id:
                return
            from transformers import AutoModel, AutoProcessor

            self._loading_id = model_id
            try:
                if self._model is not None:
                    log.info("unloading %s", self._model_id)
                    self._model = None
                    self._processor = None
                    self._model_id = None
                    if self.device == "mps":
                        torch.mps.empty_cache()
                    elif self.device == "cuda":
                        torch.cuda.empty_cache()

                t0 = time.time()
                log.info("loading %s on %s (%s)", model_id, self.device, self.dtype)
                processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
                processor.audio_tokenizer = processor.audio_tokenizer.to(self.device)
                model = AutoModel.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    dtype=self.dtype,
                    attn_implementation=settings.attn_implementation,
                ).to(self.device)
                model.eval()
                self._processor = processor
                self._model = model
                self._model_id = model_id
                log.info("model loaded in %.1fs", time.time() - t0)
            finally:
                self._loading_id = None

    def preload_async(self, model_id: str) -> bool:
        """Kick a background load. Returns False if already resident."""
        if self._model_id == model_id:
            return False
        threading.Thread(
            target=self.ensure_loaded, args=(model_id,), daemon=True
        ).start()
        return True

    def synthesize(
        self,
        text: str,
        reference_wav: Path | None = None,
        model_id: str | None = None,
    ) -> tuple[np.ndarray, int]:
        """Blocking. Returns (mono float32 waveform, sample_rate)."""
        with self._lock:
            self.ensure_loaded(model_id or DEFAULT_MODEL)
            processor = self._processor
            reference = [str(reference_wav)] if reference_wav else None
            message = processor.build_user_message(text=text, reference=reference)
            batch = processor([[message]], mode="generation")

            t0 = time.time()
            with torch.no_grad():
                outputs = self._model.generate(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                    max_new_tokens=settings.max_new_tokens,
                )
            log.info("generated in %.1fs", time.time() - t0)

            decoded = processor.decode(outputs)[0]
            # The MPS/CUDA caching allocator never returns freed blocks to
            # the OS on its own; without this the process footprint stays at
            # the high-water mark of the largest request (~10GB+ over the
            # weights).
            if self.device == "mps":
                torch.mps.empty_cache()
            elif self.device == "cuda":
                torch.cuda.empty_cache()
        wav = decoded.audio_codes_list[0].float().cpu().numpy()
        if wav.ndim > 1:
            wav = wav.squeeze()
        return wav, processor.model_config.sampling_rate


def encode_wav(wav: np.ndarray, sample_rate: int, fmt: str) -> bytes:
    """Encode waveform to wav/flac/pcm in-memory. mp3 handled in routes via ffmpeg."""
    if fmt == "pcm":
        return (np.clip(wav, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    sf.write(buf, wav, sample_rate, format=fmt.upper())
    return buf.getvalue()


engine = Engine()
