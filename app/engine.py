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
    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._load_lock = threading.Lock()
        self._generate_lock = threading.Lock()
        self.device = _resolve_device(settings.device)
        self.dtype = _resolve_dtype(settings.dtype, self.device)

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            from transformers import AutoModel, AutoProcessor

            t0 = time.time()
            log.info("loading %s on %s (%s)", settings.model_id, self.device, self.dtype)
            processor = AutoProcessor.from_pretrained(
                settings.model_id, trust_remote_code=True
            )
            processor.audio_tokenizer = processor.audio_tokenizer.to(self.device)
            model = AutoModel.from_pretrained(
                settings.model_id,
                trust_remote_code=True,
                dtype=self.dtype,
                attn_implementation=settings.attn_implementation,
            ).to(self.device)
            model.eval()
            self._processor = processor
            self._model = model
            log.info("model loaded in %.1fs", time.time() - t0)

    def synthesize(self, text: str, reference_wav: Path | None = None) -> tuple[np.ndarray, int]:
        """Blocking. Returns (mono float32 waveform, sample_rate)."""
        self._ensure_loaded()
        processor = self._processor
        reference = [str(reference_wav)] if reference_wav else None
        message = processor.build_user_message(text=text, reference=reference)
        batch = processor([[message]], mode="generation")

        with self._generate_lock:
            t0 = time.time()
            with torch.no_grad():
                outputs = self._model.generate(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                    max_new_tokens=settings.max_new_tokens,
                )
            log.info("generated in %.1fs", time.time() - t0)

        decoded = processor.decode(outputs)[0]
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
