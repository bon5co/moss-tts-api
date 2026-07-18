"""Singleton MOSS-TTS engine.

One model instance per process, lazily loaded on first request and reused
for every call after that. Generation is serialized with a lock — a single
MPS/CUDA device gains nothing from concurrent generate calls, and
interleaving them thrashes memory.
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
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
    "moss-tts-v1.5": "OpenMOSS-Team/MOSS-TTS-v1.5",  # 8B, MossTTSDelay
    "moss-tts": "OpenMOSS-Team/MOSS-TTS",  # 8B, MossTTSDelay (1.0)
    "moss-tts-local-v1.5": "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",  # 4B, MossTTSLocal, 48kHz stereo
    "moss-tts-local": "OpenMOSS-Team/MOSS-TTS-Local-Transformer",  # 1.7B, MossTTSLocal
}
# Text-to-sound-effect models (DiT + flow matching, MossSoundEffectPipeline
# interface — not the TTS AutoModel path). Same single-resident rule: loading
# one swaps out whatever TTS model is resident and vice versa.
SOUND_EFFECT_MODELS: dict[str, str] = {
    "moss-soundeffect-v2.0": "OpenMOSS-Team/MOSS-SoundEffect-v2.0",  # 1.3B DiT, 48kHz
}

# Env MODEL_ID overrides; ships as the 8B flagship (largest).
DEFAULT_MODEL = settings.model_id
DEFAULT_SOUND_EFFECT_MODEL = next(iter(SOUND_EFFECT_MODELS.values()))


def resolve_sound_effect_model_id(requested: str | None) -> str | None:
    if not requested:
        return DEFAULT_SOUND_EFFECT_MODEL
    if requested in SOUND_EFFECT_MODELS:
        return SOUND_EFFECT_MODELS[requested]
    if requested in SOUND_EFFECT_MODELS.values():
        return requested
    return None


def resolve_model_id(requested: str | None) -> str | None:
    """Map a request's `model` field to an HF id.

    Empty/None = the server's configured default. Accepts short names and
    full HF ids; anything else returns None — only MOSS models are served,
    so callers reject unknown names (422) instead of silently substituting.
    """
    if not requested:
        return DEFAULT_MODEL
    if requested in MODELS:
        return MODELS[requested]
    if requested in MODELS.values():
        return requested
    return None


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


_sfx_mps_patched = False


def _patch_sfx_float64_for_mps() -> None:
    """Replace the sound-effect DiT's float64 timestep embedding with a
    float32 version — MPS has no float64, and at these magnitudes (timesteps
    0-1000, dim ~256) float32 is exact enough that outputs are unaffected.
    """
    global _sfx_mps_patched
    if _sfx_mps_patched:
        return

    def sinusoidal_embedding_1d_f32(dim, position):
        sinusoid = torch.outer(
            position.type(torch.float32),
            torch.pow(
                10000,
                -torch.arange(dim // 2, dtype=torch.float32, device=position.device)
                .div(dim // 2),
            ),
        )
        x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
        return x.to(position.dtype)

    import sys

    from moss_soundeffect_v2.diffsynth.models import wan_video_dit

    original = wan_video_dit.sinusoidal_embedding_1d
    # `from ... import sinusoidal_embedding_1d` copies the binding into
    # several modules (wan_audio_dit, pipelines/wan_audio, ...); rebind the
    # name in every loaded module that holds the original object.
    for module in list(sys.modules.values()):
        if module is None or not getattr(module, "__name__", "").startswith(
            "moss_soundeffect_v2"
        ):
            continue
        for attr, val in list(vars(module).items()):
            if val is original:
                setattr(module, attr, sinusoidal_embedding_1d_f32)
    _sfx_mps_patched = True


class Engine:
    """One resident model; requesting a different variant swaps it in."""

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._model_id: str | None = None
        self._kind: str | None = None
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

    def _unload_resident(self) -> None:
        if self._model is None:
            return
        log.info("unloading %s", self._model_id)
        self._model = None
        self._processor = None
        self._model_id = None
        self._kind = None
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()

    def ensure_loaded(self, model_id: str, kind: str = "tts") -> None:
        """Blocking. Loads model_id, swapping out any other resident model."""
        if self._model_id == model_id:
            return
        with self._lock:
            if self._model_id == model_id:
                return
            self._loading_id = model_id
            try:
                self._unload_resident()
                t0 = time.time()
                log.info("loading %s (%s) on %s (%s)", model_id, kind, self.device, self.dtype)
                if kind == "sfx":
                    # Triton/CUDA-graph compile path breaks off-CUDA.
                    if self.device != "cuda":
                        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
                    from moss_soundeffect_v2 import MossSoundEffectPipeline

                    if self.device == "mps":
                        _patch_sfx_float64_for_mps()
                        # The DiT keeps complex128 RoPE buffers, which MPS
                        # can't hold. Assemble on CPU, downcast, then move.
                        pipe = MossSoundEffectPipeline.from_pretrained(
                            model_id, torch_dtype=self.dtype, device="cpu"
                        )
                        narrowing = {
                            torch.float64: torch.float32,
                            torch.complex128: torch.complex64,
                        }
                        for buf_module in pipe.engine.modules():
                            for name, buf in list(buf_module.named_buffers(recurse=False)):
                                if buf.dtype in narrowing:
                                    buf_module.register_buffer(
                                        name, buf.to(narrowing[buf.dtype]), persistent=False
                                    )
                        for p in pipe.engine.parameters():
                            if p.dtype in narrowing:
                                p.data = p.data.to(narrowing[p.dtype])
                        pipe.to(self.device)
                        self._model = pipe
                    else:
                        self._model = MossSoundEffectPipeline.from_pretrained(
                            model_id, torch_dtype=self.dtype, device=self.device
                        )
                else:
                    from transformers import AutoModel, AutoProcessor

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
                self._kind = kind
                log.info("model loaded in %.1fs", time.time() - t0)
            finally:
                self._loading_id = None

    def preload_async(self, model_id: str, kind: str = "tts") -> bool:
        """Kick a background load. Returns False if already resident."""
        if self._model_id == model_id:
            return False
        threading.Thread(
            target=self.ensure_loaded, args=(model_id, kind), daemon=True
        ).start()
        return True

    def synthesize_sound_effect(
        self,
        prompt: str,
        seconds: float,
        model_id: str | None = None,
        num_inference_steps: int = 100,
        cfg_scale: float = 4.0,
    ) -> tuple[np.ndarray, int]:
        """Blocking. Returns (waveform, sample_rate) — 48kHz from the DiT."""
        with self._lock:
            self.ensure_loaded(model_id or DEFAULT_SOUND_EFFECT_MODEL, kind="sfx")
            t0 = time.time()
            audio = self._model(
                prompt=prompt,
                seconds=seconds,
                num_inference_steps=num_inference_steps,
                cfg_scale=cfg_scale,
            )
            log.info("sound effect generated in %.1fs", time.time() - t0)
            # save_audio is the pipeline's only documented output contract;
            # round-trip through a temp wav rather than poke at internals.
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                self._model.save_audio(audio, tmp_path)
                wav, sr = sf.read(tmp_path, dtype="float32")
            finally:
                Path(tmp_path).unlink(missing_ok=True)
            if self.device == "mps":
                torch.mps.empty_cache()
            elif self.device == "cuda":
                torch.cuda.empty_cache()
        if wav.ndim > 1:
            wav = wav.T.squeeze()
        return wav, sr

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
