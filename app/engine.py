"""Singleton MOSS-TTS engine.

One model instance per process, lazily loaded on first request and reused
for every call after that. Generation is serialized with a lock — a single
MPS/CUDA device gains nothing from concurrent generate calls, and
interleaving them thrashes memory.
"""

from __future__ import annotations

import gc
import inspect
import io
import logging
import os
import subprocess
import sys
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
VOICE_GENERATOR_MODEL = "OpenMOSS-Team/MOSS-VoiceGenerator"


def _build_user_message(processor, *, text: str, reference, language: str | None):
    """Call processor.build_user_message, passing `language` only when set.

    Upstream MOSS-TTS supports `build_user_message(text=..., language=...)`,
    but not every processor/model revision does. When `language` is None the
    call is byte-for-byte the historical one. When it is set but the
    processor cannot take it, fall back to the language-less call with a
    warning rather than 500-ing the request: ignoring the hint is strictly
    better than failing to synthesize.
    """
    if language is None:
        return processor.build_user_message(text=text, reference=reference)

    build = processor.build_user_message
    try:
        params = inspect.signature(build).parameters
        accepts_language = "language" in params or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
    except (TypeError, ValueError):  # builtins / C-extensions have no signature
        accepts_language = True

    if not accepts_language:
        log.warning(
            "processor %s.build_user_message does not accept `language`; "
            "ignoring requested language %r",
            type(processor).__name__,
            language,
        )
        return build(text=text, reference=reference)

    try:
        return build(text=text, reference=reference, language=language)
    except TypeError:
        log.warning(
            "processor %s.build_user_message rejected `language`; retrying "
            "without it (requested language %r)",
            type(processor).__name__,
            language,
            exc_info=True,
        )
        return build(text=text, reference=reference)


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


def rss_bytes() -> int | None:
    """Current resident set size, or None where it can't be determined.

    Deliberately dependency-free: /proc on Linux, `ps` on macOS (where the
    server actually runs, on MPS). Only ever called from /health.
    """
    try:
        statm = Path("/proc/self/statm")
        if statm.exists():
            pages = int(statm.read_text().split()[1])
            return pages * os.sysconf("SC_PAGE_SIZE")
        if sys.platform == "darwin":
            out = subprocess.run(
                ["ps", "-o", "rss=", "-p", str(os.getpid())],
                capture_output=True, text=True, timeout=5,
            )
            if out.returncode == 0 and out.stdout.strip():
                return int(out.stdout.strip()) * 1024
    except Exception:  # a health endpoint must never 500 over a stat read
        log.debug("could not read RSS", exc_info=True)
    return None


class Engine:
    """One resident model; requesting a different variant swaps it in.

    A model also drops out on its own after settings.idle_unload_seconds
    without a generate — see _reaper_loop. Weights are big enough (~16GB for
    the 8B flagship) that holding them through an idle night is the larger
    cost, and on MPS that memory is taken from the whole machine.
    """

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._model_id: str | None = None
        self._kind: str | None = None
        self._loading_id: str | None = None
        # One lock for load AND generate: a swap must never run while a
        # generate is in flight (the resident model would be freed mid-use).
        self._lock = threading.RLock()
        # Monotonic, so a wall-clock jump (NTP step, laptop sleep) can't make
        # a busy model look idle for hours. None = nothing resident.
        self._last_used: float | None = None
        self._reaper: threading.Thread | None = None
        self._stop_reaper = threading.Event()
        self.device = _resolve_device(settings.device)
        self.dtype = _resolve_dtype(settings.dtype, self.device)

    @property
    def loaded_model(self) -> str | None:
        return self._model_id

    @property
    def loading_model(self) -> str | None:
        return self._loading_id

    @property
    def idle_seconds(self) -> float | None:
        """Seconds since the last generate, or None if nothing is resident."""
        if self._model is None or self._last_used is None:
            return None
        return time.monotonic() - self._last_used

    def _touch(self) -> None:
        self._last_used = time.monotonic()

    def _unload_resident(self) -> None:
        if self._model is None:
            return
        log.info("unloading %s", self._model_id)
        self._model = None
        self._processor = None
        self._model_id = None
        self._kind = None
        self._last_used = None
        # Order matters. empty_cache() only returns blocks with no live
        # tensors, and transformers models sit in reference cycles (hooks,
        # config back-refs) that refcounting alone does not break — so
        # without this collect the weights are still reachable when the
        # cache is swept, nothing is freed, and the next model loads on top
        # of the old one. That is how a 16GB swap becomes a 32GB peak.
        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()

    # --- idle reaper ---------------------------------------------------------

    def _reaper_tick(self) -> float:
        """Poll interval: often enough to be punctual, rarely enough to be free."""
        timeout = settings.idle_unload_seconds
        return max(1.0, min(30.0, timeout / 4)) if timeout > 0 else 30.0

    def unload_if_idle(self) -> bool:
        """Drop the resident model if it has gone untouched. Returns whether it did."""
        timeout = settings.idle_unload_seconds
        if timeout <= 0:
            return False
        idle = self.idle_seconds
        if idle is None or idle < timeout:
            return False
        with self._lock:
            # Re-check under the lock: waiting for it may have meant waiting
            # out a long generate, which just reset the clock.
            idle = self.idle_seconds
            if idle is None or idle < settings.idle_unload_seconds:
                return False
            log.info("unloading %s after %.0fs idle", self._model_id, idle)
            self._unload_resident()
            return True

    def _reaper_loop(self) -> None:
        while not self._stop_reaper.wait(self._reaper_tick()):
            try:
                self.unload_if_idle()
            except Exception:  # a reaper that dies silently is worse than a log line
                log.exception("idle unload failed; reaper continues")

    def _ensure_reaper(self) -> None:
        """Start the reaper on first load — never in a process that never loads."""
        if settings.idle_unload_seconds <= 0:
            return
        if self._reaper is not None and self._reaper.is_alive():
            return
        self._stop_reaper.clear()
        self._reaper = threading.Thread(
            target=self._reaper_loop, name="moss-idle-reaper", daemon=True
        )
        self._reaper.start()

    def stop_reaper(self) -> None:
        self._stop_reaper.set()
        if self._reaper is not None:
            self._reaper.join(timeout=5)
            self._reaper = None

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

                    processor_kwargs = {"trust_remote_code": True}
                    if model_id == VOICE_GENERATOR_MODEL:
                        # Required by the official VoiceGenerator example.
                        processor_kwargs["normalize_inputs"] = True
                    processor = AutoProcessor.from_pretrained(
                        model_id, **processor_kwargs
                    )
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
                # Start the idle clock at load, not at first use: a model
                # preloaded and then never called is exactly as idle as one
                # that was used once.
                self._touch()
                log.info("model loaded in %.1fs", time.time() - t0)
                self._ensure_reaper()
            finally:
                self._loading_id = None

    def preload_async(self, model_id: str, kind: str = "tts") -> bool:
        """Kick a background load. Returns False if already resident or loading."""
        # _model_id is only set once the load finishes, so checking it alone
        # let every poll of /v1/models/preload during a 60s load spawn one
        # more thread to block on the lock and then discover it had nothing
        # to do.
        if self._model_id == model_id or self._loading_id == model_id:
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
            self._touch()
        if wav.ndim > 1:
            wav = wav.T.squeeze()
        return wav, sr

    def synthesize(
        self,
        text: str,
        reference_wav: Path | None = None,
        model_id: str | None = None,
        language: str | None = None,
    ) -> tuple[np.ndarray, int]:
        """Blocking. Returns (mono float32 waveform, sample_rate).

        `language` names the generation language (upstream takes plain names
        like "Japanese" or "French"; MOSS-TTS-v1.5 supports 31 of them).
        When it is None — including when no default is configured — the
        processor is called exactly as before and the model infers the
        language from the text.
        """
        language = language or settings.default_language
        with self._lock:
            self.ensure_loaded(model_id or DEFAULT_MODEL)
            processor = self._processor
            reference = [str(reference_wav)] if reference_wav else None
            message = _build_user_message(
                processor, text=text, reference=reference, language=language
            )
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
            self._touch()
        wav = decoded.audio_codes_list[0].float().cpu().numpy()
        if wav.ndim > 1:
            wav = wav.squeeze()
        return wav, processor.model_config.sampling_rate

    def design_voice(
        self,
        text: str,
        instruction: str,
        temperature: float = 1.5,
        top_p: float = 0.6,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
    ) -> tuple[np.ndarray, int]:
        """Generate speech from a natural-language voice description."""
        with self._lock:
            self.ensure_loaded(VOICE_GENERATOR_MODEL)
            processor = self._processor
            # Deliberately no `language=` here. This path runs
            # MOSS-VoiceGenerator (1.7B), which supports Chinese and English
            # only and whose build_user_message accepts just `text` and
            # `instruction` — passing a language would raise TypeError. The
            # language passthrough belongs to the 8B MOSS-TTS-v1.5 path in
            # synthesize() above.
            message = processor.build_user_message(
                text=text,
                instruction=instruction,
            )
            batch = processor([[message]], mode="generation")

            t0 = time.time()
            with torch.no_grad():
                outputs = self._model.generate(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                    max_new_tokens=settings.voice_design_max_new_tokens,
                    audio_temperature=temperature,
                    audio_top_p=top_p,
                    audio_top_k=top_k,
                    audio_repetition_penalty=repetition_penalty,
                )
            log.info("voice design generated in %.1fs", time.time() - t0)

            decoded = processor.decode(outputs)[0]
            if decoded is None:
                raise RuntimeError("VoiceGenerator returned no decodable audio")
            if self.device == "mps":
                torch.mps.empty_cache()
            elif self.device == "cuda":
                torch.cuda.empty_cache()
            self._touch()

        wav = decoded.audio_codes_list[0]
        if isinstance(wav, torch.Tensor):
            wav = wav.detach().float().cpu().numpy()
        else:
            wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.reshape(-1)
        return wav.astype(np.float32, copy=False), int(
            getattr(processor.model_config, "sampling_rate", 24000)
        )


def encode_wav(wav: np.ndarray, sample_rate: int, fmt: str) -> bytes:
    """Encode waveform to wav/flac/pcm in-memory. mp3 handled in routes via ffmpeg."""
    if fmt == "pcm":
        return (np.clip(wav, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    sf.write(buf, wav, sample_rate, format=fmt.upper())
    return buf.getvalue()


engine = Engine()
