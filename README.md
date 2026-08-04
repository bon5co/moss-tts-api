# moss-tts-api

[![Deploy on Railway](https://railway.com/button.svg)](https://railway.com/deploy/moss-tts-voice-cloning-api-cpu-openai-co?referralCode=Z1xivh&utm_medium=integration&utm_source=template&utm_campaign=generic)

OpenAI-compatible TTS server backed by a local
[MOSS-TTS-v1.5](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-v1.5) (8B) model.
No cloud — inference runs on the local GPU (CUDA / Apple Silicon MPS / CPU).

The model is loaded **once per process** (lazy singleton) and generation is
serialized on the device. Any OpenAI SDK works by overriding `base_url`.

## Quickstart

```bash
uv sync
cp .env.example .env   # optionally set API_KEY
uv run python -m app.main
```

First request downloads the model from HuggingFace (~16GB, cached at
`~/.cache/huggingface/hub/`). On an M1 Pro (32GB), loading takes ~25s and
synthesis runs ~7 tokens/s ≈ 7s wall per 1s of audio.

`mp3` output requires `ffmpeg` on PATH (`brew install ffmpeg`).

## Endpoints

| Method | Path | Notes |
|--------|------|-------|
| `GET`  | `/` | Plain-text usage guide aimed at LLM agents |
| `POST` | `/v1/audio/speech` | OpenAI createSpeech-compatible |
| `POST` | `/v1/audio/clone` | Extension: one-shot clone (multipart ref clip + text) |
| `POST` | `/v1/audio/voice-design` | Extension: instruction-driven voice, no reference audio |
| `GET`  | `/v1/models` | Registry with `default`/`loaded`/`loading` flags |
| `POST` | `/v1/models/preload` | Order a model into memory ahead of first use |
| `GET`  | `/v1/voices` | Extension: reference clips available for cloning |
| `PUT`  | `/v1/voices/{name}` | Extension: register/replace a named voice (multipart clip; `?overwrite=false` to refuse replacing) |
| `POST` | `/v1/audio/sound-effect` | Extension: text-to-sound-effect (48kHz) |
| `GET`  | `/health` | Liveness, device, resident model, RSS + idle clock |

### OpenAI compatibility notes

- `model`: only MOSS models are served — omit/empty = the server's
  `MODEL_ID` default; anything non-MOSS (e.g. `tts-1`) is rejected with 422
  listing the available models:

  | Short name | HF id | Params | RAM (bf16) |
  |---|---|---|---|
  | `moss-tts-local` | `OpenMOSS-Team/MOSS-TTS-Local-Transformer` | 1.7B | ~3.4GB |
  | `moss-tts-local-v1.5` | `OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5` | 4B, 48kHz stereo | ~8GB |
  | `moss-tts` | `OpenMOSS-Team/MOSS-TTS` | 8B (v1.0) | ~16GB |
  | `moss-tts-v1.5` | `OpenMOSS-Team/MOSS-TTS-v1.5` | 8B | ~16GB |

  (plus a shared MOSS audio tokenizer, ~7GB download, loaded alongside every
  variant). One model resident at a time — requesting a non-resident model
  loads it on the spot (lazy). Sound effects and instruction-driven voice
  design are served too (below); the rest of the MOSS family (TTSD dialogue,
  Realtime, Nano —
  see [OpenMOSS/MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS)) uses
  different task interfaces and is not served here.
- `voice` maps to a reference clip at `voices/<voice>.wav` for zero-shot
  cloning. `default` (or any unknown name) uses the model's default voice.
- `response_format`: `wav`, `mp3`, `flac`, `pcm` (24kHz mono s16le).
- `speed` ≠ 1.0 is rejected with 400 — not supported by the backend.
- `language` (extension, optional) names the generation language on
  `/v1/audio/speech` and `/v1/audio/clone` — see below.

### Language

`/v1/audio/speech` and `/v1/audio/clone` accept an optional `language` field
that is passed straight through to the model
(`processor.build_user_message(..., language=...)`). Omit it and the model
infers the language from the text, exactly as before this field existed.

```bash
curl -X POST http://localhost:8766/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "むかしむかし、ある所に。", "language": "Japanese"}' \
  -o jp.wav

curl -X POST http://localhost:8766/v1/audio/clone \
  -F input="むかしむかし、ある所に。" -F language=Japanese -F file=@ref.wav -o jp.wav
```

Values are plain language names, taken verbatim from upstream — no enum is
enforced server-side, so anything MOSS accepts works. MOSS-TTS-v1.5 supports
31 languages: Chinese, English, Japanese, Korean, French, German, Spanish,
Portuguese, Italian, Russian, Arabic, Hindi, Indonesian, Vietnamese, Thai,
Turkish, Dutch, Polish, Swedish, Danish, Norwegian, Finnish, Czech, Greek,
Hungarian, Romanian, Ukrainian, Hebrew, Malay, Filipino and Persian. The
authoritative list lives with the model — see
[OpenMOSS/MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS) and the
[MOSS-TTS-v1.5 model card](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-v1.5).

Set `DEFAULT_LANGUAGE` (or `MOSS_DEFAULT_LANGUAGE`) to apply a language to
every request that omits the field. Unset by default, so an unconfigured
server behaves identically to before.

## Idle unloading

One model is resident at a time, and it is dropped after
`IDLE_UNLOAD_SECONDS` (default `900`) without a generate. The 8B flagship is
~16GB of weights; on MPS that is unified memory taken from the whole
machine, so a server that answers one call in the morning should not still
be holding it that evening. The next request reloads from the local HF
cache — disk-speed, not download-speed.

```
IDLE_UNLOAD_SECONDS=900   # default; 0 keeps the model loaded forever
```

Set `0` where the model is the only tenant of the box and reload latency
matters more than idle footprint. `/health` reports the state:

```json
{"loaded_model": "OpenMOSS-Team/MOSS-TTS-v1.5", "rss_mb": 16412.5,
 "idle_seconds": 42.1, "idle_unload_seconds": 900}
```

`loaded_model: null` on an idle server is the reaper having done its job,
not a crash. `rss_mb` is there so "is it leaking?" can be answered by
polling the endpoint instead of reading the source.

If the loaded processor revision does not accept the `language` keyword, the
server logs a warning and synthesizes without it rather than failing the
request.

**`/v1/audio/voice-design` does not support `language`.** That endpoint runs
MOSS-VoiceGenerator (1.7B), a different model that supports **Chinese and
English only** and whose `build_user_message` takes just `text` and
`instruction`. Sending `language` there is ignored, and the server
deliberately never forwards it — doing so would raise a `TypeError`. Steer
that endpoint's delivery through `instruction` instead.

### curl

```bash
curl -X POST http://localhost:8766/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello from MOSS.", "response_format": "wav"}' \
  -o hello.wav
```

### OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8766/v1", api_key="unused-or-your-API_KEY")
audio = client.audio.speech.create(model="moss", voice="default", input="Hello.")
audio.write_to_file("hello.wav")
```

## Voice cloning

Register a short (5–15s) clean reference clip under a name, then request
`"voice": "<name>"`:

```bash
curl -X PUT http://localhost:8766/v1/voices/alice -F file=@ref.wav
```

Accepts wav/mp3/flac (transcoded to wav server-side, 1–60s enforced). Stored
at `voices/<name>.wav` — dropping a file there by hand works too. List
available names via `GET /v1/voices`. For a one-shot clone without storing
anything, use `POST /v1/audio/clone` (multipart clip + text).

Uploading a name that already exists **replaces** that clip. There is no
versioning or backup — the previous audio is gone. The response says which
happened:

```json
{"voice": "alice", "seconds": 8.2, "replaced": true}
```

Pass `?overwrite=false` to get `409` instead of replacing:

```bash
curl -X PUT "http://localhost:8766/v1/voices/alice?overwrite=false" -F file=@ref.wav
```

An upload that fails validation (undecodable audio, out-of-range duration)
never touches the existing clip — the new clip is decoded and checked in a
temp file, and only a valid one is moved into place, atomically. Concurrent
synthesis therefore reads either the old clip or the new one, never a
half-written file.

## Voice design

Describe the desired voice and delivery without supplying a reference clip:

```bash
curl -X POST http://localhost:8766/v1/audio/voice-design \
  -H "Content-Type: application/json" \
  -d '{
    "input": "บางครั้งความจริงอาจถูกตัดสินจากเพียงสิ่งที่ตาเห็น",
    "instruction": "Warm, composed Thai female narrator, age 35, clear articulation, emotionally restrained moral-drama delivery.",
    "response_format": "wav"
  }' \
  -o voice-designed.wav
```

The first request downloads and loads MOSS-VoiceGenerator. Switching back
to `/v1/audio/speech` reloads the configured TTS model.

## Deploy on Railway

The included `Dockerfile` + `railway.json` deploy CPU-only. Attach a volume
at **`/data`** — it persists uploaded voices. Model weights (~13GB for the
default 1.7B stack) live on ephemeral disk and re-download on each cold
deploy: Railway Hobby caps volumes at 5GB so the cache can't persist there.
On Pro, grow the volume and set `HF_HOME=/data/hf` to keep weights across
deploys. Set `API_KEY` — the server is public on Railway, don't run it open.

On CPU plans use the smallest model. Recommended service variables:

```
MODEL_ID=OpenMOSS-Team/MOSS-TTS-Local-Transformer   # 1.7B — fits 8GB RAM
DTYPE=bfloat16                                      # ~3.4GB resident vs ~7GB float32
MAX_NEW_TOKENS=512                                  # ~40s audio cap; CPU is slow
IDLE_UNLOAD_SECONDS=900                             # free the weights when idle
API_KEY=<generate one>
```

and pass `"model": "moss-tts-local"` in requests (or rely on the default —
unknown names fall back to `MODEL_ID`). The 8B/4B variants need more RAM
(16GB+ in bfloat16) and are painfully slow without a GPU. First request
after deploy downloads model weights (~3.5GB for 1.7B) into the volume;
watch progress in the deploy logs or preload via `POST /v1/models/preload`.

## Sound effects

`POST /v1/audio/sound-effect` serves
[MOSS-SoundEffect-v2.0](https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect-v2.0)
(1.3B DiT + flow matching, 48kHz, 1–30s per call):

```bash
curl -X POST http://localhost:8766/v1/audio/sound-effect \
  -H "Content-Type: application/json" \
  -d '{"input": "Rain on a tin roof with distant thunder.", "seconds": 8}' \
  -o rain.wav
```

Params: `seconds` (1–30), `num_inference_steps` (10–200, default 100),
`cfg_scale` (1–10, default 4.0), `response_format` as for speech. The
single-resident rule applies — a sound-effect call swaps out the TTS model
and vice versa. Dependency note: `moss_soundeffect_v2` pins exact dep
versions; `[tool.uv] override-dependencies` in `pyproject.toml` forces the
transformers 5.x the TTS path needs (verified working).

More advanced MOSS capabilities (duration control, Pinyin/IPA pronunciation,
code-switching) can be exposed later as non-OpenAI routes under `/v1/moss/*`.

## Tests

```bash
uv sync --group dev
uv run pytest
```

The suite mocks the processor — no model weights are downloaded and no
generation runs.

## Architecture

```
app/routes.py   — FastAPI controllers, OpenAI shapes, bearer auth, mp3 encode
app/engine.py   — singleton Engine: lazy load, serialized generate, wav encode
app/config.py   — pydantic-settings over .env
tests/          — pytest, processor mocked (no weights needed)
```
