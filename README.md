# moss-tts-api

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
| `GET`  | `/v1/models` | Registry with `default`/`loaded`/`loading` flags |
| `POST` | `/v1/models/preload` | Order a model into memory ahead of first use |
| `GET`  | `/v1/voices` | Extension: reference clips available for cloning |
| `PUT`  | `/v1/voices/{name}` | Extension: register/replace a named voice (multipart clip) |
| `GET`  | `/health` | Liveness, device, resident model |

### OpenAI compatibility notes

- `model`: short names `moss-tts-v1.5` (8B, default), `moss-tts-local-v1.5`
  (4B), `moss-tts-local` (1.7B) or full HF ids. One model resident at a
  time — requesting a non-resident model loads it on the spot (lazy), and
  unknown names (e.g. `tts-1`) fall back to the default largest model.
- `voice` maps to a reference clip at `voices/<voice>.wav` for zero-shot
  cloning. `default` (or any unknown name) uses the model's default voice.
- `response_format`: `wav`, `mp3`, `flac`, `pcm` (24kHz mono s16le).
- `speed` ≠ 1.0 is rejected with 400 — not supported by the backend.

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
API_KEY=<generate one>
```

and pass `"model": "moss-tts-local"` in requests (or rely on the default —
unknown names fall back to `MODEL_ID`). The 8B/4B variants need more RAM
(16GB+ in bfloat16) and are painfully slow without a GPU. First request
after deploy downloads model weights (~3.5GB for 1.7B) into the volume;
watch progress in the deploy logs or preload via `POST /v1/models/preload`.

More advanced MOSS capabilities (duration control, Pinyin/IPA pronunciation,
code-switching) can be exposed later as non-OpenAI routes under `/v1/moss/*`.

## Architecture

```
app/routes.py   — FastAPI controllers, OpenAI shapes, bearer auth, mp3 encode
app/engine.py   — singleton Engine: lazy load, serialized generate, wav encode
app/config.py   — pydantic-settings over .env
```
