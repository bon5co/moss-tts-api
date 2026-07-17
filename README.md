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

Drop a short (5–15s) clean reference clip at `voices/<name>.wav`, then request
`"voice": "<name>"`. List available names via `GET /v1/voices`.

More advanced MOSS capabilities (duration control, Pinyin/IPA pronunciation,
code-switching) can be exposed later as non-OpenAI routes under `/v1/moss/*`.

## Architecture

```
app/routes.py   — FastAPI controllers, OpenAI shapes, bearer auth, mp3 encode
app/engine.py   — singleton Engine: lazy load, serialized generate, wav encode
app/config.py   — pydantic-settings over .env
```
