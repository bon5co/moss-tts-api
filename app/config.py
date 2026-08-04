from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", populate_by_name=True
    )

    # Optional bearer token. Empty string = auth disabled (local use).
    api_key: str = ""

    model_id: str = "OpenMOSS-Team/MOSS-TTS-v1.5"
    device: str = "auto"  # auto | cuda | mps | cpu
    dtype: str = "auto"  # auto | bfloat16 | float16 | float32
    attn_implementation: str = "sdpa"
    max_new_tokens: int = 2048  # ~163s of audio at 12.5 tok/s
    voice_design_max_new_tokens: int = 4096

    # Directory of reference clips for voice cloning; `voice` request param
    # resolves to <voices_dir>/<voice>.wav.
    voices_dir: str = "voices"

    # Drop the resident model after this many seconds without a generate.
    # The 8B flagship is ~16GB of weights, and on MPS that is unified memory
    # taken from the whole machine, not just this process — a server that
    # answers one /speak call at 09:00 should not still be holding it at
    # 17:00. The next request pays a reload (weights come from the local HF
    # cache, so it is disk-speed, not download-speed).
    # 0 disables the reaper entirely: the historical behaviour, for hosts
    # where the model is the only tenant and reload latency matters more
    # than idle footprint.
    idle_unload_seconds: int = Field(
        default=900,
        ge=0,
        validation_alias=AliasChoices(
            "MOSS_IDLE_UNLOAD_SECONDS", "IDLE_UNLOAD_SECONDS", "idle_unload_seconds"
        ),
    )

    # Optional generation language for the TTS path (/v1/audio/speech and
    # /v1/audio/clone), applied when a request omits `language`. Upstream
    # MOSS-TTS-v1.5 accepts plain language names ("Japanese", "French", ...)
    # across the 31 languages it supports. None means the argument is not
    # passed at all, letting the model infer the language from the text —
    # the historical behaviour, kept as the default so nothing changes
    # unless this is configured.
    # Not used by /v1/audio/voice-design: MOSS-VoiceGenerator (1.7B) is
    # Chinese/English only and its build_user_message takes no language.
    default_language: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "MOSS_DEFAULT_LANGUAGE", "DEFAULT_LANGUAGE", "default_language"
        ),
    )

    host: str = "0.0.0.0"
    port: int = 8766


settings = Settings()
