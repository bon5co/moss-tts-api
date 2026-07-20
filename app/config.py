from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

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

    host: str = "0.0.0.0"
    port: int = 8766


settings = Settings()
