from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    environment: str = os.getenv("APP_ENV", "development").strip().lower()
    cors_allowed_origins: tuple[str, ...] = tuple(
        origin.strip()
        for origin in os.getenv("CORS_ALLOWED_ORIGINS", "").split(",")
        if origin.strip()
    )
    gemma_model: str = os.getenv("GEMMA_MODEL", "gemma3:4b")
    gemma_base_url: str = os.getenv("GEMMA_BASE_URL", "http://localhost:11434").rstrip("/")
    gemma_timeout_seconds: float = float(os.getenv("GEMMA_TIMEOUT_SECONDS", "90"))
    elevenlabs_api_key: str = os.getenv("ELEVENLABS_API_KEY", "")
    elevenlabs_voice_id: str = os.getenv("ELEVENLABS_VOICE_ID", "")
    elevenlabs_model_id: str = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")
    data_file: Path = Path(os.getenv("DATA_FILE", str(ROOT / "data" / "sessions.json")))

    def __post_init__(self) -> None:
        if self.environment not in {"development", "production"}:
            raise ValueError("APP_ENV must be 'development' or 'production'.")

    @property
    def gemma_chat_url(self) -> str:
        return f"{self.gemma_base_url}/api/chat"

    @property
    def gemma_tags_url(self) -> str:
        return f"{self.gemma_base_url}/api/tags"


settings = Settings()
