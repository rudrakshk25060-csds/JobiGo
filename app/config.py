from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    ai_provider: str = os.getenv("AI_PROVIDER", "ollama").strip().lower()
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_model: str = os.getenv("OPENROUTER_MODEL", "google/gemma-3-4b-it:free").strip()
    openrouter_base_url: str = os.getenv(
        "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
    ).rstrip("/")
    openrouter_timeout_seconds: float = float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "90"))
    environment: str = os.getenv("APP_ENV", "development").strip().lower()
    cors_allowed_origins: tuple[str, ...] = tuple(
        origin.strip()
        for origin in os.getenv("CORS_ALLOWED_ORIGINS", "").split(",")
        if origin.strip()
    )
    gemma_model: str = os.getenv("OLLAMA_MODEL", os.getenv("GEMMA_MODEL", "gemma3:4b"))
    gemma_base_url: str = os.getenv(
        "OLLAMA_BASE_URL", os.getenv("GEMMA_BASE_URL", "http://localhost:11434")
    ).rstrip("/")
    gemma_timeout_seconds: float = float(os.getenv("GEMMA_TIMEOUT_SECONDS", "90"))
    elevenlabs_api_key: str = os.getenv("ELEVENLABS_API_KEY", "")
    elevenlabs_voice_id: str = os.getenv("ELEVENLABS_VOICE_ID", "")
    elevenlabs_model_id: str = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")
    data_file: Path = Path(os.getenv("DATA_FILE", str(ROOT / "data" / "sessions.sqlite3")))
    demo_access_username: str = os.getenv("DEMO_ACCESS_USERNAME", "jobigo")
    demo_access_token: str = os.getenv("DEMO_ACCESS_TOKEN", "")

    def __post_init__(self) -> None:
        if self.environment not in {"development", "production"}:
            raise ValueError("APP_ENV must be 'development' or 'production'.")
        if self.ai_provider not in {"ollama", "openrouter"}:
            raise ValueError("AI_PROVIDER must be 'ollama' or 'openrouter'.")
        if self.ai_provider == "openrouter" and self.openrouter_model != "google/gemma-3-4b-it:free":
            raise ValueError("OPENROUTER_MODEL must be exactly 'google/gemma-3-4b-it:free'.")
        if self.openrouter_timeout_seconds <= 0:
            raise ValueError("OPENROUTER_TIMEOUT_SECONDS must be positive.")
        if self.environment == "production" and not self.demo_access_token.strip():
            raise ValueError("DEMO_ACCESS_TOKEN is required when APP_ENV=production.")

    @property
    def gemma_chat_url(self) -> str:
        return f"{self.gemma_base_url}/api/chat"

    @property
    def gemma_tags_url(self) -> str:
        return f"{self.gemma_base_url}/api/tags"

    @property
    def openrouter_chat_url(self) -> str:
        return f"{self.openrouter_base_url}/chat/completions"

    @property
    def openrouter_models_url(self) -> str:
        return f"{self.openrouter_base_url}/models"

    @property
    def openrouter_model_endpoints_url(self) -> str:
        return f"{self.openrouter_base_url}/models/{self.openrouter_model}/endpoints"


settings = Settings()
