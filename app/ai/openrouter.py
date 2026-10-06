from __future__ import annotations

from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.ai.base import (
    AIProviderAuthenticationFailed,
    AIProviderInvalidOutput,
    AIProviderMisconfigured,
    AIProviderRateLimited,
    AIProviderTimeout,
    AIProviderUnavailable,
    AIProviderUpstreamFailure,
)
from app.ai.gemma import GemmaProvider, parse_structured_output
from app.config import Settings

T = TypeVar("T", bound=BaseModel)
REQUIRED_MODEL = "google/gemma-3-4b-it"


class OpenRouterGemmaProvider(GemmaProvider):
    """OpenRouter transport for the existing Gemma coaching and validation flow."""

    name = f"Gemma 3 4B via OpenRouter ({REQUIRED_MODEL})"

    def __init__(self, config: Settings, transport: httpx.AsyncBaseTransport | None = None):
        if config.openrouter_model != REQUIRED_MODEL:
            raise ValueError(f"OPENROUTER_MODEL must be exactly '{REQUIRED_MODEL}'.")
        self.config = config
        self.transport = transport

    @property
    def model_identifier(self) -> str:
        return REQUIRED_MODEL

    async def check_ready(self) -> tuple[bool, str]:
        if not self.config.openrouter_api_key.strip():
            return False, "OpenRouter is selected but OPENROUTER_API_KEY is not configured."
        try:
            async with httpx.AsyncClient(
                timeout=min(self.config.openrouter_timeout_seconds, 5.0),
                transport=self.transport,
            ) as client:
                response = await client.get(
                    self.config.openrouter_models_url,
                    headers=self._headers(),
                )
                response.raise_for_status()
                payload = response.json()
            models = payload.get("data")
            if not isinstance(models, list):
                return False, "OpenRouter returned an invalid model catalog."
            if not any(isinstance(item, dict) and item.get("id") == REQUIRED_MODEL for item in models):
                return False, f"Required OpenRouter model '{REQUIRED_MODEL}' is not available."
        except (httpx.HTTPError, ValueError, TypeError, AttributeError):
            return False, "OpenRouter could not be reached or did not return a valid model catalog."
        return True, f"Required model '{REQUIRED_MODEL}' is available through OpenRouter."

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.openrouter_api_key}",
            "Content-Type": "application/json",
        }

    async def _complete(self, system: str, prompt: str, schema: type[T]) -> T:
        if not self.config.openrouter_api_key.strip():
            raise AIProviderMisconfigured(
                "OpenRouter is selected but OPENROUTER_API_KEY is not configured on the server."
            )
        body = {
            "model": REQUIRED_MODEL,
            "stream": False,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__.lower(),
                    "strict": True,
                    "schema": schema.model_json_schema(),
                },
            },
            # Ensure routing only considers endpoints advertising the requested schema support.
            "provider": {"require_parameters": True},
        }
        try:
            async with httpx.AsyncClient(
                timeout=self.config.openrouter_timeout_seconds,
                transport=self.transport,
            ) as client:
                response = await client.post(
                    self.config.openrouter_chat_url,
                    headers=self._headers(),
                    json=body,
                )
                if response.status_code in {401, 403}:
                    raise AIProviderAuthenticationFailed(
                        "OpenRouter rejected the server-side API key. Check OPENROUTER_API_KEY."
                    )
                if response.status_code == 429:
                    raise AIProviderRateLimited(
                        "OpenRouter is rate limiting requests. Please wait and try again."
                    )
                if response.status_code >= 500:
                    raise AIProviderUpstreamFailure(
                        "OpenRouter is temporarily unavailable. Please try again."
                    )
                if response.is_error:
                    raise AIProviderUpstreamFailure(
                        "OpenRouter could not process the structured Gemma request."
                    )
                payload = response.json()
        except (AIProviderAuthenticationFailed, AIProviderRateLimited, AIProviderUpstreamFailure):
            raise
        except httpx.TimeoutException as exc:
            raise AIProviderTimeout(
                "OpenRouter did not respond before the configured timeout. Please try again."
            ) from exc
        except httpx.RequestError as exc:
            raise AIProviderUnavailable(
                "Could not reach OpenRouter. Check the server network and try again."
            ) from exc
        except (ValueError, TypeError) as exc:
            raise AIProviderInvalidOutput(
                "OpenRouter returned a malformed response for the structured Gemma request."
            ) from exc

        try:
            choices = payload.get("choices")
            if not isinstance(choices, list) or not choices:
                raise ValueError("missing choices")
            message = choices[0].get("message") if isinstance(choices[0], dict) else None
            raw = message.get("content") if isinstance(message, dict) else None
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError("missing message content")
            return parse_structured_output(raw, schema)
        except (ValueError, ValidationError, TypeError, AttributeError, KeyError) as exc:
            raise AIProviderInvalidOutput(
                "Gemma returned output that did not match the required structured response. Please retry."
            ) from exc
