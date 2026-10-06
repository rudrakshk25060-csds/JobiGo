from __future__ import annotations

import logging
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
from app.ai.openrouter import REQUIRED_MODEL, _openrouter_schema
from app.config import Settings

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)
_HUB_MODEL_URL = "https://huggingface.co/api/models/google/gemma-3-4b-it"


class HuggingFaceGemmaProvider(GemmaProvider):
    """Hugging Face Inference Providers transport for JobiGo's shared Gemma flow."""

    name = f"Gemma 3 4B via Hugging Face ({REQUIRED_MODEL})"

    def __init__(self, config: Settings, transport: httpx.AsyncBaseTransport | None = None):
        if config.hf_model != REQUIRED_MODEL:
            raise ValueError(f"HF_MODEL must be exactly '{REQUIRED_MODEL}'.")
        self.config = config
        self.transport = transport

    @property
    def model_identifier(self) -> str:
        return REQUIRED_MODEL

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.hf_token}",
            "Content-Type": "application/json",
        }

    async def check_ready(self) -> tuple[bool, str]:
        if not self.config.hf_token.strip():
            return False, "Hugging Face is selected but HF_TOKEN is not configured."
        try:
            async with httpx.AsyncClient(timeout=5.0, transport=self.transport) as client:
                response = await client.get(
                    _HUB_MODEL_URL,
                    params={"expand[]": "inferenceProviderMapping"},
                    headers={"Authorization": f"Bearer {self.config.hf_token}"},
                )
                response.raise_for_status()
                payload = response.json()
            mapping = payload.get("inferenceProviderMapping")
            if payload.get("id") != REQUIRED_MODEL or not isinstance(mapping, dict):
                return False, "Hugging Face returned an invalid Gemma provider catalog."
            if not any(
                isinstance(item, dict) and item.get("status") == "live"
                for item in mapping.values()
            ):
                return False, f"No live Hugging Face inference provider is listed for '{REQUIRED_MODEL}'."
        except (httpx.HTTPError, ValueError, TypeError, AttributeError):
            return False, "Hugging Face could not be reached or did not return a valid model catalog."
        return True, f"A live Hugging Face inference provider is listed for '{REQUIRED_MODEL}'."

    async def _complete(self, system: str, prompt: str, schema: type[T]) -> T:
        if not self.config.hf_token.strip():
            raise AIProviderMisconfigured(
                "Hugging Face is selected but HF_TOKEN is not configured on the server."
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
                    "schema": _openrouter_schema(schema),
                },
            },
        }
        try:
            async with httpx.AsyncClient(
                timeout=self.config.hf_timeout_seconds, transport=self.transport
            ) as client:
                response = await client.post(
                    self.config.hf_chat_url,
                    headers=self._headers(),
                    json=body,
                )
                if response.status_code in {401, 403}:
                    raise AIProviderAuthenticationFailed(
                        "Hugging Face rejected the server-side token. Check HF_TOKEN permissions."
                    )
                if response.status_code == 429:
                    raise AIProviderRateLimited(
                        "Hugging Face is rate limiting requests. Please wait and try again."
                    )
                if response.status_code >= 500:
                    raise AIProviderUpstreamFailure(
                        "Hugging Face Inference Providers are temporarily unavailable."
                    )
                if response.is_error:
                    # Do not log the upstream body; it may echo request or credential data.
                    logger.warning("Hugging Face rejected a structured Gemma request (HTTP %s).", response.status_code)
                    raise AIProviderUpstreamFailure(
                        "Hugging Face could not process the structured Gemma request."
                    )
                payload = response.json()
        except (AIProviderAuthenticationFailed, AIProviderRateLimited, AIProviderUpstreamFailure):
            raise
        except httpx.TimeoutException as exc:
            raise AIProviderTimeout(
                "Hugging Face did not respond before the configured timeout. Please try again."
            ) from exc
        except httpx.RequestError as exc:
            raise AIProviderUnavailable(
                "Could not reach Hugging Face Inference Providers. Check the server network."
            ) from exc
        except (ValueError, TypeError) as exc:
            raise AIProviderInvalidOutput(
                "Hugging Face returned a malformed structured Gemma response."
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
