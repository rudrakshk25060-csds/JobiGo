
from __future__ import annotations

import json
import logging
import re
from typing import Any, TypeVar

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
REQUIRED_MODEL = "google/gemma-3-4b-it:free"
logger = logging.getLogger(__name__)
_OPENROUTER_UNSUPPORTED_KEYS = {
    "title",
    "default",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
    "pattern",
    "format",
    "examples",
}


def _openrouter_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Convert Pydantic's schema into a provider-friendly strict JSON Schema."""
    raw = model.model_json_schema()
    definitions = raw.get("$defs", {})

    def clean(node: Any, resolving: tuple[str, ...] = ()) -> Any:
        if isinstance(node, list):
            return [clean(item, resolving) for item in node]

        if not isinstance(node, dict):
            return node

        if "$ref" in node:
            ref_name = node["$ref"].rsplit("/", 1)[-1]
            if ref_name in resolving:
                raise ValueError(f"Circular schema reference: {ref_name}")
            if ref_name not in definitions:
                raise ValueError(f"Unknown schema reference: {ref_name}")
            return clean(definitions[ref_name], resolving + (ref_name,))

        result = {
            key: clean(value, resolving)
            for key, value in node.items()
            if key not in _OPENROUTER_UNSUPPORTED_KEYS
            and key not in {"$defs", "$schema"}
        }

        if result.get("type") == "object":
            properties = result.get("properties")
            if isinstance(properties, dict):
                result["required"] = list(properties.keys())
            result["additionalProperties"] = False

        return result

    return clean(raw)


def _safe_upstream_error(response: httpx.Response, config: Settings) -> str:
    """Extract only bounded provider diagnostics; never log a response body or request data."""
    try:
        payload = response.json()
    except (ValueError, TypeError):
        payload = {}

    error = payload.get("error") if isinstance(payload, dict) else None
    metadata = error.get("metadata") if isinstance(error, dict) else None
    code = error.get("code") if isinstance(error, dict) else None
    message = error.get("message") if isinstance(error, dict) else None
    provider = metadata.get("provider_name") if isinstance(metadata, dict) else None
    provider_code = None

    # Some OpenRouter errors wrap the provider's useful message in metadata.raw.
    raw = metadata.get("raw") if isinstance(metadata, dict) else None
    if isinstance(raw, str):
        try:
            raw_payload = json.loads(raw)
        except (ValueError, TypeError):
            raw_payload = None
        raw_error = raw_payload.get("error") if isinstance(raw_payload, dict) else None
        if isinstance(raw_error, dict) and isinstance(raw_error.get("message"), str):
            message = raw_error["message"]
        if isinstance(raw_error, dict):
            provider_code = raw_error.get("code")

    def safe(value: Any, limit: int = 400) -> str:
        if not isinstance(value, (str, int, float)):
            return "unknown"
        result = str(value)
        for secret in (
            config.openrouter_api_key,
            config.demo_access_token,
            config.elevenlabs_api_key,
        ):
            if secret:
                result = result.replace(secret, "[redacted]")
        result = re.sub(r"(?i)\bBearer\s+[^\s,;]+", "Bearer [redacted]", result)
        result = re.sub(r"\bsk-or-[A-Za-z0-9_-]+", "[redacted-key]", result)
        result = re.sub(r"(?i)\b(api[_-]?key|token)\s*[:=]\s*[^\s,;]+", r"\1=[redacted]", result)
        result = " ".join(result.split())
        return result[:limit]

    return (
        f"status={response.status_code} code={safe(code, 80)} "
        f"provider={safe(provider, 100)} provider_code={safe(provider_code, 80)} "
        f"message={safe(message)}"
    )


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
                    self.config.openrouter_model_endpoints_url,
                    headers=self._headers(),
                )
                response.raise_for_status()
                payload = response.json()
            model = payload.get("data")
            if not isinstance(model, dict) or model.get("id") != REQUIRED_MODEL:
                return False, "OpenRouter returned an invalid endpoint catalog for the required model."
            endpoints = model.get("endpoints")
            if not isinstance(endpoints, list):
                return False, "OpenRouter returned an invalid endpoint catalog for the required model."
            if not any(
                isinstance(endpoint, dict)
                and "structured_outputs" in endpoint.get("supported_parameters", [])
                for endpoint in endpoints
            ):
                return False, (
                    f"No available endpoint for '{REQUIRED_MODEL}' currently advertises structured outputs."
                )
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
                    "schema": _openrouter_schema(schema),
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
                if response.status_code == 402:
                    logger.warning(
                        "OpenRouter rejected request for account billing: %s",
                        _safe_upstream_error(response, self.config),
                    )
                    raise AIProviderUpstreamFailure(
                        "OpenRouter reports insufficient account credits or spending allowance. "
                        "Add credits or configure a key from a funded account."
                    )
                if response.is_error:
                    logger.warning(
                        "OpenRouter rejected structured Gemma request: %s",
                        _safe_upstream_error(response, self.config),
                    )
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
