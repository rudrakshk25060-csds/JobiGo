from __future__ import annotations

from urllib.parse import quote

import httpx

from app.config import Settings
from app.voice.base import VoiceProviderMisconfigured, VoiceProviderUnavailable


class ElevenLabsVoiceProvider:
    """Optional server-side ElevenLabs streaming text-to-speech adapter."""

    def __init__(self, config: Settings, transport: httpx.AsyncBaseTransport | None = None):
        self.config = config
        self.transport = transport

    async def synthesize(self, script: str) -> bytes:
        api_key = self.config.elevenlabs_api_key.strip()
        voice_id = self.config.elevenlabs_voice_id.strip()
        if not api_key or not voice_id:
            raise VoiceProviderMisconfigured(
                "Coach audio is not configured. Your written mission is still ready."
            )
        if not script or len(script) > 1500:
            raise VoiceProviderUnavailable("The coach script was not suitable for audio generation.")

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{quote(voice_id, safe='')}/stream"
        try:
            async with httpx.AsyncClient(timeout=35.0, transport=self.transport) as client:
                response = await client.post(
                    url,
                    params={"output_format": "mp3_44100_128"},
                    headers={
                        "xi-api-key": api_key,
                        "Accept": "audio/mpeg",
                        "Content-Type": "application/json",
                    },
                    json={"text": script, "model_id": self.config.elevenlabs_model_id},
                )
        except httpx.RequestError as exc:
            raise VoiceProviderUnavailable(
                "Coach audio is temporarily unavailable. Your written mission is still ready."
            ) from exc
        if response.status_code < 200 or response.status_code >= 300:
            raise VoiceProviderUnavailable(
                "Coach audio could not be generated. Your written mission is still ready."
            )
        if not response.content or len(response.content) > 8_000_000:
            raise VoiceProviderUnavailable(
                "Coach audio was empty or too large. Your written mission is still ready."
            )
        if not response.headers.get("content-type", "").lower().startswith("audio/"):
            raise VoiceProviderUnavailable(
                "Coach audio was not returned. Your written mission is still ready."
            )
        return response.content
