from __future__ import annotations

from typing import Protocol


class VoiceProviderError(Exception):
    """Voice audio could not be generated."""


class VoiceProviderMisconfigured(VoiceProviderError):
    """Optional voice service credentials or settings are missing."""


class VoiceProviderUnavailable(VoiceProviderError):
    """The voice service could not be reached or did not return audio."""


class VoiceProvider(Protocol):
    async def synthesize(self, script: str) -> bytes: ...
