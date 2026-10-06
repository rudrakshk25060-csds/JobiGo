from __future__ import annotations

from typing import Protocol

from app.models import (
    Mission,
    MissionSetup,
    NextSessionRecommendation,
    SessionReport,
    SessionReview,
)


class AIProviderError(Exception):
    """The configured AI service could not produce a usable response."""


class AIProviderUnavailable(AIProviderError):
    """The configured model endpoint could not be reached."""


class AIProviderInvalidOutput(AIProviderError):
    """The model returned output that failed schema validation."""


class AIProviderMisconfigured(AIProviderError):
    """The selected AI provider is missing required server configuration."""


class AIProviderAuthenticationFailed(AIProviderError):
    """The upstream AI provider rejected its server-side credentials."""


class AIProviderRateLimited(AIProviderError):
    """The upstream AI provider is temporarily rate limiting requests."""


class AIProviderUpstreamFailure(AIProviderError):
    """The upstream AI provider returned a non-retryable service error."""


class AIProviderTimeout(AIProviderUnavailable):
    """The upstream AI provider did not respond before the configured timeout."""


class AIProvider(Protocol):
    name: str

    async def check_ready(self) -> tuple[bool, str]: ...

    async def generate_mission(
        self, setup: MissionSetup, history: list[dict]
    ) -> Mission: ...

    async def evaluate_session(
        self, mission: Mission, report: SessionReport, history: list[dict]
    ) -> SessionReview: ...

    async def recommend_next_session(
        self,
        review: SessionReview,
        mission: Mission,
        report: SessionReport,
        history: list[dict],
    ) -> NextSessionRecommendation: ...
