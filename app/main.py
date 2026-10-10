from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from secrets import compare_digest
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles

from app.ai.base import (
    AIProvider,
    AIProviderAuthenticationFailed,
    AIProviderError,
    AIProviderInvalidOutput,
    AIProviderMisconfigured,
    AIProviderRateLimited,
    AIProviderTimeout,
    AIProviderUnavailable,
    AIProviderUpstreamFailure,
)
from app.ai.gemma import GemmaProvider
from app.ai.huggingface import HuggingFaceGemmaProvider
from app.ai.openrouter import OpenRouterGemmaProvider
from app.config import ROOT, settings
from app.models import MissionResponse, MissionSetup, SessionProgress, SessionRecord, SessionSubmission
from app.store import SessionStore, SessionStoreError
from app.models import MissionVoiceRequest
from app.voice.base import VoiceProvider, VoiceProviderError, VoiceProviderMisconfigured
from app.voice.elevenlabs import ElevenLabsVoiceProvider
from app.voice.script import build_coaching_script

app = FastAPI(
    title="JobiGo",
    description="A Gemma-powered outdoor football mission coach for Jobi Anand.",
    version="0.1.0",
    docs_url="/docs" if settings.environment == "development" else None,
    redoc_url="/redoc" if settings.environment == "development" else None,
    openapi_url="/openapi.json" if settings.environment == "development" else None,
)


def configure_cors(application: FastAPI, allowed_origins: tuple[str, ...]) -> None:
    if "*" in allowed_origins:
        raise ValueError("CORS_ALLOWED_ORIGINS must list exact origins; wildcard origins are not allowed.")
    if allowed_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=list(allowed_origins),
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "Authorization"],
            allow_credentials=True,
        )


configure_cors(app, settings.cors_allowed_origins)
app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")

demo_basic = HTTPBasic(auto_error=False)


def require_demo_access() -> None:
    """Public demo: all visitors and Hacktoberfest judges can access the demo without credentials."""
    return None


def get_ai_provider() -> AIProvider:
    if settings.ai_provider == "openrouter":
        return OpenRouterGemmaProvider(settings)
    if settings.ai_provider == "huggingface":
        return HuggingFaceGemmaProvider(settings)
    return GemmaProvider(settings)


def provider_http_exception(exc: AIProviderError) -> HTTPException:
    if isinstance(exc, AIProviderMisconfigured):
        return HTTPException(status_code=503, detail=str(exc))
    if isinstance(exc, AIProviderAuthenticationFailed):
        return HTTPException(status_code=502, detail=str(exc))
    if isinstance(exc, AIProviderRateLimited):
        return HTTPException(status_code=503, detail=str(exc))
    if isinstance(exc, AIProviderTimeout):
        return HTTPException(status_code=504, detail=str(exc))
    if isinstance(exc, AIProviderUnavailable):
        return HTTPException(status_code=503, detail=str(exc))
    if isinstance(exc, (AIProviderUpstreamFailure, AIProviderInvalidOutput)):
        return HTTPException(status_code=502, detail=str(exc))
    return HTTPException(status_code=502, detail="The AI coach could not complete the request.")


def get_store() -> SessionStore:
    return SessionStore(settings.data_file)


def get_voice_provider() -> VoiceProvider:
    return ElevenLabsVoiceProvider(settings)


def recent_history(store: SessionStore) -> list[dict]:
    try:
        return [record.model_dump(mode="json") for record in store.list_sessions(limit=5)]
    except SessionStoreError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(ROOT / "app" / "static" / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness only: confirms the application process can answer HTTP requests."""
    return {"status": "ok", "check": "application_liveness"}


@app.get("/ready")
async def readiness(provider: AIProvider = Depends(get_ai_provider)) -> dict[str, str]:
    """Readiness: confirms the configured Gemma model is available from the selected provider."""
    ready, detail = await provider.check_ready()
    if not ready:
        raise HTTPException(
            status_code=503,
            detail={"status": "not_ready", "check": "gemma_model", "message": detail},
        )
    model = getattr(provider, "model_identifier", settings.gemma_model)
    return {"status": "ready", "check": "gemma_model", "model": model}


@app.get("/api/sessions", response_model=SessionProgress)
def list_sessions(store: SessionStore = Depends(get_store)) -> SessionProgress:
    try:
        return SessionProgress(session_count=store.session_count())
    except SessionStoreError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/missions", response_model=MissionResponse)
async def create_mission(
    setup: MissionSetup,
    provider: AIProvider = Depends(get_ai_provider),
    store: SessionStore = Depends(get_store),
    _access: None = Depends(require_demo_access),
) -> MissionResponse:
    history = recent_history(store)
    try:
        mission = await provider.generate_mission(setup, history)
    except AIProviderError as exc:
        raise provider_http_exception(exc) from exc
    return MissionResponse(mission_id=str(uuid4()), mission=mission, ai_provider=provider.name)


@app.post("/api/missions/{mission_id}/voice")
async def mission_voice(
    mission_id: str,
    request: MissionVoiceRequest,
    provider: VoiceProvider = Depends(get_voice_provider),
    _access: None = Depends(require_demo_access),
) -> Response:
    """Return optional spoken audio for the browser-held mission; do not persist it."""
    if mission_id != request.mission_id:
        raise HTTPException(status_code=400, detail="Mission ID does not match the request.")
    script = build_coaching_script(request.mission)
    try:
        audio = await provider.synthesize(script)
    except VoiceProviderMisconfigured as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except VoiceProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return Response(
        content=audio,
        media_type="audio/mpeg",
        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
    )


@app.post("/api/sessions", response_model=SessionRecord)
async def complete_session(
    submission: SessionSubmission,
    provider: AIProvider = Depends(get_ai_provider),
    store: SessionStore = Depends(get_store),
    _access: None = Depends(require_demo_access),
) -> SessionRecord:
    history = recent_history(store)
    try:
        review = await provider.evaluate_session(submission.mission, submission.report, history)
        next_session = await provider.recommend_next_session(
            review, submission.mission, submission.report, history
        )
    except AIProviderError as exc:
        raise provider_http_exception(exc) from exc

    record = SessionRecord(
        session_id=submission.mission_id,
        mission=submission.mission,
        report=submission.report,
        review=review,
        next_session=next_session,
        completed_at=datetime.now(timezone.utc).isoformat(),
    )
    try:
        store.add_session(record)
    except SessionStoreError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return record
