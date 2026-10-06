from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.ai.base import AIProvider, AIProviderError, AIProviderInvalidOutput, AIProviderUnavailable
from app.ai.gemma import GemmaProvider
from app.config import ROOT, settings
from app.models import MissionResponse, MissionSetup, SessionProgress, SessionRecord, SessionSubmission
from app.store import SessionStore, SessionStoreError

app = FastAPI(
    title="JobiGo",
    description="A Gemma-powered outdoor football mission coach for Jobi Anand.",
    version="0.1.0",
    docs_url="/docs" if settings.environment == "development" else None,
    redoc_url="/redoc" if settings.environment == "development" else None,
    openapi_url="/openapi.json" if settings.environment == "development" else None,
)
if "*" in settings.cors_allowed_origins:
    raise ValueError("CORS_ALLOWED_ORIGINS must list exact origins; wildcard origins are not allowed.")
if settings.cors_allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_allowed_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        allow_credentials=False,
    )
app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")


def get_ai_provider() -> AIProvider:
    return GemmaProvider(settings)


def get_store() -> SessionStore:
    return SessionStore(settings.data_file)


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
    """Readiness: confirms the configured Gemma model can be reached through Ollama."""
    ready, detail = await provider.check_ready()
    if not ready:
        raise HTTPException(
            status_code=503,
            detail={"status": "not_ready", "check": "gemma_model", "message": detail},
        )
    return {"status": "ready", "check": "gemma_model", "model": settings.gemma_model}


@app.get("/api/sessions", response_model=SessionProgress)
def list_sessions(store: SessionStore = Depends(get_store)) -> SessionProgress:
    try:
        return SessionProgress(session_count=len(store.list_sessions(limit=100)))
    except SessionStoreError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/missions", response_model=MissionResponse)
async def create_mission(
    setup: MissionSetup,
    provider: AIProvider = Depends(get_ai_provider),
    store: SessionStore = Depends(get_store),
) -> MissionResponse:
    history = recent_history(store)
    try:
        mission = await provider.generate_mission(setup, history)
    except AIProviderUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except AIProviderInvalidOutput as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except AIProviderError as exc:
        raise HTTPException(status_code=502, detail="The AI coach could not create a mission.") from exc
    return MissionResponse(mission_id=str(uuid4()), mission=mission, ai_provider=provider.name)


@app.post("/api/sessions", response_model=SessionRecord)
async def complete_session(
    submission: SessionSubmission,
    provider: AIProvider = Depends(get_ai_provider),
    store: SessionStore = Depends(get_store),
) -> SessionRecord:
    history = recent_history(store)
    try:
        review = await provider.evaluate_session(submission.mission, submission.report, history)
        next_session = await provider.recommend_next_session(
            review, submission.mission, submission.report, history
        )
    except AIProviderUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except AIProviderInvalidOutput as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except AIProviderError as exc:
        raise HTTPException(status_code=502, detail="The AI coach could not review this session.") from exc

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
