from __future__ import annotations

import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.main as main
from app.config import settings
from app.main import configure_cors, get_store
from app.models import (
    MissionSetup,
    NextSessionRecommendation,
    SessionRecord,
    SessionReport,
    SessionReview,
)
from app.store import SessionStore
from test_app import MockAIProvider, build_mission


class DeploymentPreparationTests(unittest.TestCase):
    def record(self, session_id: str = "session-12345678") -> SessionRecord:
        setup = MissionSetup(
            available_time=45, level="intermediate", goal="finishing", equipment="football_only"
        )
        mission = build_mission(setup)
        return SessionRecord(
            session_id=session_id,
            mission=mission,
            report=SessionReport(
                attempts=20, successful_attempts=13, goals=3, completed_drills=3,
                difficulty="right", notes="Weak foot felt harder.",
            ),
            review=SessionReview(
                performance_summary="You completed 13 attempts with focus.",
                strongest_area="Consistency", weakness="Finishing after movement",
                actionable_recommendation="Add a calm touch before each shot.",
            ),
            next_session=NextSessionRecommendation(
                goal="finishing", available_time=45, level="intermediate",
                equipment="football_only", setting="ground",
                reason="Build steadily on the controlled attempts.",
            ),
            completed_at="2026-10-06T00:00:00+00:00",
        )

    def test_data_file_is_configurable_and_reopens_with_persisted_records(self):
        with tempfile.TemporaryDirectory() as folder:
            database = Path(folder) / "configured-history.sqlite3"
            config = replace(settings, data_file=database)
            self.assertEqual(config.data_file, database)
            SessionStore(config.data_file).add_session(self.record())
            reopened = SessionStore(config.data_file)
            self.assertEqual(len(reopened.list_sessions()), 1)
            self.assertEqual(reopened.session_count(), 1)

    def test_legacy_json_history_is_imported_once_and_left_as_backup(self):
        with tempfile.TemporaryDirectory() as folder:
            legacy = Path(folder) / "sessions.json"
            record = self.record()
            legacy.write_text(json.dumps([record.model_dump(mode="json")]), encoding="utf-8")
            store = SessionStore(legacy)
            self.assertEqual(store.session_count(), 1)
            self.assertEqual(store.list_sessions()[0].session_id, record.session_id)
            self.assertTrue(legacy.exists())
            self.assertTrue(legacy.with_suffix(".sqlite3").exists())

    def test_concurrent_session_writes_are_serialized_by_sqlite(self):
        with tempfile.TemporaryDirectory() as folder:
            database = Path(folder) / "sessions.sqlite3"
            SessionStore(database).session_count()
            records = [self.record(f"session-{index:08d}") for index in range(20)]
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(lambda item: SessionStore(database).add_session(item), records))
            self.assertEqual(SessionStore(database).session_count(), 20)

    def test_production_allows_public_access_without_demo_token(self):
        production = replace(settings, environment="production", demo_access_token="")
        self.assertEqual(production.environment, "production")
        self.assertEqual(production.demo_access_token, "")
        configured = replace(settings, environment="production", demo_access_token="long-demo-secret")
        self.assertEqual(configured.demo_access_token, "long-demo-secret")

    def test_public_demo_allows_session_write_without_credentials_in_production(self):
        provider = MockAIProvider()
        with tempfile.TemporaryDirectory() as folder:
            store = SessionStore(Path(folder) / "sessions.sqlite3")
            main.app.dependency_overrides[get_store] = lambda: store
            main.app.dependency_overrides[main.get_ai_provider] = lambda: provider
            setup = MissionSetup(
                available_time=45, level="intermediate", goal="finishing", equipment="football_only"
            )
            mission = build_mission(setup)
            body = {
                "mission_id": "public-session-123456",
                "mission": mission.model_dump(mode="json"),
                "report": self.record().report.model_dump(mode="json"),
            }
            production = replace(settings, environment="production", demo_access_token="")
            try:
                with patch.object(main, "settings", production), TestClient(main.app) as client:
                    response = client.post("/api/sessions", json=body)
                    self.assertEqual(response.status_code, 200)
                    self.assertNotIn("WWW-Authenticate", response.headers)
                    self.assertEqual(client.get("/api/sessions").json(), {"session_count": 1})
                    self.assertNotIn("Weak foot felt harder", client.get("/api/sessions").text)
                    self.assertNotIn("public-session-123456", client.get("/api/sessions").text)
            finally:
                main.app.dependency_overrides.clear()

    def test_all_demo_routes_accessible_without_credentials_in_production(self):
        provider = MockAIProvider()
        with tempfile.TemporaryDirectory() as folder:
            main.app.dependency_overrides[get_store] = lambda: SessionStore(Path(folder) / "sessions.sqlite3")
            main.app.dependency_overrides[main.get_ai_provider] = lambda: provider
            production = replace(settings, environment="production", demo_access_token="demo-secret")
            setup = MissionSetup(
                available_time=45, level="intermediate", goal="finishing", equipment="football_only"
            )
            mission = build_mission(setup)
            try:
                with patch.object(main, "settings", production), TestClient(main.app) as client:
                    # Landing page opens without credentials
                    home_response = client.get("/")
                    self.assertEqual(home_response.status_code, 200)
                    self.assertNotIn("WWW-Authenticate", home_response.headers)

                    # Static assets are accessible
                    css_response = client.get("/static/style.css")
                    self.assertEqual(css_response.status_code, 200)
                    js_response = client.get("/static/app.js")
                    self.assertEqual(js_response.status_code, 200)

                    # Mission creation works without credentials (no 401 popup)
                    mission_response = client.post("/api/missions", json=setup.model_dump(mode="json"))
                    self.assertEqual(mission_response.status_code, 200)
                    self.assertNotIn("WWW-Authenticate", mission_response.headers)

                    # Voice endpoint does not return 401 for absent credentials
                    voice_response = client.post("/api/missions/private-id-123456/voice", json={
                        "mission_id": "private-id-123456", "mission": mission.model_dump(mode="json")
                    })
                    self.assertNotEqual(voice_response.status_code, 401)
                    self.assertNotIn("WWW-Authenticate", voice_response.headers)
            finally:
                main.app.dependency_overrides.clear()

    def test_validation_and_error_handling_work_without_authentication(self):
        with tempfile.TemporaryDirectory() as folder:
            main.app.dependency_overrides[get_store] = lambda: SessionStore(Path(folder) / "sessions.sqlite3")
            production = replace(settings, environment="production", demo_access_token="")
            try:
                with patch.object(main, "settings", production), TestClient(main.app) as client:
                    # Invalid mission input returns 422, not 401
                    invalid_mission = client.post("/api/missions", json={"available_time": 10})
                    self.assertEqual(invalid_mission.status_code, 422)
                    self.assertNotIn("WWW-Authenticate", invalid_mission.headers)

                    # Invalid session report returns 422, not 401
                    invalid_session = client.post("/api/sessions", json={"mission_id": "short"})
                    self.assertEqual(invalid_session.status_code, 422)
                    self.assertNotIn("WWW-Authenticate", invalid_session.headers)
            finally:
                main.app.dependency_overrides.clear()

    def test_secrets_are_not_exposed_in_public_endpoints_or_errors(self):
        secret_token = "secret-token-xyz-123"
        secret_openrouter = "sk-or-secret-key-456"
        secret_hf = "hf_secret_token_789"
        secret_eleven = "eleven_secret_key_abc"
        configured = replace(
            settings,
            environment="production",
            demo_access_token=secret_token,
            openrouter_api_key=secret_openrouter,
            hf_token=secret_hf,
            elevenlabs_api_key=secret_eleven,
        )
        with tempfile.TemporaryDirectory() as folder:
            main.app.dependency_overrides[get_store] = lambda: SessionStore(Path(folder) / "sessions.sqlite3")
            try:
                with patch.object(main, "settings", configured), TestClient(main.app) as client:
                    for endpoint in ["/health", "/ready", "/api/sessions"]:
                        res = client.get(endpoint)
                        for secret in [secret_token, secret_openrouter, secret_hf, secret_eleven]:
                            self.assertNotIn(secret, res.text)
            finally:
                main.app.dependency_overrides.clear()

    def test_cors_allows_only_configured_exact_origin_and_supports_basic_auth(self):
        application = FastAPI()
        configure_cors(application, ("https://demo.example",))
        application.post("/write")(lambda: {"ok": True})
        with TestClient(application) as client:
            allowed = client.options("/write", headers={
                "Origin": "https://demo.example",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization,content-type",
            })
            self.assertEqual(allowed.status_code, 200)
            self.assertEqual(allowed.headers["access-control-allow-origin"], "https://demo.example")
            self.assertEqual(allowed.headers["access-control-allow-credentials"], "true")
            rejected = client.options("/write", headers={
                "Origin": "https://unlisted.example",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization,content-type",
            })
            self.assertEqual(rejected.status_code, 400)
            self.assertNotIn("access-control-allow-origin", rejected.headers)
        with self.assertRaisesRegex(ValueError, "exact origins"):
            configure_cors(FastAPI(), ("*",))

        same_origin_app = FastAPI()
        configure_cors(same_origin_app, ())
        same_origin_app.get("/")(lambda: {"ok": True})
        with TestClient(same_origin_app) as client:
            response = client.get("/", headers={"Origin": "https://demo.example"})
            self.assertNotIn("access-control-allow-origin", response.headers)

    def test_render_free_blueprint_uses_ephemeral_sqlite_and_health_liveness(self):
        render = Path(__file__).resolve().parents[1] / "render.yaml"
        content = render.read_text(encoding="utf-8")
        self.assertIn("type: web", content)
        self.assertIn("plan: free", content)
        self.assertEqual(content.count("type: web"), 1)
        self.assertIn("--workers 1", content)
        self.assertIn("--host 0.0.0.0 --port $PORT", content)
        self.assertIn("healthCheckPath: /health", content)
        self.assertIn("DATA_FILE", content)
        self.assertIn("./data/sessions.sqlite3", content)
        self.assertIn("APP_ENV", content)
        self.assertIn("AI_PROVIDER", content)
        self.assertIn("value: openrouter", content)
        self.assertIn("google/gemma-3-4b-it", content)
        self.assertIn("OPENROUTER_API_KEY", content)
        self.assertIn("HF_TOKEN", content)
        self.assertIn("key: HF_TOKEN\n        sync: false", content)
        self.assertIn("DEMO_ACCESS_TOKEN", content)
        self.assertNotIn("OLLAMA_", content)
        self.assertNotIn("disk:", content)
        self.assertNotIn("Starter", content)
        self.assertNotIn("/var/data", content)
        self.assertIn("sync: false", content)
        self.assertEqual(main.health(), {"status": "ok", "check": "application_liveness"})

    def test_corrupt_database_error_does_not_reveal_filesystem_path(self):
        with tempfile.TemporaryDirectory() as folder:
            database = Path(folder) / "sessions.sqlite3"
            database.write_bytes(b"not a database")
            main.app.dependency_overrides[get_store] = lambda: SessionStore(database)
            try:
                with TestClient(main.app) as client:
                    response = client.get("/api/sessions")
                self.assertEqual(response.status_code, 500)
                self.assertNotIn(folder, response.text)
                self.assertNotIn("sqlite", response.text.lower())
            finally:
                main.app.dependency_overrides.clear()


if __name__ == "__main__":
    unittest.main()
