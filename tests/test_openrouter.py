from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from app.ai.base import (
    AIProviderAuthenticationFailed,
    AIProviderInvalidOutput,
    AIProviderMisconfigured,
    AIProviderRateLimited,
    AIProviderTimeout,
    AIProviderUpstreamFailure,
)
from app.ai.gemma import GemmaProvider
from app.ai.openrouter import OpenRouterGemmaProvider, REQUIRED_MODEL, _openrouter_schema
from app.config import settings
from app.main import app, get_ai_provider, get_store
from app.models import (
    Goal,
    Mission,
    MissionSetup,
    SessionReport,
)
from app.store import SessionStore

API_KEY = "test-openrouter-key-do-not-leak"


def mission_payload(goal: str = "finishing") -> dict:
    return {
        "title": "Sharp Finishing",
        "duration_minutes": 45,
        "goal": goal,
        "warmup_minutes": 5,
        "warmup": ["Move lightly and get comfortable with the ball."],
        "drills": [{
            "name": "Controlled shots",
            "duration_minutes": 30,
            "instructions": "Take controlled shots toward an imaginary safe target.",
        }],
        "challenge": {"description": "Score 7 of 10 attempts.", "success_metric": "7 successful attempts"},
        "cooldown_minutes": 5,
        "cooldown": ["Walk easily and relax."],
        "motivation": "Stay focused and enjoy each repetition.",
        "safety_note": "Use a clear space and stop if anything hurts.",
    }


def completion(content: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


class OpenRouterTests(unittest.TestCase):
    def setUp(self):
        self.config = replace(
            settings,
            ai_provider="openrouter",
            openrouter_api_key=API_KEY,
            openrouter_model=REQUIRED_MODEL,
        )

    def provider(self, handler):
        return OpenRouterGemmaProvider(self.config, httpx.MockTransport(handler))

    def test_successful_mission_uses_exact_model_and_json_schema_routing(self):
        seen = {}
        self.assertEqual(REQUIRED_MODEL, "google/gemma-3-4b-it:free")

        def handler(request):
            seen["url"] = str(request.url)
            seen["authorization"] = request.headers.get("authorization")
            seen["body"] = json.loads(request.content)
            return completion(json.dumps(mission_payload()))

        setup = MissionSetup(available_time=45, level="intermediate", goal="finishing", equipment="football_only")
        mission = asyncio.run(self.provider(handler).generate_mission(setup, []))
        self.assertEqual(mission.goal, Goal.finishing)
        self.assertTrue(seen["url"].endswith("/chat/completions"))
        self.assertEqual(seen["authorization"], f"Bearer {API_KEY}")
        self.assertEqual(seen["body"]["model"], REQUIRED_MODEL)
        self.assertEqual(seen["body"]["response_format"]["type"], "json_schema")
        self.assertTrue(seen["body"]["response_format"]["json_schema"]["strict"])
        self.assertTrue(seen["body"]["provider"]["require_parameters"])

    def test_successful_session_review_is_validated(self):
        review = {
            "performance_summary": "You completed 13 of 20 attempts with steady focus.",
            "strongest_area": "Consistent effort",
            "weakness": "Finishing after movement",
            "actionable_recommendation": "Add a controlled touch before each finishing attempt.",
        }
        mission = Mission.model_validate(mission_payload())
        report = SessionReport(
            attempts=20, successful_attempts=13, goals=3, completed_drills=1,
            difficulty="right", notes="Weak foot felt harder.",
        )
        result = asyncio.run(self.provider(lambda _request: completion(json.dumps(review))).evaluate_session(
            mission, report, []
        ))
        self.assertEqual(result.weakness, "Finishing after movement")

    def test_missing_api_key_fails_clearly_without_network(self):
        provider = OpenRouterGemmaProvider(replace(self.config, openrouter_api_key=""))
        with self.assertRaisesRegex(AIProviderMisconfigured, "OPENROUTER_API_KEY"):
            asyncio.run(provider._complete("coach", "{}", Mission))

    def test_wrong_model_configuration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly"):
            replace(self.config, openrouter_model="google/gemma-3-12b-it")

    def test_upstream_statuses_are_mapped_without_exposing_body(self):
        cases = [
            (401, AIProviderAuthenticationFailed),
            (403, AIProviderAuthenticationFailed),
            (429, AIProviderRateLimited),
            (503, AIProviderUpstreamFailure),
        ]
        for status, error_type in cases:
            with self.subTest(status=status):
                def handler(_request, status=status):
                    return httpx.Response(status, json={"error": f"sensitive body {API_KEY}"})
                with self.assertRaises(error_type) as raised:
                    asyncio.run(self.provider(handler)._complete("coach", "{}", Mission))
                self.assertNotIn(API_KEY, str(raised.exception))

    def test_rejected_schema_logs_safe_provider_diagnostic_only(self):
        body = {
            "error": {
                "code": 400,
                "message": "Provider returned error",
                "metadata": {
                    "provider_name": "DeepInfra",
                    "raw": json.dumps({
                        "error": {
                            "code": "INVALID_ARGUMENT",
                            "message": f"Unsupported response schema; bearer {API_KEY}"
                        }
                    }),
                },
            },
            "debug": f"request body contains {API_KEY}",
        }

        def handler(_request):
            return httpx.Response(400, json=body)

        with self.assertLogs("app.ai.openrouter", level="WARNING") as captured:
            with self.assertRaises(AIProviderUpstreamFailure) as raised:
                asyncio.run(self.provider(handler)._complete("private prompt", "private input", Mission))
        logs = "\n".join(captured.output)
        self.assertIn("DeepInfra", logs)
        self.assertIn("Unsupported response schema", logs)
        self.assertIn("INVALID_ARGUMENT", logs)
        self.assertNotIn(API_KEY, logs)
        self.assertNotIn("private prompt", logs)
        self.assertNotIn("private input", logs)
        self.assertNotIn(API_KEY, str(raised.exception))

    def test_payment_required_explains_credit_block_without_exposing_upstream_body(self):
        def handler(_request):
            return httpx.Response(402, json={
                "error": {
                    "code": 402,
                    "message": f"Insufficient credits for key {API_KEY}",
                }
            })

        with self.assertLogs("app.ai.openrouter", level="WARNING") as captured:
            with self.assertRaises(AIProviderUpstreamFailure) as raised:
                asyncio.run(self.provider(handler)._complete("coach", "{}", Mission))
        self.assertIn("insufficient account credits", str(raised.exception).lower())
        self.assertIn("funded account", str(raised.exception).lower())
        self.assertNotIn(API_KEY, str(raised.exception))
        self.assertNotIn(API_KEY, "\n".join(captured.output))

    def test_openrouter_schema_is_flattened_and_strictly_closed(self):
        schema = _openrouter_schema(Mission)
        encoded = json.dumps(schema)
        self.assertNotIn("$defs", encoded)
        self.assertNotIn("$ref", encoded)

        def assert_strict_objects(node):
            if isinstance(node, dict):
                if node.get("type") == "object":
                    self.assertFalse(node.get("additionalProperties", True))
                    self.assertEqual(set(node.get("required", [])), set(node.get("properties", {})))
                for value in node.values():
                    assert_strict_objects(value)
            elif isinstance(node, list):
                for value in node:
                    assert_strict_objects(value)

        assert_strict_objects(schema)

    def test_network_timeout_is_classified(self):
        def handler(_request):
            raise httpx.ReadTimeout("timeout")
        with self.assertRaises(AIProviderTimeout):
            asyncio.run(self.provider(handler)._complete("coach", "{}", Mission))

    def test_malformed_json_response_is_rejected(self):
        with self.assertRaises(AIProviderInvalidOutput):
            asyncio.run(self.provider(lambda _request: httpx.Response(200, content=b"{bad json"))._complete(
                "coach", "{}", Mission
            ))

    def test_invalid_structured_output_is_rejected(self):
        with self.assertRaises(AIProviderInvalidOutput):
            asyncio.run(self.provider(lambda _request: completion('{"title":"incomplete"}'))._complete(
                "coach", "{}", Mission
            ))

    def test_history_sent_to_model_omits_notes_identifiers_and_unneeded_text(self):
        history = [{
            "session_id": "private-session-id",
            "mission": {"goal": "finishing", "duration_minutes": 45, "title": "Private title"},
            "report": {"attempts": 10, "successful_attempts": 6, "goals": 2,
                       "completed_drills": 2, "difficulty": "right", "notes": "private player note"},
            "review": {"weakness": "weak foot", "actionable_recommendation": "practice carefully",
                       "performance_summary": "long prior summary"},
            "next_session": {"goal": "finishing", "available_time": 45, "reason": "long plan text"},
        }]
        context = GemmaProvider._history_context(history)
        encoded = json.dumps(context)
        self.assertNotIn("private-session-id", encoded)
        self.assertNotIn("private player note", encoded)
        self.assertNotIn("Private title", encoded)
        self.assertNotIn("long prior summary", encoded)
        self.assertNotIn("long plan text", encoded)
        self.assertEqual(context[0]["goal"], "finishing")

    def test_existing_mission_validation_and_correction_retry_are_preserved(self):
        responses = [mission_payload("dribbling"), mission_payload("finishing")]
        calls = []

        def handler(request):
            body = json.loads(request.content)
            calls.append(body)
            return completion(json.dumps(responses.pop(0)))

        setup = MissionSetup(available_time=45, level="intermediate", goal="finishing", equipment="football_only")
        result = asyncio.run(self.provider(handler).generate_mission(setup, []))
        self.assertEqual(result.goal, Goal.finishing)
        self.assertEqual(len(calls), 2)
        self.assertIn("correction", calls[1]["messages"][1]["content"])

    def test_api_key_never_appears_in_api_response(self):
        temporary = tempfile.TemporaryDirectory()
        provider = self.provider(lambda _request: completion(json.dumps(mission_payload())))
        app.dependency_overrides[get_ai_provider] = lambda: provider
        app.dependency_overrides[get_store] = lambda: SessionStore(Path(temporary.name) / "sessions.json")
        try:
            with TestClient(app) as client:
                response = client.post("/api/missions", json={
                    "available_time": 45, "level": "intermediate", "goal": "finishing",
                    "equipment": "football_only", "setting": "ground",
                })
            self.assertEqual(response.status_code, 200)
            self.assertNotIn(API_KEY, response.text)
        finally:
            app.dependency_overrides.clear()
            temporary.cleanup()

    def test_application_errors_have_clear_status_and_do_not_leak_secret(self):
        errors = [
            (AIProviderMisconfigured("OpenRouter is selected but OPENROUTER_API_KEY is not configured on the server."), 503),
            (AIProviderAuthenticationFailed("OpenRouter rejected the server-side API key."), 502),
            (AIProviderRateLimited("rate limited"), 503),
            (AIProviderUpstreamFailure("upstream failure"), 502),
            (AIProviderTimeout("timed out"), 504),
        ]
        class ErrorProvider:
            name = "test"
            async def check_ready(self): return True, "ready"
            async def generate_mission(self, *_args): raise errors.pop(0)[0]
            async def evaluate_session(self, *_args): raise AssertionError("unused")
            async def recommend_next_session(self, *_args): raise AssertionError("unused")

        temporary = tempfile.TemporaryDirectory()
        app.dependency_overrides[get_ai_provider] = lambda: ErrorProvider()
        app.dependency_overrides[get_store] = lambda: SessionStore(Path(temporary.name) / "sessions.json")
        try:
            with TestClient(app) as client:
                details = []
                for _error, expected in list(errors):
                    response = client.post("/api/missions", json={
                        "available_time": 45, "level": "intermediate", "goal": "finishing",
                        "equipment": "football_only", "setting": "ground",
                    })
                    details.append(response.text)
                    self.assertEqual(response.status_code, expected)
            self.assertTrue(all(API_KEY not in text for text in details))
        finally:
            app.dependency_overrides.clear()
            temporary.cleanup()

    def test_readiness_check_is_authenticated_and_does_not_return_api_key(self):
        seen = {}
        def handler(request):
            seen["authorization"] = request.headers.get("authorization")
            return httpx.Response(200, json={"data": [{"id": REQUIRED_MODEL}]})
        ready, _detail = asyncio.run(self.provider(handler).check_ready())
        self.assertTrue(ready)
        self.assertEqual(seen["authorization"], f"Bearer {API_KEY}")
        self.assertNotIn(API_KEY, _detail)

    def test_existing_ollama_transport_remains_available(self):
        seen = {}
        def handler(request):
            seen["url"] = str(request.url)
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json={"message": {"content": json.dumps(mission_payload())}})
        local_config = replace(settings, gemma_model="gemma3:4b", gemma_base_url="http://localhost:11434")
        setup = MissionSetup(available_time=45, level="intermediate", goal="finishing", equipment="football_only")
        result = asyncio.run(GemmaProvider(local_config, httpx.MockTransport(handler)).generate_mission(setup, []))
        self.assertEqual(result.goal, Goal.finishing)
        self.assertTrue(seen["url"].endswith("/api/chat"))
        self.assertEqual(seen["body"]["model"], "gemma3:4b")
        self.assertIn("format", seen["body"])


if __name__ == "__main__":
    unittest.main()
