from __future__ import annotations

import asyncio
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from app.ai.base import AIProviderUnavailable
from app.main import app, get_ai_provider, get_store, get_voice_provider
from app.models import (
    Challenge,
    Drill,
    Goal,
    Mission,
    MissionSetup,
    NextSessionRecommendation,
    SessionReport,
    SessionReview,
)
from app.ai.gemma import GemmaProvider, parse_structured_output
from app.config import settings
from app.voice.base import VoiceProviderMisconfigured, VoiceProviderUnavailable
from app.voice.elevenlabs import ElevenLabsVoiceProvider
from app.voice.script import ENDING, build_coaching_script


def build_mission(setup: MissionSetup) -> Mission:
    # Forty scheduled minutes leaves five minutes of flexibility in the 45-minute fixture.
    drill_count = 3 if setup.available_time >= 45 else 1
    each = 10 if drill_count == 3 else max(5, setup.available_time - 10)
    return Mission(
        title="Sharp Finishing",
        duration_minutes=setup.available_time,
        goal=setup.goal,
        warmup_minutes=5,
        warmup=["Light movement and comfortable ball touches."],
        drills=[
            Drill(name=f"Finishing drill {number + 1}", duration_minutes=each,
                  instructions="Take controlled attempts at a safe target and reset between shots.")
            for number in range(drill_count)
        ],
        challenge=Challenge(description="Score 7 out of 10 attempts.", success_metric="7 successful attempts"),
        cooldown_minutes=5,
        cooldown=["Walk easily and take a few relaxed breaths."],
        motivation="Steady, focused practice is a win.",
        safety_note="Use a clear space, warm up, and stop if you feel pain or dizziness.",
    )


class MockAIProvider:
    name = "Mock AI provider (tests only)"

    def __init__(self):
        self.mission_histories = []
        self.review_calls = 0
        self.next_calls = 0
        self.unavailable = False
        self.ready = True

    async def check_ready(self):
        return self.ready, "Mock provider ready." if self.ready else "Mock provider unavailable."

    async def generate_mission(self, setup, history):
        self.mission_histories.append(history)
        if self.unavailable:
            raise AIProviderUnavailable("mock offline")
        return build_mission(setup)

    async def evaluate_session(self, mission, report, history):
        if self.unavailable:
            raise AIProviderUnavailable("mock offline")
        self.review_calls += 1
        return SessionReview(
            performance_summary=f"You completed {report.successful_attempts} successful attempts.",
            strongest_area="Staying focused on the routine",
            weakness="Repeat attempts under movement",
            actionable_recommendation="Start each attempt with a calm setup and balanced first touch.",
        )

    async def recommend_next_session(self, review, mission, report, history):
        if self.unavailable:
            raise AIProviderUnavailable("mock offline")
        self.next_calls += 1
        return NextSessionRecommendation(
            goal=Goal.finishing,
            available_time=45,
            level="intermediate",
            equipment="football_only",
            setting="ground",
            reason="Build on the completed session with a little movement before each shot.",
        )


class MockVoiceProvider:
    def __init__(self, audio=b"test-mp3"):
        self.audio = audio
        self.script = None
        self.failure = None

    async def synthesize(self, script):
        self.script = script
        if self.failure:
            raise self.failure
        return self.audio

class JobiGoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = __import__("app.store", fromlist=["SessionStore"]).SessionStore(
            Path(self.temp.name) / "sessions.json"
        )
        self.provider = MockAIProvider()
        self.voice_provider = MockVoiceProvider()
        app.dependency_overrides[get_store] = lambda: self.store
        app.dependency_overrides[get_ai_provider] = lambda: self.provider
        app.dependency_overrides[get_voice_provider] = lambda: self.voice_provider
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()
        self.client.close()
        self.temp.cleanup()

    def setup_body(self, **updates):
        body = {
            "available_time": 45,
            "level": "intermediate",
            "goal": "finishing",
            "equipment": "football_only",
            "setting": "ground",
        }
        body.update(updates)
        return body

    def mission_and_report(self):
        setup = MissionSetup.model_validate(self.setup_body())
        return build_mission(setup), {
            "attempts": 20,
            "successful_attempts": 13,
            "goals": 3,
            "completed_drills": 3,
            "difficulty": "right",
            "notes": "Weak foot felt harder.",
        }

    def test_home_health_and_empty_history(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/health").json(), {
            "status": "ok", "check": "application_liveness"
        })
        self.assertEqual(self.client.get("/ready").json()["status"], "ready")
        self.assertEqual(self.client.get("/api/sessions").json(), {"session_count": 0})

    def test_readiness_is_separate_from_liveness(self):
        self.provider.ready = False
        self.assertEqual(self.client.get("/health").status_code, 200)
        response = self.client.get("/ready")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"]["status"], "not_ready")

    def test_progress_endpoint_does_not_expose_saved_notes_or_identifiers(self):
        mission, report = self.mission_and_report()
        response = self.client.post("/api/sessions", json={
            "mission_id": "test-mission-private", "mission": mission.model_dump(mode="json"), "report": report
        })
        self.assertEqual(response.status_code, 200)
        history = self.client.get("/api/sessions").json()
        self.assertEqual(history, {"session_count": 1})
        self.assertNotIn("Weak foot felt harder", str(history))
        self.assertNotIn("test-mission-private", str(history))

    def test_mission_input_validation(self):
        self.assertEqual(self.client.post("/api/missions", json=self.setup_body()).status_code, 200)
        invalid_time = self.client.post("/api/missions", json=self.setup_body(available_time=25))
        self.assertEqual(invalid_time.status_code, 422)
        invalid_goal = self.client.post("/api/missions", json=self.setup_body(goal="become_pro")).status_code
        self.assertEqual(invalid_goal, 422)

    def test_gemma_json_parser_validates_schema_and_rejects_malformed_output(self):
        mission = build_mission(MissionSetup.model_validate(self.setup_body()))
        parsed = parse_structured_output(mission.model_dump_json(), Mission)
        self.assertEqual(parsed.goal, Goal.finishing)
        with self.assertRaises(ValueError):
            parse_structured_output("I could not make a plan.", Mission)
        with self.assertRaises(ValueError):
            parse_structured_output('{"title":"incomplete"}', Mission)

    def test_mission_is_rejected_if_it_requires_unavailable_equipment_or_invents_history(self):
        setup = MissionSetup.model_validate(self.setup_body())
        mission = build_mission(setup)
        mission.drills[0].instructions = "Set up cones, then take controlled attempts at a safe target."
        issue = GemmaProvider._mission_issue(mission, setup, [])
        self.assertIn("cones", issue)
        mission = build_mission(setup)
        mission.motivation = "You've been working hard on this skill."
        issue = GemmaProvider._mission_issue(mission, setup, [])
        self.assertIn("history", issue)

    def test_next_session_focus_follows_explicit_actionable_review(self):
        review = SessionReview(
            performance_summary="You completed 13 successful attempts.",
            strongest_area="Consistency",
            weakness="Weak-foot finishing needs more practice.",
            actionable_recommendation="Add controlled weak-foot finishing to the next session.",
        )
        self.assertEqual(GemmaProvider._recommended_focus(review, "dribbling"), "finishing")
        general = review.model_copy(update={
            "weakness": "The weaker foot felt harder.",
            "actionable_recommendation": "Practice controlled touches with your weaker foot.",
        })
        self.assertEqual(GemmaProvider._recommended_focus(general, "finishing"), "finishing")
        self.assertEqual(GemmaProvider._next_session_time(45, "right"), 45)
        self.assertEqual(GemmaProvider._next_session_time(45, "too_easy"), 60)
        self.assertEqual(GemmaProvider._next_session_time(45, "too_hard"), 30)

    def test_mission_generation_returns_explicit_provider_label(self):
        response = self.client.post("/api/missions", json=self.setup_body())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["ai_provider"], "Mock AI provider (tests only)")
        self.assertEqual(response.json()["mission"]["goal"], "finishing")

    def test_session_submission_runs_review_and_recommendation_then_saves_history(self):
        mission, report = self.mission_and_report()
        response = self.client.post("/api/sessions", json={
            "mission_id": "test-mission-0001", "mission": mission.model_dump(mode="json"), "report": report
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("13 successful attempts", data["review"]["performance_summary"])
        self.assertEqual(data["next_session"]["goal"], "finishing")
        self.assertEqual(self.provider.review_calls, 1)
        self.assertEqual(self.provider.next_calls, 1)
        history = self.client.get("/api/sessions").json()
        self.assertEqual(history, {"session_count": 1})
        self.assertNotIn("notes", history)
        self.assertNotIn("session_id", history)

    def test_progression_context_includes_prior_completed_session(self):
        mission, report = self.mission_and_report()
        body = {"mission_id": "test-mission-0001", "mission": mission.model_dump(mode="json"), "report": report}
        self.assertEqual(self.client.post("/api/sessions", json=body).status_code, 200)
        response = self.client.post("/api/missions", json=self.setup_body())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(self.provider.mission_histories[-1]), 1)
        self.assertEqual(self.provider.mission_histories[-1][0]["session_id"], "test-mission-0001")

    def test_invalid_session_report_is_rejected(self):
        mission, report = self.mission_and_report()
        report["successful_attempts"] = report["attempts"] + 1
        response = self.client.post("/api/sessions", json={
            "mission_id": "test-mission-0002", "mission": mission.model_dump(mode="json"), "report": report
        })
        self.assertEqual(response.status_code, 422)

    def test_unavailable_ai_is_reported_without_fabricated_success(self):
        self.provider.unavailable = True
        response = self.client.post("/api/missions", json=self.setup_body())
        self.assertEqual(response.status_code, 503)
        self.assertIn("mock offline", response.json()["detail"])

    def test_voice_endpoint_returns_audio_without_exposing_provider_secrets(self):
        mission, _ = self.mission_and_report()
        response = self.client.post("/api/missions/test-mission-0001/voice", json={
            "mission_id": "test-mission-0001", "mission": mission.model_dump(mode="json")
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/mpeg")
        self.assertEqual(response.content, b"test-mp3")
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertTrue(self.voice_provider.script.endswith(ENDING))
        self.assertNotIn("xi-api-key", response.text)

    def test_voice_missing_credentials_does_not_break_mission_creation(self):
        async def missing(_script):
            raise VoiceProviderMisconfigured("Coach audio is not configured. Your written mission is still ready.")
        self.voice_provider.synthesize = missing
        mission, _ = self.mission_and_report()
        voice = self.client.post("/api/missions/test-mission-0001/voice", json={
            "mission_id": "test-mission-0001", "mission": mission.model_dump(mode="json")
        })
        self.assertEqual(voice.status_code, 503)
        self.assertEqual(self.client.post("/api/missions", json=self.setup_body()).status_code, 200)

    def test_voice_upstream_failure_keeps_mission_and_go_outside_available(self):
        self.voice_provider.failure = VoiceProviderUnavailable("Audio service failed; text mission remains available.")
        mission, _ = self.mission_and_report()
        voice = self.client.post("/api/missions/test-mission-0001/voice", json={
            "mission_id": "test-mission-0001", "mission": mission.model_dump(mode="json")
        })
        self.assertEqual(voice.status_code, 502)
        self.assertIn("mission remains available", voice.json()["detail"])
        self.assertEqual(self.client.post("/api/missions", json=self.setup_body()).status_code, 200)

    def test_voice_request_mission_id_must_match_path(self):
        mission, _ = self.mission_and_report()
        response = self.client.post("/api/missions/path-mission-id/voice", json={
            "mission_id": "body-mission-id", "mission": mission.model_dump(mode="json")
        })
        self.assertEqual(response.status_code, 400)

    def test_coaching_script_is_concise_complete_and_ends_with_requested_line(self):
        mission, _ = self.mission_and_report()
        script = build_coaching_script(mission)
        self.assertLess(len(script), 900)
        self.assertIn("Sharp Finishing", script)
        self.assertIn("Warm up", script)
        self.assertIn("Challenge", script)
        self.assertIn("Stay safe", script)
        self.assertTrue(script.endswith("Your mission is ready. Put your phone away and go play."))

    def test_elevenlabs_adapter_sends_server_side_key_and_handles_success_and_errors(self):
        seen = {}
        def handler(request):
            seen["key"] = request.headers.get("xi-api-key")
            seen["body"] = request.read().decode()
            return httpx.Response(200, content=b"mp3", headers={"content-type": "audio/mpeg"})
        config = replace(settings, elevenlabs_api_key="test-secret-never-return", elevenlabs_voice_id="voice-id")
        audio = asyncio.run(ElevenLabsVoiceProvider(config, httpx.MockTransport(handler)).synthesize("short script"))
        self.assertEqual(audio, b"mp3")
        self.assertEqual(seen["key"], "test-secret-never-return")
        self.assertIn('"model_id":"eleven_multilingual_v2"', seen["body"])

        def api_error(_request):
            return httpx.Response(401, json={"detail": "secret-bearing upstream response"})
        with self.assertRaises(VoiceProviderUnavailable):
            asyncio.run(ElevenLabsVoiceProvider(config, httpx.MockTransport(api_error)).synthesize("short script"))

        def network_error(_request):
            raise httpx.ConnectError("offline")
        with self.assertRaises(VoiceProviderUnavailable):
            asyncio.run(ElevenLabsVoiceProvider(config, httpx.MockTransport(network_error)).synthesize("short script"))

    def test_elevenlabs_missing_credentials_are_rejected_without_network(self):
        config = replace(settings, elevenlabs_api_key="", elevenlabs_voice_id="")
        with self.assertRaises(VoiceProviderMisconfigured):
            asyncio.run(ElevenLabsVoiceProvider(config).synthesize("short script"))


if __name__ == "__main__":
    unittest.main()
