from __future__ import annotations

import asyncio
import json
import unittest
from dataclasses import replace

import httpx

from app.ai.base import AIProviderAuthenticationFailed, AIProviderInvalidOutput, AIProviderMisconfigured
from app.ai.huggingface import HuggingFaceGemmaProvider
from app.config import Settings, settings
from app import main
from app.models import Goal, MissionSetup

TOKEN = "hf-test-token-do-not-leak"


def mission_payload() -> dict:
    return {
        "title": "Sharp Finishing",
        "duration_minutes": 45,
        "goal": "finishing",
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


class HuggingFaceTests(unittest.TestCase):
    def setUp(self):
        self.config = replace(settings, ai_provider="huggingface", hf_token=TOKEN)

    def provider(self, handler):
        return HuggingFaceGemmaProvider(self.config, httpx.MockTransport(handler))

    def test_configuration_accepts_huggingface_and_defaults_to_gemma_route(self):
        config = replace(self.config, hf_model="google/gemma-3-4b-it", hf_base_url="https://router.huggingface.co/v1")
        self.assertEqual(config.hf_chat_url, "https://router.huggingface.co/v1/chat/completions")
        self.assertEqual(config.hf_model, "google/gemma-3-4b-it")
        with self.assertRaisesRegex(ValueError, "HF_MODEL"):
            HuggingFaceGemmaProvider(replace(config, hf_model="another/model"))
        with self.assertRaisesRegex(ValueError, "AI_PROVIDER"):
            Settings(ai_provider="unknown")

        previous = main.settings
        try:
            main.settings = config
            self.assertIsInstance(main.get_ai_provider(), HuggingFaceGemmaProvider)
        finally:
            main.settings = previous

    def test_mission_requests_strict_schema_and_runs_existing_validation(self):
        seen = {}

        def handler(request):
            seen["url"] = str(request.url)
            seen["auth"] = request.headers.get("authorization")
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(mission_payload())}}]})

        setup = MissionSetup(available_time=45, level="intermediate", goal="finishing", equipment="football_only")
        result = asyncio.run(self.provider(handler).generate_mission(setup, []))
        self.assertEqual(result.goal, Goal.finishing)
        self.assertEqual(seen["url"], "https://router.huggingface.co/v1/chat/completions")
        self.assertEqual(seen["auth"], f"Bearer {TOKEN}")
        self.assertEqual(seen["body"]["model"], "google/gemma-3-4b-it")
        self.assertEqual(seen["body"]["response_format"]["type"], "json_schema")
        self.assertTrue(seen["body"]["response_format"]["json_schema"]["strict"])

    def test_readiness_checks_live_provider_mapping_without_generation(self):
        def handler(request):
            self.assertEqual(request.url.host, "huggingface.co")
            self.assertEqual(request.url.params.get("expand[]"), "inferenceProviderMapping")
            return httpx.Response(200, json={
                "id": "google/gemma-3-4b-it",
                "inferenceProviderMapping": {
                    "deepinfra": {"status": "live"},
                    "featherless-ai": {"status": "live"},
                },
            })

        ready, message = asyncio.run(self.provider(handler).check_ready())
        self.assertTrue(ready)
        self.assertIn("live", message)

    def test_missing_token_does_not_send_request(self):
        provider = HuggingFaceGemmaProvider(replace(self.config, hf_token=""))
        ready, _ = asyncio.run(provider.check_ready())
        self.assertFalse(ready)
        setup = MissionSetup(available_time=45, level="intermediate", goal="finishing", equipment="football_only")
        with self.assertRaises(AIProviderMisconfigured):
            asyncio.run(provider.generate_mission(setup, []))

    def test_auth_failure_and_invalid_model_output_are_explicit_and_secret_safe(self):
        setup = MissionSetup(available_time=45, level="intermediate", goal="finishing", equipment="football_only")

        def unauthorized(request):
            return httpx.Response(401, text=TOKEN)

        with self.assertRaises(AIProviderAuthenticationFailed) as error:
            asyncio.run(self.provider(unauthorized).generate_mission(setup, []))
        self.assertNotIn(TOKEN, str(error.exception))

        def invalid(request):
            return httpx.Response(200, json={"choices": [{"message": {"content": '{"not":"a mission"}'}}]})

        with self.assertRaises(AIProviderInvalidOutput):
            asyncio.run(self.provider(invalid).generate_mission(setup, []))


if __name__ == "__main__":
    unittest.main()
