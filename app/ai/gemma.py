from __future__ import annotations

import json
import re
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.ai.base import AIProviderInvalidOutput, AIProviderUnavailable
from app.config import Settings
from app.models import (
    Mission,
    MissionSetup,
    NextSessionRecommendation,
    SessionReport,
    SessionReview,
)

T = TypeVar("T", bound=BaseModel)
STANDARD_SAFETY_NOTE = (
    "Choose a clear, suitable playing area, warm up properly, drink water, "
    "and stop if you feel pain or dizziness."
)


def parse_structured_output(raw: str, schema: type[T]) -> T:
    """Parse JSON (including a fenced JSON response) into a validated model."""
    candidate = raw.strip()
    candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.I)
    if not candidate.startswith("{"):
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Model response did not contain a JSON object.")
        candidate = candidate[start : end + 1]
    return schema.model_validate_json(candidate)


class GemmaProvider:
    name = "Gemma 3 4B via local Ollama"

    def __init__(self, config: Settings, transport: httpx.AsyncBaseTransport | None = None):
        self.config = config
        self.transport = transport

    @property
    def model_identifier(self) -> str:
        return self.config.gemma_model

    async def check_ready(self) -> tuple[bool, str]:
        """Check whether Ollama responds and has the configured model installed."""
        try:
            async with httpx.AsyncClient(
                timeout=min(self.config.gemma_timeout_seconds, 3.0), transport=self.transport
            ) as client:
                response = await client.get(self.config.gemma_tags_url)
                response.raise_for_status()
                models = response.json().get("models", [])
        except (httpx.HTTPError, ValueError, TypeError, AttributeError):
            return False, "Ollama is unavailable or returned an invalid model list."

        if not isinstance(models, list):
            return False, "Ollama returned an invalid model list."
        available = {entry.get("name") for entry in models if isinstance(entry, dict)}
        if self.config.gemma_model not in available:
            return False, f"Configured model '{self.config.gemma_model}' is not installed in Ollama."
        return True, f"Configured model '{self.config.gemma_model}' is available."

    async def _complete(self, system: str, prompt: str, schema: type[T]) -> T:
        try:
            async with httpx.AsyncClient(
                timeout=self.config.gemma_timeout_seconds, transport=self.transport
            ) as client:
                response = await client.post(
                    self.config.gemma_chat_url,
                    json={
                        "model": self.config.gemma_model,
                        "stream": False,
                        "format": schema.model_json_schema(),
                        "options": {"temperature": 0},
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": prompt},
                        ],
                    },
                )
                response.raise_for_status()
                raw = response.json().get("message", {}).get("content", "")
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            raise AIProviderUnavailable(
                "Could not reach local Gemma through Ollama. Check Ollama and the model name."
            ) from exc
        try:
            return parse_structured_output(raw, schema)
        except (ValueError, ValidationError) as exc:
            raise AIProviderInvalidOutput(
                "Gemma returned a response that did not match the required structure. Please retry."
            ) from exc

    async def generate_mission(self, setup: MissionSetup, history: list[dict]) -> Mission:
        system = (
            "You are JobiGo, a supportive football coach. Return only the JSON object required by the schema. "
            "Design a safe, achievable outdoor football mission using only the selected equipment and the stated level. "
            "Equipment meanings: football_only means exactly one football and no other object or person; football_cones "
            "means football and cones; football_goal means football and goal; full_equipment allows normal football gear. "
            "For football_only, make every drill solo and use only the ball and open space: DO NOT mention or require "
            "cones, markers, poles, goal, net, wall, partner, or rebounder. Use an imaginary target or a line already "
            "on the ground. Never prescribe equipment outside the selected option. When recent_sessions is empty, do not claim Jobi has "
            "practiced, improved, or has a known weakness before. Do not infer a preferred foot or past performance. "
            "Every drill and the challenge must directly train the selected goal; do not substitute passing for finishing. "
            "For finishing with football_only, practice controlled shooting technique at an imaginary target area in "
            "open space, then retrieve the ball at a walk; do not require a physical target. "
            "Keep all timed activities within the selected duration. Avoid roads, traffic, unsafe surfaces, maximal or "
            "explosive effort, collisions, and medical claims. Include a warm-up, drills, a measurable challenge, "
            "cooldown, motivation, and a short safety note."
        )
        inputs = {
            "setup": setup.model_dump(mode="json"),
            "recent_sessions": self._history_context(history),
        }
        last_issue = ""
        for attempt in range(2):
            if last_issue:
                inputs["correction"] = (
                    f"Your previous draft failed validation: {last_issue} Fix that issue in the new JSON."
                )
            prompt = json.dumps(inputs, ensure_ascii=False)
            try:
                mission = await self._complete(system, prompt, Mission)
                issue = self._mission_issue(mission, setup, history)
            except AIProviderInvalidOutput as exc:
                issue = str(exc)
            if not issue:
                # A deterministic app-level safety message is shown on every mission.
                return mission.model_copy(update={"safety_note": STANDARD_SAFETY_NOTE})
            last_issue = issue
        raise AIProviderInvalidOutput(
            "Gemma could not produce a mission that matches your time, goal, and equipment. "
            "Try again or choose different equipment."
        )

    @staticmethod
    def _mission_issue(mission: Mission, setup: MissionSetup, history: list[dict]) -> str:
        if mission.duration_minutes != setup.available_time:
            return "mission duration must match the selected time"
        if mission.goal != setup.goal:
            return "mission goal must match the selected goal"
        if not history and re.search(
            r"\b(you(?:'|’)ve been|you have been|last session|previous session|you improved|your weak foot|your dominant foot)\b",
            mission.motivation,
            re.I,
        ):
            return "do not claim previous practice or personal history when none was provided"
        activity = " ".join(
            [*mission.warmup, *mission.cooldown, mission.challenge.description, mission.challenge.success_metric]
            + [f"{drill.name} {drill.instructions}" for drill in mission.drills]
        ).lower()
        if setup.goal == "finishing" and not re.search(r"\b(finish(?:ing)?|shoot(?:ing)?|shots?)\b", activity):
            return "the activities must actually practice finishing or shooting, not just passing"
        asks_for_cones = re.search(r"\b(cone|cones|markers?|training poles?)\b", activity)
        asks_for_goal = re.search(r"\b(goal|net)\b", activity)
        asks_for_extra = re.search(r"\b(wall|partner|rebounder)\b", activity)
        if setup.equipment in {"football_only", "football_goal"} and asks_for_cones:
            return "the drills must not require cones or markers with this equipment choice"
        if setup.equipment in {"football_only", "football_cones"} and asks_for_goal:
            return "the drills must not require a goal or net with this equipment choice"
        if setup.equipment == "football_only" and asks_for_extra:
            return "football-only drills must be solo and use only the football and open space"
        return ""

    async def evaluate_session(
        self, mission: Mission, report: SessionReport, history: list[dict]
    ) -> SessionReview:
        system = (
            "You are JobiGo, a supportive football coach. Return only JSON matching the schema. "
            "Evaluate only the provided mission and report. Be constructive, specific, concise, and safe. "
            "Interpret each number literally: successful_attempts is the player's self-reported successful attempts, "
            "not automatically shots on target; goals is the reported goal count; completed_drills is a count. "
            "Do not claim a drill was completed beyond that count. Do not infer a preferred foot or a proven strength. "
            "Use the notes as the player's own observation, not as proof of a measured skill. Do not make medical claims. "
            "Write complete sentences and end each text field cleanly."
        )
        prompt = json.dumps(
            {
                "mission": mission.model_dump(mode="json"),
                "result": report.model_dump(mode="json"),
                "recent_sessions": self._history_context(history),
            },
            ensure_ascii=False,
        )
        return await self._complete(system, prompt, SessionReview)

    async def recommend_next_session(
        self,
        review: SessionReview,
        mission: Mission,
        report: SessionReport,
        history: list[dict],
    ) -> NextSessionRecommendation:
        system = (
            "You are JobiGo, a supportive football coach. Return only JSON matching the schema. "
            "Recommend one progressive outdoor session that directly addresses the weakness and actionable recommendation. "
            "Use the completed mission's goal as the next goal unless the review explicitly recommends a different named skill. "
            "Do not jump to an unrelated skill. Keep the same level and equipment. Use the player's difficulty rating: "
            "if it was right, keep the same time; if too_easy, move up one available time step; if too_hard, move down one step. "
            "Keep intensity safe. Give a concise, complete reason that clearly matches the selected next-session goal. "
            "End the reason with a complete sentence."
        )
        inputs = {
            "latest_review": review.model_dump(mode="json"),
            "completed_mission": mission.model_dump(mode="json"),
            "player_report": report.model_dump(mode="json"),
            "recent_sessions": self._history_context(history),
        }
        desired_goal = self._recommended_focus(review, mission.goal.value)
        desired_time = self._next_session_time(mission.duration_minutes, report.difficulty.value)
        last_issue = ""
        for _ in range(2):
            if last_issue:
                inputs["correction"] = last_issue
            recommendation = await self._complete(
                system, json.dumps(inputs, ensure_ascii=False), NextSessionRecommendation
            )
            if recommendation.goal.value == desired_goal and recommendation.available_time == desired_time:
                return recommendation
            last_issue = f"Set next-session goal to {desired_goal} and available_time to {desired_time}. Align the reason."
        raise AIProviderInvalidOutput(
            "Gemma could not align the next session with its review. Please retry."
        )

    @staticmethod
    def _history_context(history: list[dict]) -> list[dict]:
        """Keep only coaching-relevant prior outcomes; omit IDs, notes and repeated plan text."""
        context: list[dict] = []
        for record in history[-5:]:
            mission = record.get("mission", {})
            report = record.get("report", {})
            review = record.get("review", {})
            next_session = record.get("next_session", {})
            context.append({
                "goal": mission.get("goal"),
                "duration_minutes": mission.get("duration_minutes"),
                "attempts": report.get("attempts"),
                "successful_attempts": report.get("successful_attempts"),
                "goals": report.get("goals"),
                "completed_drills": report.get("completed_drills"),
                "difficulty": report.get("difficulty"),
                "weakness": review.get("weakness"),
                "actionable_recommendation": review.get("actionable_recommendation"),
                "next_goal": next_session.get("goal"),
                "next_duration_minutes": next_session.get("available_time"),
            })
        return context

    @staticmethod
    def _recommended_focus(review: SessionReview, fallback_goal: str) -> str:
        """Use an explicitly named actionable skill, otherwise continue the completed mission's goal."""
        action = f"{review.actionable_recommendation} {review.weakness}".lower()
        candidates = [
            (r"\b(finishing|finish|shooting|shoot|shots?)\b", "finishing"),
            (r"\b(dribbling|dribble)\b", "dribbling"),
            (r"\b(passing|pass)\b", "passing"),
            (r"\b(first touch)\b", "first_touch"),
            (r"\b(speed|agility|sprint)\b", "speed_agility"),
            (r"\b(ball control|control touches)\b", "ball_control"),
            (r"\b(general fitness|endurance|fitness)\b", "general_fitness"),
        ]
        matches = [
            (found.start(), goal)
            for pattern, goal in candidates
            if (found := re.search(pattern, action))
        ]
        return min(matches)[1] if matches else fallback_goal

    @staticmethod
    def _next_session_time(current_time: int, difficulty: str) -> int:
        options = [20, 30, 45, 60, 90]
        index = options.index(current_time)
        if difficulty == "too_easy":
            index = min(index + 1, len(options) - 1)
        elif difficulty == "too_hard":
            index = max(index - 1, 0)
        return options[index]
