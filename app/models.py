from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Choice(str, Enum):
    beginner = "beginner"
    intermediate = "intermediate"
    advanced = "advanced"


class Goal(str, Enum):
    finishing = "finishing"
    dribbling = "dribbling"
    passing = "passing"
    first_touch = "first_touch"
    speed_agility = "speed_agility"
    ball_control = "ball_control"
    general_fitness = "general_fitness"


class Equipment(str, Enum):
    football_only = "football_only"
    football_cones = "football_cones"
    football_goal = "football_goal"
    full_equipment = "full_equipment"


class Setting(str, Enum):
    ground = "ground"
    turf = "turf"
    park = "park"
    open_space = "open_space"


class Difficulty(str, Enum):
    too_easy = "too_easy"
    right = "right"
    too_hard = "too_hard"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class MissionSetup(StrictModel):
    available_time: int = Field(ge=20, le=90)
    level: Choice
    goal: Goal
    equipment: Equipment
    setting: Setting | None = None

    @model_validator(mode="after")
    def supported_time(self) -> "MissionSetup":
        if self.available_time not in {20, 30, 45, 60, 90}:
            raise ValueError("Choose 20, 30, 45, 60, or 90 minutes.")
        return self


class Drill(StrictModel):
    name: str = Field(min_length=2, max_length=80)
    duration_minutes: int = Field(ge=1, le=90)
    instructions: str = Field(min_length=8, max_length=500)


class Challenge(StrictModel):
    description: str = Field(min_length=4, max_length=240)
    success_metric: str = Field(min_length=2, max_length=120)


class Mission(StrictModel):
    title: str = Field(min_length=3, max_length=100)
    duration_minutes: int = Field(ge=20, le=90)
    goal: Goal
    warmup_minutes: int = Field(ge=3, le=20)
    warmup: list[str] = Field(min_length=1, max_length=4)
    drills: list[Drill] = Field(min_length=1, max_length=6)
    challenge: Challenge
    cooldown_minutes: int = Field(ge=2, le=15)
    cooldown: list[str] = Field(min_length=1, max_length=4)
    motivation: str = Field(min_length=4, max_length=240)
    safety_note: str = Field(min_length=12, max_length=240)

    @model_validator(mode="after")
    def fits_session(self) -> "Mission":
        scheduled = self.warmup_minutes + self.cooldown_minutes + sum(
            drill.duration_minutes for drill in self.drills
        )
        if scheduled > self.duration_minutes:
            raise ValueError("Warm-up, drills, and cooldown must fit the mission duration.")
        return self


class SessionReport(StrictModel):
    attempts: int = Field(ge=0, le=500)
    successful_attempts: int = Field(ge=0, le=500)
    goals: int = Field(ge=0, le=200)
    completed_drills: int = Field(ge=0, le=6)
    difficulty: Difficulty
    notes: str = Field(default="", max_length=500)

    @model_validator(mode="after")
    def attempts_consistent(self) -> "SessionReport":
        if self.successful_attempts > self.attempts:
            raise ValueError("Successful attempts cannot exceed total attempts.")
        return self


class SessionReview(StrictModel):
    performance_summary: str = Field(min_length=8, max_length=500)
    strongest_area: str = Field(min_length=2, max_length=160)
    weakness: str = Field(min_length=2, max_length=160)
    actionable_recommendation: str = Field(min_length=8, max_length=300)


class NextSessionRecommendation(StrictModel):
    goal: Goal
    available_time: int = Field(ge=20, le=90)
    level: Choice
    equipment: Equipment
    setting: Setting | None = None
    reason: str = Field(min_length=8, max_length=300)

    @model_validator(mode="after")
    def supported_time(self) -> "NextSessionRecommendation":
        if self.available_time not in {20, 30, 45, 60, 90}:
            raise ValueError("Recommended session time must be 20, 30, 45, 60, or 90 minutes.")
        return self


class MissionResponse(StrictModel):
    mission_id: str
    mission: Mission
    ai_provider: str


class SessionProgress(StrictModel):
    session_count: int = Field(ge=0)


class SessionSubmission(StrictModel):
    mission_id: str = Field(min_length=8, max_length=80)
    mission: Mission
    report: SessionReport


class SessionRecord(StrictModel):
    session_id: str
    mission: Mission
    report: SessionReport
    review: SessionReview
    next_session: NextSessionRecommendation
    completed_at: str
