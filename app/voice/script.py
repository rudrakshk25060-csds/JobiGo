from __future__ import annotations

from app.models import Mission

ENDING = "Your mission is ready. Put your phone away and go play."


def _short(value: str, limit: int) -> str:
    value = " ".join(value.split())
    if len(value) <= limit:
        return value
    return value[: limit - 1].rsplit(" ", 1)[0] + "…"


def build_coaching_script(mission: Mission) -> str:
    """Make a short spoken briefing from the current mission, not session notes."""
    lines = [
        f"Your {mission.duration_minutes}-minute {mission.goal.value.replace('_', ' ')} mission: {_short(mission.title, 70)}.",
        f"Warm up for {mission.warmup_minutes} minutes: {_short(mission.warmup[0], 130)}",
    ]
    for index, drill in enumerate(mission.drills[:2], start=1):
        lines.append(
            f"Drill {index}, {drill.duration_minutes} minutes: {_short(drill.name, 55)}. "
            f"{_short(drill.instructions, 125)}"
        )
    lines.append(
        f"Challenge: {_short(mission.challenge.description, 115)} "
        f"Aim for {_short(mission.challenge.success_metric, 70)}."
    )
    lines.append(f"Stay safe: {_short(mission.safety_note, 115)}")
    lines.append(_short(mission.motivation, 100))
    lines.append(ENDING)
    return "\n".join(lines)
