from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from app.models import SessionRecord


class SessionStoreError(Exception):
    """Local session history could not be read or written safely."""


class SessionStore:
    def __init__(self, path: Path):
        self.path = path

    def list_sessions(self, limit: int = 100) -> list[SessionRecord]:
        if not self.path.exists():
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                raise ValueError("Session history must be a JSON list.")
            return [SessionRecord.model_validate(row) for row in raw[-limit:]]
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise SessionStoreError("Session history could not be read.") from exc

    def add_session(self, record: SessionRecord) -> None:
        rows = self.list_sessions(limit=10_000)
        rows.append(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_name = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=self.path.parent, delete=False
            ) as temp:
                temp_name = temp.name
                json.dump([row.model_dump(mode="json") for row in rows[-100:]], temp, indent=2)
                temp.flush()
                os.fsync(temp.fileno())
            os.replace(temp_name, self.path)
        except OSError as exc:
            if temp_name and os.path.exists(temp_name):
                os.unlink(temp_name)
            raise SessionStoreError("Session history could not be saved.") from exc
