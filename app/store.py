from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import closing
from pathlib import Path

from app.models import SessionRecord

_DB_INIT_LOCK = threading.Lock()


class SessionStoreError(Exception):
    """Local session history could not be read, migrated, or written safely."""


class SessionStore:
    """Small SQLite history store; SQLite serializes writers and commits atomically."""

    MAX_SESSIONS = 100

    def __init__(self, path: Path):
        configured_path = Path(path)
        # Compatibility with older local .env files which pointed DATA_FILE at sessions.json.
        # Keep the JSON file as a backup and write the migrated history beside it.
        if configured_path.suffix.lower() == ".json":
            self.path = configured_path.with_suffix(".sqlite3")
            self.legacy_json_path = configured_path
        else:
            self.path = configured_path
            self.legacy_json_path = configured_path.with_suffix(".json")

    def _connect(self) -> sqlite3.Connection:
        # Make first-run schema creation and legacy import single-flight within this process.
        with _DB_INIT_LOCK:
            return self._connect_locked()

    def _connect_locked(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        database_exists = self.path.exists()
        connection = sqlite3.connect(self.path, timeout=15.0)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA busy_timeout = 15000")
            if not database_exists:
                connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS sessions ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, payload TEXT NOT NULL)"
            )
            if not database_exists:
                self._migrate_legacy_json(connection)
            return connection
        except Exception:
            connection.close()
            if not database_exists:
                for suffix in ("", "-wal", "-shm"):
                    try:
                        Path(f"{self.path}{suffix}").unlink(missing_ok=True)
                    except OSError:
                        pass
            raise

    def _migrate_legacy_json(self, connection: sqlite3.Connection) -> None:
        if not self.legacy_json_path.exists():
            return
        try:
            raw = json.loads(self.legacy_json_path.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                raise ValueError("Session history must be a JSON list.")
            records = [SessionRecord.model_validate(row) for row in raw[-self.MAX_SESSIONS :]]
            with connection:
                connection.executemany(
                    "INSERT INTO sessions (payload) VALUES (?)",
                    [(record.model_dump_json(),) for record in records],
                )
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
            raise SessionStoreError("Existing session history could not be migrated safely.") from exc

    def list_sessions(self, limit: int = MAX_SESSIONS) -> list[SessionRecord]:
        try:
            with closing(self._connect()) as connection:
                rows = connection.execute(
                    "SELECT payload FROM sessions ORDER BY id DESC LIMIT ?",
                    (max(0, limit),),
                ).fetchall()
            return [SessionRecord.model_validate_json(row["payload"]) for row in reversed(rows)]
        except SessionStoreError:
            raise
        except (sqlite3.Error, ValueError, TypeError) as exc:
            raise SessionStoreError("Session history could not be read safely.") from exc

    def session_count(self) -> int:
        try:
            with closing(self._connect()) as connection:
                count = connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            return min(count, self.MAX_SESSIONS)
        except SessionStoreError:
            raise
        except sqlite3.Error as exc:
            raise SessionStoreError("Session history count could not be read safely.") from exc

    def add_session(self, record: SessionRecord) -> None:
        try:
            with closing(self._connect()) as connection:
                with connection:
                    connection.execute("BEGIN IMMEDIATE")
                    connection.execute(
                        "INSERT INTO sessions (payload) VALUES (?)",
                        (record.model_dump_json(),),
                    )
                    connection.execute(
                        "DELETE FROM sessions WHERE id NOT IN ("
                        "SELECT id FROM sessions ORDER BY id DESC LIMIT ?)",
                        (self.MAX_SESSIONS,),
                    )
        except SessionStoreError:
            raise
        except (sqlite3.Error, ValueError, TypeError) as exc:
            raise SessionStoreError("Session history could not be saved safely.") from exc
