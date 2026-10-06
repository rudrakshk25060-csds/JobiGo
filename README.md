# JobiGo

**An AI football coach that gets Jobi off the screen and onto the field.**

JobiGo is a mobile-first outdoor football coach built for Jobi Anand. Its loop is **plan → go outside → play → return → report → improve**. Finishing is the default goal because Jobi said it is the skill he especially wants to improve.

## What it does

1. Choose available time, level, training goal, equipment, and an optional setting.
2. Ask Gemma for a structured mission with warm-up, drills, challenge, cooldown, and safety guidance.
3. Put the phone away with the **PHONE DOWN. GO PLAY.** screen.
4. Return to record attempts, goals, completed drills, perceived difficulty, and optional notes.
5. Receive a Gemma review and a next-session recommendation. Completed records are kept locally and recent history informs future coaching.

The UI is intentionally focused on the outdoor session; it is not a statistics dashboard or chatbot.

## Architecture

```text
Browser (HTML / CSS / JavaScript)
  ├── GET /api/sessions ──> count-only progress response
  ├── POST /api/missions ─> FastAPI validation -> AIProvider -> GemmaProvider -> Ollama -> Gemma 3 4B
  ├── POST /api/missions/{mission_id}/voice -> VoiceProvider -> ElevenLabs streaming TTS -> MP3
  └── POST /api/sessions -> FastAPI validation -> Gemma review + next-session recommendation -> local JSON history
```

`AIProvider` defines mission generation, session evaluation, next-session recommendations, and readiness checks. `GemmaProvider` calls Ollama's local `/api/chat` endpoint, supplies a Pydantic JSON Schema through Ollama's `format` option, and validates responses. Mission constraints are checked against the selected time, goal, and equipment. A single correction attempt is made for a mismatch; invalid output or unavailable AI produces an explicit API error rather than mock output.

Completed sessions are stored as JSON at `DATA_FILE`, with at most 100 records retained. File replacement is atomic against partial writes, but the store is intended for a local, single-process MVP; concurrent writes and multiple server workers can lose updates.

Text-to-speech is a separate, optional `VoiceProvider` integration. `ElevenLabsVoiceProvider` receives a concise script built from the current mission and returns MP3 audio on demand. The API key stays on the server; audio is not persisted. Missing credentials, provider errors, or network failures leave the written mission and **GO OUTSIDE** action available. Gemma remains the core coach for mission creation and session review.

## Gemma and Ollama setup

Requirements: Python 3.10+ and [Ollama](https://ollama.com/).

1. Install Ollama and download the model:

   ```bash
   ollama pull gemma3:4b
   ```

2. Start Ollama. The Ollama desktop app normally starts the local service; otherwise run `ollama serve` in a separate terminal.
3. From the JobiGo directory, create the environment and install the pinned dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate       # Windows: .venv\Scripts\activate
   python -m pip install -r requirements.txt
   cp .env.example .env            # Windows: copy .env.example .env
   ```

4. Start the app:

   ```bash
   uvicorn app.main:app --reload
   ```

5. Open <http://127.0.0.1:8000>.

The default model configuration is `GEMMA_MODEL=gemma3:4b`, `GEMMA_BASE_URL=http://localhost:11434`, and `GEMMA_TIMEOUT_SECONDS=90`. `GEMMA_BASE_URL` may point to another Ollama-compatible endpoint; this app does not currently configure authentication headers for a protected remote model service. If a remote endpoint is used, mission/report data is sent there.

## Configuration and health checks

Copy `.env.example` to `.env`. Supported settings:

| Variable | Default | Purpose |
| --- | --- | --- |
| `APP_ENV` | `development` | `development` enables FastAPI API docs; `production` disables them. |
| `CORS_ALLOWED_ORIGINS` | empty | Optional comma-separated exact origins. Empty means no cross-origin browser access is configured. Wildcard origins are rejected. |
| `GEMMA_MODEL` | `gemma3:4b` | Ollama model tag. |
| `GEMMA_BASE_URL` | `http://localhost:11434` | Ollama-compatible service base URL. |
| `GEMMA_TIMEOUT_SECONDS` | `90` | Per-request model timeout. |
| `ELEVENLABS_API_KEY` | empty | Optional server-side ElevenLabs credential; never expose it in frontend code. |
| `ELEVENLABS_VOICE_ID` | empty | Optional ElevenLabs voice identifier. Voice is unavailable until both voice variables are configured. |
| `ELEVENLABS_MODEL_ID` | `eleven_multilingual_v2` | ElevenLabs text-to-speech model. |
| `DATA_FILE` | `<project>/data/sessions.json` | Local history path. The `.env.example` value is relative to the process working directory. |

`GET /health` is a liveness check for the FastAPI process only. `GET /ready` separately checks whether Ollama responds and the configured model is installed; it returns `503` if not. Model availability does not block application startup.

## API

- `GET /` — frontend
- `GET /health` — application liveness
- `GET /ready` — configured Gemma/Ollama readiness
- `GET /api/sessions` — only `{"session_count": N}`; it does not return notes, records, or identifiers
- `POST /api/missions` — validate setup and generate a mission
- `POST /api/missions/{mission_id}/voice` — accept the current mission and return `audio/mpeg`; this endpoint uses ElevenLabs only when requested and does not save audio
- `POST /api/sessions` — evaluate a submitted report, recommend the next session, save it, and return the review to the submitting caller

## Privacy, security, and current limitations

- The local default keeps prompts and session history on the machine running JobiGo, provided `GEMMA_BASE_URL` remains local.
- There are no accounts or authentication. `POST /api/sessions` accepts submissions without identity or authorization and returns a review to the caller. Do not expose the service publicly until access control, abuse controls, and a privacy plan are added.
- `GET /api/sessions` exposes only a count, but this does not protect the write endpoint or make public deployment safe.
- The local JSON store is not a shared or concurrency-safe production database.
- Google Fonts are loaded from Google; system font fallbacks are present if they are unavailable.
- AI-generated drills are checked for structure and selected constraints, but automated checks cannot guarantee every coaching instruction is safe or suitable. Use a clear playing area and stop if anything feels unsafe.
- Tests use a mock provider and do not require Ollama. A live model is required to generate and review real missions.
- The pinned Starlette `TestClient` emits a non-fatal deprecation warning with the current HTTPX test dependency; the suite passes, but this should be revisited when updating the test stack.

## Integrations and deployment status

- **ElevenLabs:** optional spoken mission briefings are integrated through `VoiceProvider`. Add `ELEVENLABS_API_KEY` and `ELEVENLABS_VOICE_ID` to the untracked local `.env` to enable **🎧 COACH ME**. No live credentials are included in the repository; without them, the written mission remains usable. The feature uses the ElevenLabs streaming text-to-speech API and requires network access.
- **Backboard:** not integrated. Recent structured session history is passed directly to Gemma; no hosted memory service is used.
- **Render:** not deployed or deployment-complete. Render cannot reach a developer machine's `localhost:11434`. A deployment needs reachable Gemma inference, a persistent database or disk for session history, a production start command using Render's `PORT` and `0.0.0.0`, and access control before public use. A JSON file on an ephemeral filesystem would not survive restarts or deploys.

`APP_ENV=production` disables FastAPI's interactive docs; it does **not** add authentication, durable storage, remote-model access, or other production protections. For same-origin hosting, leave `CORS_ALLOWED_ORIGINS` empty. If hosting the frontend separately, set only its exact origin(s).

## Tests

The `unittest` suite covers request validation, structured-output parsing, mission constraints, API flows with mock AI and voice providers, voice script completeness and failure handling, safe progress output, readiness/liveness responses, session storage, and progression helpers. ElevenLabs HTTP success, API error, and network error behavior are tested with an in-process HTTP transport; tests never call the live service:

```bash
python -m unittest discover -s tests -v
```

## License

JobiGo is released under the MIT License. See [LICENSE](LICENSE).
