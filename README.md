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
  ├── POST /api/missions ─> FastAPI validation -> AIProvider -> selected Gemma provider
  ├── POST /api/missions/{mission_id}/voice -> VoiceProvider -> ElevenLabs streaming TTS -> MP3
  └── POST /api/sessions -> FastAPI validation -> Gemma review + next-session recommendation -> SQLite history

Local:      AIProvider -> GemmaProvider -> Ollama -> Gemma 3 4B
Production: AIProvider -> OpenRouterGemmaProvider -> OpenRouter -> google/gemma-3-4b-it
```

`AIProvider` defines mission generation, session evaluation, next-session recommendations, and readiness checks. The existing `GemmaProvider` continues to call Ollama's local `/api/chat` endpoint. `OpenRouterGemmaProvider` uses OpenRouter's chat-completions endpoint and requires the exact `google/gemma-3-4b-it` model. Both use the same mission, review, progression, and validation logic. Structured JSON Schema output is requested, then parsed and validated by Pydantic; the mission is checked against the selected time, goal, and equipment, with one correction attempt for a mismatch. Invalid output or unavailable AI produces an explicit API error rather than a fabricated mission.

The provider abstraction keeps local development independent of the production inference transport. Gemma remains the AI model in both paths; OpenRouter is a managed inference layer, not a replacement model. ElevenLabs remains a separate optional voice integration.

Completed sessions are stored in SQLite at `DATA_FILE`, with at most 100 records retained. SQLite transactions, WAL journaling, and a busy timeout serialize concurrent writes and recover committed data after process restarts. If an older local `sessions.json` exists next to the configured database path, JobiGo imports its last 100 records once and leaves the JSON file as a backup. Keep the Render service at one instance: its persistent disk is attached to that instance and cannot be shared or used for multi-instance scaling.

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

The local default is `AI_PROVIDER=ollama`, `OLLAMA_MODEL=gemma3:4b`, `OLLAMA_BASE_URL=http://localhost:11434`, and `GEMMA_TIMEOUT_SECONDS=90`. Older `.env` files using `GEMMA_MODEL` and `GEMMA_BASE_URL` remain supported. Ollama is not removed or required when `AI_PROVIDER=openrouter`.

## OpenRouter production inference

Set these server-side environment variables for the OpenRouter path:

```dotenv
APP_ENV=production
AI_PROVIDER=openrouter
OPENROUTER_API_KEY=<secret stored in the hosting platform's secret settings>
OPENROUTER_MODEL=google/gemma-3-4b-it
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_TIMEOUT_SECONDS=90
```

The selected model is validated against the exact required ID; a different configured model fails configuration instead of silently substituting another model. The API key is read only by FastAPI, sent as a server-side Bearer header, and is never included in browser JavaScript, logs, or API responses. OpenRouter requests strict JSON Schema output and asks the router to select only endpoints that support the requested parameters. See [OpenRouter structured outputs](https://openrouter.ai/docs/guides/features/structured-outputs).

When `AI_PROVIDER=openrouter`, the current mission setup is sent for mission generation. For a session review, the current mission, submitted match statistics, and the player's optional notes are sent because the notes can inform the review. Progression context is limited to recent coaching-relevant fields; prior notes, session IDs, mission titles, and long prior plan text are excluded. Relevant JobiGo data therefore leaves the hosting environment and is processed by OpenRouter and the upstream inference provider(s) it routes to. Review their current privacy/data-retention terms before using real personal data. Production requires a valid OpenRouter API key and network access; no real API call is used by the automated test suite.

## Configuration and health checks

Copy `.env.example` to `.env`. Supported settings:

| Variable | Default | Purpose |
| --- | --- | --- |
| `APP_ENV` | `development` | `development` enables FastAPI API docs; `production` disables them. |
| `CORS_ALLOWED_ORIGINS` | empty | Optional comma-separated exact origins. Empty means no cross-origin browser access is configured. Wildcard origins are rejected. |
| `OLLAMA_MODEL` | `gemma3:4b` | Local Ollama model tag (`GEMMA_MODEL` remains a compatibility alias). |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local Ollama service base URL (`GEMMA_BASE_URL` remains a compatibility alias). |
| `GEMMA_TIMEOUT_SECONDS` | `90` | Per-request model timeout. |
| `AI_PROVIDER` | `ollama` | Selects `ollama` for local inference or `openrouter` for managed inference. |
| `OPENROUTER_API_KEY` | empty | Required server-side key when `AI_PROVIDER=openrouter`; never place in frontend code. |
| `OPENROUTER_MODEL` | `google/gemma-3-4b-it` | Exact production model. Other values are rejected when OpenRouter is selected. |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter-compatible API base URL. |
| `OPENROUTER_TIMEOUT_SECONDS` | `90` | Per-request OpenRouter timeout. |
| `ELEVENLABS_API_KEY` | empty | Optional server-side ElevenLabs credential; never expose it in frontend code. |
| `ELEVENLABS_VOICE_ID` | empty | Optional ElevenLabs voice identifier. Voice is unavailable until both voice variables are configured. |
| `ELEVENLABS_MODEL_ID` | `eleven_multilingual_v2` | ElevenLabs text-to-speech model. |
| `DATA_FILE` | `<project>/data/sessions.sqlite3` | SQLite database path. Local paths are relative to the process working directory; Render uses `/var/data/sessions.sqlite3` on its disk. |
| `DEMO_ACCESS_USERNAME` | `jobigo` | Username for the shared HTTP Basic private-demo credential. |
| `DEMO_ACCESS_TOKEN` | empty locally | Shared password/token for protected write operations; required when `APP_ENV=production`. Set it only in Render's secret settings for deployment. |

`GET /health` is a liveness check for the FastAPI process only. `GET /ready` checks the selected provider: for Ollama it checks the local model list, and for OpenRouter it makes a lightweight authenticated model-catalog request and checks that the exact required model is listed. It does not generate text, spend inference credits, or confirm account balance or model-generation quota. It returns `503` if the selected provider is not ready. Provider unavailability does not block application startup.

## API

- `GET /` — frontend
- `GET /health` — application liveness
- `GET /ready` — configured Gemma provider readiness
- `GET /api/sessions` — only `{"session_count": N}`; it does not return notes, records, or identifiers
- `POST /api/missions` — protected by the shared demo credential; validate setup and generate a mission
- `POST /api/missions/{mission_id}/voice` — protected by the shared demo credential; accept the current mission and return `audio/mpeg`; ElevenLabs is used only when requested and audio is not saved
- `POST /api/sessions` — protected by the shared demo credential; evaluate a report, recommend the next session, save it, and return the review to the authorized caller

## Privacy, security, and current limitations

- The local Ollama path keeps prompts and session history on the machine running JobiGo, provided `OLLAMA_BASE_URL` remains local. The OpenRouter path sends current mission data and submitted session notes to OpenRouter for inference, as described above.
- Production uses one shared HTTP Basic credential for mission generation, voice generation, and session submission. The username is `DEMO_ACCESS_USERNAME` and the password/token is `DEMO_ACCESS_TOKEN`; the browser requests it through its native credential prompt. The token is never embedded in JavaScript. Share it privately with demo participants and use HTTPS.
- This is a small private-demo boundary, not individual accounts: everyone uses the same reusable credential, there is no per-user identity, lockout, or rate limiter, and browser software may remember the credential. Rotate it after the demo. Keep the service controlled rather than treating this as public SaaS security.
- `/health`, `/ready`, and count-only `GET /api/sessions` remain public. The count endpoint does not reveal session notes or IDs. CORS is disabled when `CORS_ALLOWED_ORIGINS` is empty; when enabled, exact origins are required and wildcard origins fail configuration. CORS controls which browser origins may read responses; it does not authenticate requests.
- SQLite is appropriate here because the demo retains only 100 small session records and uses one Render instance/one Uvicorn worker. SQLite transactions and WAL reduce partial-write and concurrent-writer risks. A Render disk is single-instance storage, so this architecture does not support horizontal scaling or zero-downtime deploys. Keep backups before changing or removing the disk.
- Google Fonts are loaded from Google; system font fallbacks are present if they are unavailable.
- AI-generated drills are checked for structure and selected constraints, but automated checks cannot guarantee every coaching instruction is safe or suitable. Use a clear playing area and stop if anything feels unsafe.
- Tests use a mock provider and do not require Ollama. A live model is required to generate and review real missions.
- The pinned Starlette `TestClient` emits a non-fatal deprecation warning with the current HTTPX test dependency; the suite passes, but this should be revisited when updating the test stack.

## Integrations and deployment status

- **ElevenLabs:** optional spoken mission briefings are integrated through `VoiceProvider`. Add `ELEVENLABS_API_KEY` and `ELEVENLABS_VOICE_ID` to the untracked local `.env` to enable **🎧 COACH ME**. No live credentials are included in the repository; without them, the written mission remains usable. The feature uses the ElevenLabs streaming text-to-speech API and requires network access.
- **Backboard:** not integrated. Recent structured session history is passed directly to Gemma; no hosted memory service is used.
- **Privacy:** OpenRouter receives the current mission setup, the mission being reviewed, submitted statistics and notes, and a reduced coaching-history summary. Prior notes, session IDs, mission titles, and long plan text are excluded from history context. ElevenLabs receives only the short spoken mission script; no session notes are included in that script.
- **Render:** prepared in `render.yaml`, but not deployed. It specifies a Singapore Starter web service, one Uvicorn worker, a 1 GB persistent disk at `/var/data`, and `DATA_FILE=/var/data/sessions.sqlite3`. Render preserves files only under the disk mount; a persistent disk requires a paid service plan and prevents scaling to multiple instances. For local Ollama mode, Render cannot reach a developer machine's `localhost:11434`.

`APP_ENV=production` disables FastAPI's interactive docs and requires `DEMO_ACCESS_TOKEN`. The Render Blueprint supplies the durable disk and remote Gemma provider configuration; it does not create a Render service until you explicitly create/sync the Blueprint.

### Render deployment steps (not yet performed)

1. Push this repository to the Git provider you want Render to read, then create a new **Blueprint** in Render and connect that repository. Render reads the checked-in `render.yaml`.
2. Choose the paid Starter plan shown in the Blueprint; the Free plan cannot attach the persistent disk required by this configuration.
3. In the Blueprint setup prompt/dashboard, set `OPENROUTER_API_KEY` and a long random `DEMO_ACCESS_TOKEN`. Keep both as secrets and never put them in Git or frontend code. Give demo participants the username `jobigo` and token privately.
4. Keep `CORS_ALLOWED_ORIGINS` unset for the bundled same-origin frontend. If you later serve the frontend elsewhere, set a comma-separated list of exact `https://` origins.
5. After deployment, check `/health` and `/ready`. `/ready` verifies OpenRouter connectivity and model listing only; make one controlled generation request using the private demo credential to verify actual inference.
6. Confirm a session survives a service restart before inviting participants. The mounted disk path is `/var/data`; the SQLite file is `/var/data/sessions.sqlite3`.

Render requires the web service to listen on `0.0.0.0` and its `PORT` environment variable; the Blueprint start command configures both. Render docs note disks are attached to one service instance and stop zero-downtime instance swaps, so deploys can have a short interruption. See [Render web services](https://render.com/docs/web-services), [persistent disks](https://render.com/docs/disks), and [health checks](https://render.com/docs/health-checks).

## Tests

The `unittest` suite covers request validation, structured-output parsing, mission constraints, API flows with mock AI and voice providers, voice script completeness and failure handling, safe progress output, readiness/liveness responses, SQLite persistence/migration/concurrency, demo access control, CORS, and progression helpers. Ollama, OpenRouter, and ElevenLabs adapter tests use in-process mocked HTTP transports; tests do not call live AI services or spend API credits:

```bash
python -m unittest discover -s tests -v
```

## License

JobiGo is released under the MIT License. See [LICENSE](LICENSE).
