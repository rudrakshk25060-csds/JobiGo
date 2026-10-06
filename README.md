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
Alternative: AIProvider -> HuggingFaceGemmaProvider -> HF Inference Providers -> google/gemma-3-4b-it
```

`AIProvider` defines mission generation, session evaluation, next-session recommendations, and readiness checks. The existing `GemmaProvider` continues to call Ollama's local `/api/chat` endpoint. `OpenRouterGemmaProvider` and `HuggingFaceGemmaProvider` use their OpenAI-compatible chat-completions endpoints and both require the exact `google/gemma-3-4b-it` model. All providers share the same mission, review, progression, and validation logic. Hosted providers request strict JSON Schema output, then the response is parsed and validated by Pydantic; the mission is checked against the selected time, goal, and equipment, with one correction attempt for a mismatch. Invalid output or unavailable AI produces an explicit API error rather than a fabricated mission. Hugging Face documents `json_schema` support on its router, while noting compatibility depends on the provider/model combination; JobiGo fails visibly if the selected upstream route rejects it or returns invalid output.

The provider abstraction keeps local development independent of the inference transport. Gemma remains the AI model in each path; OpenRouter and Hugging Face are managed inference layers, not replacement models. Hugging Face's live model metadata currently lists DeepInfra and Featherless AI for this Gemma model; availability can change. Hugging Face currently documents $0.10/month in inference credits for free users (subject to change); usage beyond available credits requires purchasing credits. This may enable a no-purchase demo within the allowance, not unlimited or guaranteed-free usage. ElevenLabs remains a separate optional voice integration.

Completed sessions are stored in SQLite at `DATA_FILE`, with at most 100 records retained. SQLite transactions, WAL journaling, and a busy timeout serialize concurrent writes while the database file remains available. If an older local `sessions.json` exists next to the configured database path, JobiGo imports its last 100 records once and leaves the JSON file as a backup. Local history remains on your computer. Render Free uses an ephemeral filesystem, so session history can disappear after a restart, redeploy, or spin-down; it is not durable storage.

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

The local default is `AI_PROVIDER=ollama`, `OLLAMA_MODEL=gemma3:4b`, `OLLAMA_BASE_URL=http://localhost:11434`, and `GEMMA_TIMEOUT_SECONDS=90`. Older `.env` files using `GEMMA_MODEL` and `GEMMA_BASE_URL` remain supported. Ollama is not removed or required when a hosted provider is selected.

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

## Hugging Face hosted Gemma alternative

Select Hugging Face without changing the application flow:

```dotenv
AI_PROVIDER=huggingface
HF_TOKEN=<server-side token with Inference Providers permission>
HF_MODEL=google/gemma-3-4b-it
HF_BASE_URL=https://router.huggingface.co/v1
HF_TIMEOUT_SECONDS=90
```

The token is sent only by FastAPI as a Bearer credential. `/ready` checks Hugging Face model metadata for at least one currently live provider; it does not perform inference or confirm quota. Hugging Face's router accepts strict JSON Schema requests, but compatibility depends on the selected provider/model. JobiGo keeps the strict schema request and its existing Pydantic/mission checks; unsupported schemas or invalid responses fail explicitly rather than producing invented content. A live provider listing does not guarantee quota, zero cost, or successful generation. Check current Hugging Face pricing and account allowance before use.

## Configuration and health checks

Copy `.env.example` to `.env`. Supported settings:

| Variable | Default | Purpose |
| --- | --- | --- |
| `APP_ENV` | `development` | `development` enables FastAPI API docs; `production` disables them. |
| `CORS_ALLOWED_ORIGINS` | empty | Optional comma-separated exact origins. Empty means no cross-origin browser access is configured. Wildcard origins are rejected. |
| `OLLAMA_MODEL` | `gemma3:4b` | Local Ollama model tag (`GEMMA_MODEL` remains a compatibility alias). |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local Ollama service base URL (`GEMMA_BASE_URL` remains a compatibility alias). |
| `GEMMA_TIMEOUT_SECONDS` | `90` | Per-request model timeout. |
| `AI_PROVIDER` | `ollama` | Selects `ollama`, `openrouter`, or `huggingface` for Gemma inference. |
| `OPENROUTER_API_KEY` | empty | Required server-side key when `AI_PROVIDER=openrouter`; never place in frontend code. |
| `OPENROUTER_MODEL` | `google/gemma-3-4b-it` | Exact production model. Other values are rejected when OpenRouter is selected. |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter-compatible API base URL. |
| `OPENROUTER_TIMEOUT_SECONDS` | `90` | Per-request OpenRouter timeout. |
| `HF_TOKEN` | empty | Required server-side token when `AI_PROVIDER=huggingface`; never place in frontend code. |
| `HF_MODEL` | `google/gemma-3-4b-it` | Exact Gemma model used by Hugging Face Inference Providers. |
| `HF_BASE_URL` | `https://router.huggingface.co/v1` | Hugging Face OpenAI-compatible router base URL. |
| `HF_TIMEOUT_SECONDS` | `90` | Per-request Hugging Face timeout. |
| `ELEVENLABS_API_KEY` | empty | Optional server-side ElevenLabs credential; never expose it in frontend code. |
| `ELEVENLABS_VOICE_ID` | empty | Optional ElevenLabs voice identifier. Voice is unavailable until both voice variables are configured. |
| `ELEVENLABS_MODEL_ID` | `eleven_multilingual_v2` | ElevenLabs text-to-speech model. |
| `DATA_FILE` | `<project>/data/sessions.sqlite3` | SQLite database path. Local paths are relative to the process working directory. The Render Free Blueprint uses `./data/sessions.sqlite3` on its ephemeral service filesystem. |
| `DEMO_ACCESS_USERNAME` | `jobigo` | Username for the shared HTTP Basic private-demo credential. |
| `DEMO_ACCESS_TOKEN` | empty locally | Shared password/token for protected write operations; required when `APP_ENV=production`. Set it only in Render's secret settings for deployment. |

`GET /health` is a liveness check for the FastAPI process only. `GET /ready` checks the selected provider: Ollama's local model list, OpenRouter's endpoint catalog, or Hugging Face model metadata for a live inference provider. These checks do not generate text, spend inference credits, or confirm account balance/quota. Readiness returns `503` if the selected provider is not ready. Provider unavailability does not block application startup.

## API

- `GET /` — frontend
- `GET /health` — application liveness
- `GET /ready` — configured Gemma provider readiness
- `GET /api/sessions` — only `{"session_count": N}`; it does not return notes, records, or identifiers
- `POST /api/missions` — protected by the shared demo credential; validate setup and generate a mission
- `POST /api/missions/{mission_id}/voice` — protected by the shared demo credential; accept the current mission and return `audio/mpeg`; ElevenLabs is used only when requested and audio is not saved
- `POST /api/sessions` — protected by the shared demo credential; evaluate a report, recommend the next session, save it, and return the review to the authorized caller

## Privacy, security, and current limitations

- The local Ollama path keeps prompts and session history on the machine running JobiGo, provided `OLLAMA_BASE_URL` remains local. OpenRouter sends current mission data and submitted session notes to OpenRouter; the Hugging Face path sends them to Hugging Face Inference Providers and the selected inference provider. Review each service's privacy and data-retention terms before using real personal data.
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
- **Render:** deployed from `render.yaml` as one Singapore Free Python web service. It specifies one Singapore Free Python web service and one Uvicorn worker, with no disk or database. Production uses OpenRouter and `google/gemma-3-4b-it`; Render does not need or use local Ollama. SQLite writes to `./data/sessions.sqlite3` on Render's ephemeral filesystem. Render may spin the service down after 15 minutes without incoming traffic, and local filesystem changes are lost on spin-down, restart, or redeploy. The free deployment is intended for a controlled hackathon demonstration, not production SaaS or durable session history. See [Render Free services](https://render.com/docs/free) and [Render web services](https://render.com/docs/web-services).

`APP_ENV=production` disables FastAPI's interactive docs and requires `DEMO_ACCESS_TOKEN`. The Render Blueprint configures the Free web service and remote Gemma provider.

### Current Render deployment

The live service is [jobigo-demo.onrender.com](https://jobigo-demo.onrender.com/). Render reads this repository’s checked-in `render.yaml`.
The service uses the **Free** plan and requires `OPENROUTER_API_KEY` and `DEMO_ACCESS_TOKEN` in Render's secret settings. Optional voice requires `ELEVENLABS_API_KEY` and `ELEVENLABS_VOICE_ID`. Keep `CORS_ALLOWED_ORIGINS` unset for the bundled same-origin frontend; if the frontend is hosted separately, list only its exact HTTPS origin. Check `/health` and `/ready` after deploy. `/ready` confirms OpenRouter connectivity and model listing; it does not make a generation request. Session history uses `./data/sessions.sqlite3` on Render Free's ephemeral filesystem and may be lost during spin-down, restart, or redeploy.

Render requires the web service to listen on `0.0.0.0` and its `PORT` environment variable; the Blueprint start command configures both. Free services spin down after 15 minutes of inactivity and can take about a minute to start again. Filesystem changes, including the SQLite history, are lost after spin-down, restart, or redeploy. See [Render Free services](https://render.com/docs/free), [Render web services](https://render.com/docs/web-services), and [health checks](https://render.com/docs/health-checks).

## Tests

The `unittest` suite covers request validation, structured-output parsing, mission constraints, API flows with mock AI and voice providers, voice script completeness and failure handling, safe progress output, readiness/liveness responses, SQLite persistence/migration/concurrency, demo access control, CORS, and progression helpers. Ollama, OpenRouter, and ElevenLabs adapter tests use in-process mocked HTTP transports; tests do not call live AI services or spend API credits:

```bash
python -m unittest discover -s tests -v
```

## License

JobiGo is released under the MIT License. See [LICENSE](LICENSE).
