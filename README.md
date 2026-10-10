# JobiGo

**An AI football coach that gets Jobi off the screen and onto the field.**

Built for Hacktoberfest 2026 — Week 1 **“Touch Grass”** Challenge.

- **GitHub Repository:** [https://github.com/rudrakshk25060-csds/JobiGo](https://github.com/rudrakshk25060-csds/JobiGo)
- **Live Demo:** [https://jobigo-demo.onrender.com/](https://jobigo-demo.onrender.com/)

---

## The Problem & The Solution

Most AI applications are designed to maximize screen time, trapping users in endless chats, feeds, and dashboards.

**JobiGo does the exact opposite.** Built specifically for Jobi Anand, JobiGo uses AI to make screen time as brief as possible so the player can get outside and play football. Finishing is the default focus because Jobi specifically wanted to sharpen his goal-scoring instincts.

### The Core Loop
$$\text{PLAN} \longrightarrow \text{GO OUTSIDE} \longrightarrow \text{PLAY} \longrightarrow \text{RETURN} \longrightarrow \text{REPORT} \longrightarrow \text{IMPROVE}$$

1. **Plan:** Choose available time (20–90 min), skill level, training goal, and available equipment.
2. **Go Outside:** Receive a tailored mission, see the **PHONE DOWN. GO PLAY.** screen, and put the device away.
3. **Play:** Train outdoors on the pitch, park, turf, or open space.
4. **Return:** Record total attempts, successful attempts, goals, completed drills, difficulty rating, and brief personal notes.
5. **Report & Improve:** Gemma 3 4B evaluates the session, identifies key strengths and weaknesses, and recommends a progressive next session.

---

## Where Gemma 3 4B is Used

Google's **Gemma 3 4B** is the core intelligence driving JobiGo across three critical stages:

1. **Mission Generation:** Generates safe, achievable outdoor training plans with warm-ups, progressive drills, measurable challenges, cool-downs, motivation, and safety notes matched strictly to available time, equipment, and level.
2. **Structured JSON Output & Schema Validation:** Enforces strict Pydantic schemas for all drills, timing calculations, and equipment rules. If a draft exceeds duration limits or misinterprets equipment (e.g. asking for cones during a football-only session), JobiGo runs an automated correction retry.
3. **Performance Review & Progression:** Analyzes match results and qualitative notes, generates actionable coaching reflections, and recommends the next session's focus and duration based on reported difficulty.

---

## Architecture

```text
Browser (HTML / CSS / JavaScript)
  ├── GET /api/sessions ──> count-only progress response
  ├── POST /api/missions ─> FastAPI validation -> AIProvider -> selected Gemma provider
  ├── POST /api/missions/{mission_id}/voice -> VoiceProvider -> ElevenLabs streaming TTS -> MP3
  └── POST /api/sessions -> FastAPI validation -> Gemma review + next-session recommendation -> SQLite history

Local:       AIProvider -> GemmaProvider -> Ollama -> Gemma 3 4B (100% Free / ₹0)
Production:  AIProvider -> OpenRouterGemmaProvider -> OpenRouter -> google/gemma-3-4b-it
Alternative: AIProvider -> HuggingFaceGemmaProvider -> HF Inference Providers -> google/gemma-3-4b-it
```

- **Backend:** FastAPI (Python 3.10+) with Pydantic v2 strict models.
- **Frontend:** Lightweight, mobile-first responsive HTML5/CSS3/ES6 JavaScript (zero heavy client-side frameworks).
- **Storage:** Local SQLite database with WAL mode and concurrent transaction serialization.
- **AI Abstraction (`AIProvider`):** Unified interface for Ollama, OpenRouter, and Hugging Face providers.
- **Voice Briefing (`VoiceProvider`):** Optional ElevenLabs text-to-speech coaching briefing.

---

## Local Development (Ollama + Gemma 3 4B)

Local development runs 100% free with zero API keys or external dependencies:

### 1. Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/)

### 2. Pull Gemma 3 4B Model
```bash
ollama pull gemma3:4b
```

### 3. Clone & Install
```bash
git clone https://github.com/rudrakshk25060-csds/JobiGo.git
cd JobiGo
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # Windows: copy .env.example .env
```

### 4. Run Development Server
```bash
uvicorn app.main:app --reload --port 8000
```
Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

---

## Cloud Deployment (Render)

JobiGo is configured for deployment as a web service via [render.yaml](render.yaml):

- **Live URL:** [https://jobigo-demo.onrender.com/](https://jobigo-demo.onrender.com/)
- **Hosting:** Render Free Python Web Service (Singapore region, 1 Uvicorn worker).
- **Storage:** Ephemeral SQLite storage at `./data/sessions.sqlite3`.
- **Health Check:** `GET /health` validates application liveness.
- **Readiness Check:** `GET /ready` validates model and provider connectivity.
- **Inference Note:** Render hosts the FastAPI web application and frontend. External hosted inference requires configuring `OPENROUTER_API_KEY` or `HF_TOKEN` in Render's secret settings. Local Ollama runs completely offline and free on your development machine.

---

## Configuration Reference

Settings can be customized via `.env`:

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `APP_ENV` | `development` | `development` enables FastAPI docs; `production` disables docs for clean public deployment. |
| `CORS_ALLOWED_ORIGINS` | empty | Comma-separated allowed browser origins (wildcards rejected). |
| `DATA_FILE` | `./data/sessions.sqlite3` | SQLite database file path. |
| `AI_PROVIDER` | `ollama` | Provider selection: `ollama`, `openrouter`, or `huggingface`. |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local Ollama service base URL. |
| `OLLAMA_MODEL` | `gemma3:4b` | Ollama model identifier. |
| `GEMMA_TIMEOUT_SECONDS` | `90` | Timeout in seconds for AI completions. |
| `OPENROUTER_API_KEY` | empty | Server-side key for OpenRouter hosted inference. |
| `OPENROUTER_MODEL` | `google/gemma-3-4b-it` | Exact model ID for OpenRouter. |
| `HF_TOKEN` | empty | Server-side Hugging Face Inference Providers token. |
| `HF_MODEL` | `google/gemma-3-4b-it` | Exact model ID for Hugging Face Router. |
| `ELEVENLABS_API_KEY` | empty | Optional ElevenLabs API key for spoken briefings. |
| `ELEVENLABS_VOICE_ID` | empty | Optional ElevenLabs voice identifier. |
| `DEMO_ACCESS_USERNAME` | `jobigo` | Optional demo username. |
| `DEMO_ACCESS_TOKEN` | empty | Optional demo token (public demo does not prompt for credentials). |

---

## Privacy, Security & Data Safety

- **Privacy-Safe Progress:** The public progress endpoint (`GET /api/sessions`) returns only `{"session_count": N}`. It never exposes player notes, session IDs, timestamps, or drill details.
- **No Secret Exposure:** All credentials (`HF_TOKEN`, `OPENROUTER_API_KEY`, `ELEVENLABS_API_KEY`, `DEMO_ACCESS_TOKEN`) stay strictly on the server and are scrubbed from logs and API error responses.
- **Sanitized Progression Context:** When history is passed to Gemma for session progression, only high-level statistical summaries (goals, completion rates, weakness areas) are forwarded; personal notes, session IDs, and raw text are excluded.
- **Deterministic Safety:** Every generated mission enforces a standard physical safety note and warm-up/cool-down requirements.

---

## Automated Test Suite

JobiGo includes a comprehensive test suite covering schema validation, error boundaries, session concurrency, CORS, demo auth, and provider adapters:

```bash
python -m unittest discover -s tests -v
```

```text
Ran 51 tests in 0.12s — OK
```

---

## License

JobiGo is open-source software licensed under the [MIT License](LICENSE).
