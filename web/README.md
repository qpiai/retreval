# ReTreVal Web UI

An interactive chat front-end for the ReTreVal reasoning-tree agent. The left
pane is a chat; the right pane shows the **reasoning tree** (React Flow) growing
live and a streaming **log/trace** panel, plus the validated answer, score, and
the tools the agent used.

```
┌────────────────────┬───────────────────────────────┐
│                    │   🌳 Reasoning tree (React Flow)│
│   💬 Chat          ├───────────────────────────────┤
│                    │   📜 Live logs / trace         │
└────────────────────┴───────────────────────────────┘
```

## Architecture

```
web/ (Next.js)  ──EventSource (SSE)──►  server/app.py (FastAPI)  ──►  LangGraphAgent
```

The agent is Python (LangGraph), so a thin FastAPI bridge streams its progress
to the browser as Server-Sent Events: `status`, `log`, `step`, `tree`, `final`.

## Run (dev)

**1 — backend** (from `retreval_oss/`):

```bash
. .venv/bin/activate
pip install -r server/requirements.txt        # fastapi + uvicorn (one-time)

# Demo without any API key — streams a canned reasoning run:
RETREVAL_MOCK=1 uvicorn server.app:app --port 7373

# Real agent — put a key in retreval_oss/.env first (LLM_PROVIDER, GEMINI_API_KEY…):
uvicorn server.app:app --port 7373
```

**2 — frontend** (from `retreval_oss/web/`):

```bash
npm install
npm run dev        # → http://localhost:7575
```

The UI talks to `NEXT_PUBLIC_API_BASE` (default `http://localhost:7373`, see
`.env.local`).

## Build (static)

The UI is a pure client-side SPA, so `next build` emits a fully static site to
`out/` — serve it with any static file server (no Node runtime needed):

```bash
npm run build
cd out && python3 -m http.server 7575
```

## Notes

- **Ports:** UI on **7575**, backend on **7373** (both memorable and
  Chromium-safe — note 6666 is blocked by browsers as an unsafe port).
- Provider / iterations / memory are selectable in the header; they map to the
  agent's `--provider`, `--iterations`, and `--no-memory` options.
- **Mock mode** (`RETREVAL_MOCK=1`) needs no model or key and powers the demo
  capture in [`scripts/`](../scripts/).
