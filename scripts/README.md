# Scripts

Demo + test helpers (not part of the importable package).

| script | what it does |
|---|---|
| `smoke_ui.py` | Headless-browser smoke test: loads the UI, sends a problem, asserts the reasoning tree grows and a validated answer appears. |
| `capture_demo.py` | Drives the running UI (`:7575`) with Playwright + Chromium through a scripted run and writes 1080p footage to `video/public/app.mp4` + a static `docs/screenshot.png`. |
| `build_demo.sh` | One-shot pipeline: capture → Remotion render (branded intro/outro) → high-quality `docs/demo.mp4` + `docs/demo.gif`. |

```bash
# 0. deps
. .venv/bin/activate
pip install playwright pillow && playwright install chromium

# 1. start both servers (mock backend = no API key needed)
RETREVAL_MOCK=1 RETREVAL_MOCK_DELAY=0.55 uvicorn server.app:app --port 7373 &
( cd web && npm run build && cd out && python3 -m http.server 7575 ) &

# 2. smoke test (optional)
python scripts/smoke_ui.py

# 3. build the demo
scripts/build_demo.sh
# → docs/demo.mp4, docs/demo.gif, docs/screenshot.png
```

Tunables (env): `RT_UI_URL` (default `http://localhost:7575`),
`RETREVAL_MOCK_DELAY` (pacing of the mock run), `GIF_WIDTH`, `GIF_FPS`.
The Remotion project lives in [`video/`](../video/).
