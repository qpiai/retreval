# Scripts

| script | what it does |
|---|---|
| `smoke_ui.py` | Headless-browser smoke test: loads the UI, sends a problem, and asserts the reasoning tree grows and a validated answer appears. |

```bash
. .venv/bin/activate
pip install playwright && playwright install chromium

# start both servers (mock backend = no API key needed), then:
RETREVAL_MOCK=1 uvicorn server.app:app --port 7373 &
( cd web && npm run build && cd out && python3 -m http.server 7575 ) &

python scripts/smoke_ui.py        # → "SMOKE OK …"
```

The demo media in [`docs/`](../docs/) (`demo.mp4`, `demo.gif`, `screenshot.png`)
is pre-rendered and committed; the generation pipeline is kept out of the repo.
