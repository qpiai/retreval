"""Capture 1080p footage of the running ReTreVal chat UI for the demo video.

Run AFTER both servers are up:

    # 1. backend (mock mode — deterministic, no API key needed)
    cd retreval_oss && . .venv/bin/activate
    RETREVAL_MOCK=1 RETREVAL_MOCK_DELAY=0.5 uvicorn server.app:app --port 7373 &

    # 2. static UI on :7575
    ( cd web && npm run build && cd out && python3 -m http.server 7575 ) &

    # 3. capture
    pip install playwright pillow && playwright install chromium
    python scripts/capture_demo.py
    # → video/public/app.mp4   (raw footage, fed to Remotion)
    # → docs/screenshot.png    (static README fallback)

The script drives a scripted run — type a problem, watch the reasoning tree
grow node-by-node with live logs, then read the validated answer — with
human-like keystrokes so the recording reads naturally.

Kept in scripts/ (not the package) so it never ships in a wheel.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

UI_URL = os.environ.get("RT_UI_URL", "http://localhost:7575")
ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
APP_MP4 = ROOT / "video" / "public" / "app.mp4"
DOCS.mkdir(parents=True, exist_ok=True)
APP_MP4.parent.mkdir(parents=True, exist_ok=True)

CAP_W = int(os.environ.get("RT_CAP_W", "1280"))
CAP_H = int(os.environ.get("RT_CAP_H", "720"))
OUT_W, OUT_H = 1920, 1080

# (query, hold-after-done seconds) — two different-domain prompts, back-to-back.
SCRIPT: list[tuple[str, float]] = [
    ("How many positive whole-number divisors does 196 have?", 3.0),
    ("Who introduced the GELU activation function, and what is its formula?", 3.6),
]


def _wait_for_idle(page, timeout_ms: int = 120_000) -> None:
    """Wait until the run finishes — main[data-phase] leaves 'running'."""
    page.wait_for_function(
        """() => {
            const m = document.querySelector('main[data-phase]');
            return m && m.getAttribute('data-phase') !== 'running';
        }""",
        timeout=timeout_ms,
    )


def _send(page, query: str, hold: float) -> None:
    box = page.locator('[data-testid="chat-input"]')
    box.click()
    box.fill("")
    box.type(query, delay=32)            # human-like keystrokes
    time.sleep(0.5)
    box.press("Enter")
    time.sleep(0.4)
    _wait_for_idle(page)
    time.sleep(hold)                     # let the viewer read the answer + tree


def main() -> int:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("ERROR: pip install playwright pillow && playwright install chromium",
              file=sys.stderr)
        return 1

    record_dir = DOCS / ".rec"
    if record_dir.exists():
        shutil.rmtree(record_dir)
    record_dir.mkdir(parents=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": CAP_W, "height": CAP_H},
            device_scale_factor=2,
            record_video_dir=str(record_dir),
            record_video_size={"width": CAP_W, "height": CAP_H},
        )
        # Capture in LIGHT mode.
        context.add_init_script(
            "try{localStorage.setItem('retreval-theme','light')}catch(e){}"
        )
        page = context.new_page()
        print(f"[capture] opening {UI_URL}", flush=True)
        page.goto(UI_URL, wait_until="networkidle", timeout=30_000)
        page.wait_for_selector('body[data-testid="app-ready"]', timeout=10_000)
        time.sleep(1.6)                  # hold on the empty state + example chips

        for query, hold in SCRIPT:
            print(f"[capture] {query[:50]}…", flush=True)
            _send(page, query, hold)
            page.screenshot(path=str(DOCS / "screenshot.png"), full_page=False)
            print(f"  → {DOCS / 'screenshot.png'}", flush=True)

        # Never ship a demo with an error baked in.
        body = page.inner_text("body")
        bad = next((m for m in ("⚠️", "Could not reach", "Something went wrong")
                    if m in body), None)
        if bad:
            context.close(); browser.close()
            print(f"ERROR: UI showed an error during capture ({bad!r}); re-run.",
                  file=sys.stderr)
            return 2

        time.sleep(1.0)
        context.close()                  # finalises the .webm
        browser.close()

    webm = next(iter(record_dir.glob("*.webm")), None)
    if not webm:
        print("ERROR: no .webm recorded", file=sys.stderr)
        return 1

    if not shutil.which("ffmpeg"):
        print(f"[capture] ffmpeg not on PATH; raw footage at {webm}", file=sys.stderr)
        return 1

    print(f"[capture] ffmpeg → {APP_MP4}", flush=True)
    subprocess.run([
        "ffmpeg", "-y", "-i", str(webm),
        "-vf", f"scale={OUT_W}:{OUT_H}:flags=lanczos,fps=30",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(APP_MP4),
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"  → {APP_MP4} ({APP_MP4.stat().st_size / 1e6:.1f} MB)", flush=True)

    shutil.rmtree(record_dir, ignore_errors=True)
    print("[capture] done. Next: scripts/build_demo.sh", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
