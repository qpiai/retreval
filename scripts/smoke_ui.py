"""Headless browser smoke test for the ReTreVal UI.

Asserts the full path works end-to-end in a real browser: app loads, a problem
streams a growing tree + logs, and a validated answer appears. Run with both
servers up (static UI on :7575, mock backend on :7373).
"""
from __future__ import annotations

import os
import sys

UI_URL = os.environ.get("RT_UI_URL", "http://localhost:7575")


def main() -> int:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_context(viewport={"width": 1280, "height": 720}).new_page()
        errors: list[str] = []
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)

        page.goto(UI_URL, wait_until="networkidle", timeout=30_000)
        page.wait_for_selector('body[data-testid="app-ready"]', timeout=10_000)
        assert page.locator('[data-testid="example-chip"]').count() >= 1, "no example chips"

        box = page.locator('[data-testid="chat-input"]')
        box.click()
        box.fill("How many divisors does 196 have?")
        box.press("Enter")

        # phase should enter running, then settle on done
        page.wait_for_function(
            "() => document.querySelector('main[data-phase]')?.getAttribute('data-phase')==='done'",
            timeout=60_000,
        )

        node_count = page.locator(".react-flow__node").count()
        assert node_count >= 4, f"expected a grown tree, got {node_count} nodes"

        body = page.inner_text("body")
        assert "9" in body, "expected answer '9' in the page"
        assert "best" in body.lower(), "expected a best-score readout"

        # No fatal React/console errors (ignore benign resource warnings).
        fatal = [e for e in errors if "Failed to load resource" not in e]
        assert not fatal, f"console errors: {fatal[:3]}"

        print(f"SMOKE OK — tree nodes={node_count}, phase=done, answer found")
        browser.close()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as e:
        print(f"SMOKE FAIL: {e}", file=sys.stderr)
        sys.exit(1)
