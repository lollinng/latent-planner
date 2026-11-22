"""Headless browser check of the demo: loads, plans, reaches goals, no console errors.

    python tests/web_smoke.py [--url http://127.0.0.1:8899/] [--shots DIR]

Serves web/ on 127.0.0.1 itself when no --url is given (loopback only).
"""
import argparse
import functools
import http.server
import json
import threading
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def serve(port: int) -> None:
    h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT / "web"))
    http.server.ThreadingHTTPServer.allow_reuse_address = True
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), h)
    threading.Thread(target=srv.serve_forever, daemon=True).start()


def run(url: str, shots: Path, goals: int, engine: str) -> dict:
    out = {"engine": engine}
    with sync_playwright() as p:
        if engine == "chrome":
            browser = p.chromium.launch(channel="chrome", headless=True)
        else:
            browser = getattr(p, engine).launch(headless=True)
        for name, vp in [("desktop", {"width": 1200, "height": 900}), ("phone", {"width": 390, "height": 844})]:
            ctx = browser.new_context(viewport=vp, device_scale_factor=2 if name == "phone" else 1,
                                      is_mobile=(name == "phone" and engine != "firefox"),
                                      has_touch=(name == "phone"))
            page = ctx.new_page()
            errors = []
            page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            page.on("pageerror", lambda e: errors.append(str(e)))
            t0 = time.time()
            page.goto(url)
            page.wait_for_function("window.__lp && window.__lp.S.ready", timeout=120_000)
            load_s = time.time() - t0
            reached = []
            for g in range(goals):
                page.wait_for_function("window.__lp.S.reached", timeout=120_000)
                reached.append(page.evaluate("window.__lp.S.steps"))
                if g == 0:
                    page.evaluate("document.getElementById('showCost').click()")
                    page.screenshot(path=str(shots / f"{engine}_{name}.png"), full_page=(name == "phone"))
                    page.evaluate("document.getElementById('showCost').click()")
                page.click("#newTask")
            times = page.evaluate("window.__lp.S.planTimes")
            scroll_w = page.evaluate("document.documentElement.scrollWidth")
            out[name] = {"load_s": round(load_s, 2), "steps_to_goal": reached,
                         "plan_ms_median": sorted(times)[len(times) // 2] if times else None,
                         "horizontal_overflow": scroll_w > vp["width"], "console_errors": errors}
            ctx.close()
        browser.close()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None)
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--shots", default=str(ROOT / "results" / "screens"))
    ap.add_argument("--goals", type=int, default=3)
    ap.add_argument("--engine", default="chrome", choices=["chrome", "chromium", "webkit", "firefox"])
    a = ap.parse_args()
    if a.url is None:
        serve(a.port)
        a.url = f"http://127.0.0.1:{a.port}/"
    Path(a.shots).mkdir(parents=True, exist_ok=True)
    print(json.dumps(run(a.url, Path(a.shots), a.goals, a.engine), indent=1))
