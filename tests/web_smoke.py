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


def run(url: str, shots: Path, goals: int, engine: str, cpu_throttle: float = 1.0, scene: str = "tworoom") -> dict:
    out = {"engine": engine, "cpu_throttle": cpu_throttle, "scene": scene}
    if scene == "push":
        url = url.split("#")[0] + "#push"
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
            if cpu_throttle > 1 and engine in ("chrome", "chromium"):  # rough stand-in for a slower phone CPU
                ctx.new_cdp_session(page).send("Emulation.setCPUThrottlingRate", {"rate": cpu_throttle})
            errors = []
            page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            page.on("pageerror", lambda e: errors.append(str(e)))
            t0 = time.time()
            page.goto(url)
            page.wait_for_function("window.__lp && window.__lp.S.ready", timeout=120_000)
            load_s = time.time() - t0
            reached = []
            for g in range(goals):
                # a task ends when the goal is reached or the demo gives up (scene.giveUp steps)
                try:
                    page.wait_for_function("window.__lp.S.reached || window.__lp.S.gaveUp", timeout=120_000)
                    n = page.evaluate("window.__lp.S.steps")
                    reached.append(n if page.evaluate("window.__lp.S.reached") else f"gave up at {n}")
                except Exception:
                    reached.append(f"no end in 120 s ({page.evaluate('window.__lp.S.steps')} steps)")
                if g == 0:
                    page.evaluate("document.getElementById('showCost').click()")
                    page.screenshot(path=str(shots / f"{engine}_{scene}_{name}.png"), full_page=(name == "phone"))
                    page.evaluate("document.getElementById('showCost').click()")
                if g == goals - 1:  # last task: the demo must move on to a new task by itself
                    v = page.evaluate("window.__lp.S.goalVersion")
                    page.wait_for_function(f"window.__lp.S.goalVersion > {v}", timeout=10_000)
                else:
                    page.click("#newTask")
            times = page.evaluate("window.__lp.S.planTimes")
            scroll_w = page.evaluate("document.documentElement.scrollWidth")
            assert page.evaluate("window.__lp.scene") == scene
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
    ap.add_argument("--cpu-throttle", type=float, default=1.0)
    ap.add_argument("--scene", default="tworoom", choices=["tworoom", "push"])
    a = ap.parse_args()
    if a.url is None:
        serve(a.port)
        a.url = f"http://127.0.0.1:{a.port}/"
    Path(a.shots).mkdir(parents=True, exist_ok=True)
    print(json.dumps(run(a.url, Path(a.shots), a.goals, a.engine, a.cpu_throttle, a.scene), indent=1))
