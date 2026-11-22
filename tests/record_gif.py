"""Record a short GIF of the demo planning in headless Chrome (for the README).

    python tests/record_gif.py --out docs/demo.gif --seconds 14
"""
import argparse
import io
import sys
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from web_smoke import serve  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/demo.gif")
    ap.add_argument("--seconds", type=float, default=14)
    ap.add_argument("--fps", type=float, default=8)
    ap.add_argument("--port", type=int, default=8897)
    ap.add_argument("--cost-map", action="store_true")
    a = ap.parse_args()
    serve(a.port)
    frames = []
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True)
        page = b.new_page(viewport={"width": 1100, "height": 760})
        page.goto(f"http://127.0.0.1:{a.port}/")
        page.wait_for_function("window.__lp && window.__lp.S.ready", timeout=120_000)
        if a.cost_map:
            page.click("#showCost")
        el = page.locator("main.app")
        t_end = time.time() + a.seconds
        while time.time() < t_end:
            t0 = time.time()
            frames.append(Image.open(io.BytesIO(el.screenshot())).convert("RGB"))
            time.sleep(max(0.0, 1 / a.fps - (time.time() - t0)))
        b.close()
    w = 720
    frames = [f.resize((w, round(f.height * w / f.width)), Image.LANCZOS) for f in frames]
    pal = [f.quantize(colors=96, method=Image.Quantize.MEDIANCUT) for f in frames]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pal[0].save(a.out, save_all=True, append_images=pal[1:], duration=int(1000 / a.fps), loop=0, optimize=True)
    print(f"{a.out}: {len(frames)} frames, {Path(a.out).stat().st_size / 1e6:.2f} MB")
