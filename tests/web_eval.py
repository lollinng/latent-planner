"""Run the browser's own planner (web/src/cem.js + ONNX Runtime Web + the JS env) on the 100 fixed test tasks
in headless Chrome, with the same success rule and budget as latent_planner/eval.py (two rooms) or
latent_planner/eval_push.py (push), and log the row.

    python tests/web_eval.py [--n 100] [--scene push]
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from web_smoke import ROOT, serve  # noqa: E402

sys.path.insert(0, str(ROOT))
from latent_planner import eval_push  # noqa: E402
from latent_planner.eval import BUDGET, SUCCESS_PX, append_csv  # noqa: E402

JS = """
async ([tasks, budget, successPx, scene]) => {
  const { S, meta, planner, E, EP, Planner } = window.__lp;
  const push = scene === "push";
  const render = push ? EP.render : E.render, step = push ? EP.stepState : E.stepPos;
  // two rooms: agent within successPx of the goal; push: block centre within successPx of the goal block
  const gap = (p, g) => (push ? Math.hypot(p[2] - g[2], p[3] - g[3]) : Math.hypot(p[0] - g[0], p[1] - g[1]));
  S.running = false;                       // stop the demo loop; we drive the planner directly
  await new Promise((r) => setTimeout(r, 500));
  const out = [];
  for (let k = 0; k < tasks.starts.length; k++) {
    let pos = tasks.starts[k].slice();
    const goal = tasks.goals[k];
    const zg = await planner.encode(render(goal));
    let warm = null, t = 0, done = false, ms = [];
    while (t < budget && !done) {
      const z = await planner.encode(render(pos));
      const res = await planner.plan(z, zg, warm);
      ms.push(res.ms);
      for (let j = 0; j < meta.cem.hold && t < budget; j++) {
        pos = step(pos, [res.mean[0], res.mean[1]]);
        t++;
        if (gap(pos, goal) <= successPx) { done = true; break; }
      }
      warm = Planner.shift(res.mean);
    }
    out.push({ done, steps: t, final: gap(pos, goal),
               ms: ms.reduce((a, b) => a + b, 0) / ms.length });
  }
  return out;
}
"""

def run_push(a) -> None:
    tasks = json.loads(Path(a.tasks or ROOT / "results" / "push_tasks_test.json").read_text())
    tasks = {k: (v[: a.n] if isinstance(v, list) else v) for k, v in tasks.items()}
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True)
        page = b.new_page()
        page.goto(f"http://127.0.0.1:{a.port}/#push")
        page.wait_for_function("window.__lp && window.__lp.S.ready && window.__lp.scene === 'push'", timeout=120_000)
        meta = page.evaluate("window.__lp.meta")
        t0 = time.time()
        res = page.evaluate(JS, [tasks, eval_push.BUDGET, eval_push.SUCCESS_PX, "push"])
        b.close()
    ok = [r["done"] for r in res]
    long_ = [m >= eval_push.LONG_PX for m in tasks["moved"]]
    succ = [r["steps"] for r in res if r["done"]]
    row = {"tag": a.tag, "taskset": "test", "method": "browser_cem", "run": f"web/model_push ({meta['run']})",
           "frameskip": meta["frameskip"], "success": sum(ok) / len(ok),
           "success_short": sum(o for o, l in zip(ok, long_) if not l) / max(1, long_.count(False)),
           "success_long": sum(o for o, l in zip(ok, long_) if l) / max(1, long_.count(True)),
           "mean_steps_success": sum(succ) / max(1, len(succ)),
           "final_block_dist_px": sum(r["final"] for r in res) / len(res),
           "sec_per_replan": sum(r["ms"] for r in res) / len(res) / 1000,
           "cem": "browser: " + json.dumps(meta["cem"], separators=(",", ":"))}
    if a.n == 100 and not a.tasks:
        with open(ROOT / "results" / "eval_push.csv", "a", newline="") as f:
            csv.DictWriter(f, fieldnames=eval_push.FIELDS, extrasaction="ignore").writerow(row)
    print(json.dumps({k: row[k] for k in ("success", "success_short", "success_long", "mean_steps_success",
                                          "sec_per_replan")}), f"{time.time() - t0:.0f}s",
          "failed tasks:", [i for i, o in enumerate(ok) if not o])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--port", type=int, default=8893)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--scene", default="tworoom", choices=["tworoom", "push"])
    ap.add_argument("--tasks", default=None, help="push only: another task file, e.g. one made by the JS generator "
                    "(not logged)")
    a = ap.parse_args()
    serve(a.port)
    if a.scene == "push":
        sys.exit(run_push(a))
    tasks = json.loads((ROOT / "results" / "tasks_test.json").read_text())
    tasks = {k: (v[: a.n] if isinstance(v, list) else v) for k, v in tasks.items()}
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True)
        page = b.new_page()
        page.goto(f"http://127.0.0.1:{a.port}/")
        page.wait_for_function("window.__lp && window.__lp.S.ready", timeout=120_000)
        meta = page.evaluate("window.__lp.meta")
        t0 = time.time()
        res = page.evaluate(JS, [tasks, BUDGET, SUCCESS_PX, "tworoom"])
        b.close()
    cross = tasks["cross"]
    ok = [r["done"] for r in res]
    succ = [r["steps"] for r in res if r["done"]]
    row = {"tag": a.tag, "taskset": "test", "method": "browser_cem", "run": f"web/model ({meta['run']})",
           "frameskip": meta["frameskip"], "success": sum(ok) / len(ok),
           "success_same": sum(o for o, c in zip(ok, cross) if not c) / max(1, cross.count(False)),
           "success_cross": sum(o for o, c in zip(ok, cross) if c) / max(1, cross.count(True)),
           "mean_steps_success": sum(succ) / max(1, len(succ)),
           "final_dist_px": sum(r["final"] for r in res) / len(res), "replans": None,
           "sec_per_replan": sum(r["ms"] for r in res) / len(res) / 1000,
           "cem": "browser: " + json.dumps(meta["cem"], separators=(",", ":"))}
    if a.n == 100:
        append_csv(ROOT / "results" / "eval.csv", row)
        fails = [i for i, o in enumerate(ok) if not o]
        (ROOT / "results" / "eval" / a.tag / "test__browser_cem.json").write_text(
            json.dumps({**row, "per_task": res, "failed_tasks": fails}))
    print(json.dumps({k: row[k] for k in ("success", "success_same", "success_cross", "mean_steps_success",
                                          "sec_per_replan")}), f"{time.time() - t0:.0f}s",
          "failed tasks:", [i for i, o in enumerate(ok) if not o])
