"""Run the browser's own planner (web/src/cem.js + ONNX Runtime Web + web/src/env.js) on the 100 fixed test tasks
in headless Chrome, with the same success rule and budget as latent_planner/eval.py, and log the row.

    python tests/web_eval.py [--n 100]
"""
import argparse
import json
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from web_smoke import ROOT, serve  # noqa: E402

sys.path.insert(0, str(ROOT))
from latent_planner.eval import BUDGET, SUCCESS_PX, append_csv  # noqa: E402

JS = """
async ([tasks, budget, successPx]) => {
  const { S, meta, planner, E, Planner } = window.__lp;
  S.running = false;                       // stop the demo loop; we drive the planner directly
  await new Promise((r) => setTimeout(r, 500));
  const out = [];
  for (let k = 0; k < tasks.starts.length; k++) {
    let pos = tasks.starts[k].slice();
    const goal = tasks.goals[k];
    const zg = await planner.encode(E.render(goal));
    let warm = null, t = 0, done = false, ms = [];
    while (t < budget && !done) {
      const z = await planner.encode(E.render(pos));
      const res = await planner.plan(z, zg, warm);
      ms.push(res.ms);
      for (let j = 0; j < meta.cem.hold && t < budget; j++) {
        pos = E.stepPos(pos, [res.mean[0], res.mean[1]]);
        t++;
        if (Math.hypot(pos[0] - goal[0], pos[1] - goal[1]) <= successPx) { done = true; break; }
      }
      warm = Planner.shift(res.mean);
    }
    out.push({ done, steps: t, final: Math.hypot(pos[0] - goal[0], pos[1] - goal[1]),
               ms: ms.reduce((a, b) => a + b, 0) / ms.length });
  }
  return out;
}
"""

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--port", type=int, default=8893)
    ap.add_argument("--tag", default="main")
    a = ap.parse_args()
    serve(a.port)
    tasks = json.loads((ROOT / "results" / "tasks_test.json").read_text())
    tasks = {k: (v[: a.n] if isinstance(v, list) else v) for k, v in tasks.items()}
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True)
        page = b.new_page()
        page.goto(f"http://127.0.0.1:{a.port}/")
        page.wait_for_function("window.__lp && window.__lp.S.ready", timeout=120_000)
        meta = page.evaluate("window.__lp.meta")
        t0 = time.time()
        res = page.evaluate(JS, [tasks, BUDGET, SUCCESS_PX])
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
