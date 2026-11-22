"""Tables for results/horizon/SUBGOAL.md from results/horizon/eval.csv and the per-run JSON files.

    .venv/bin/python scripts/subgoal_table.py
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RES = Path("results/horizon")
rows = list(csv.DictReader(open(RES / "eval.csv")))
groups = defaultdict(list)
for r in rows:
    if r["tag"].startswith("check"):
        continue
    K = int(r["cem"].split("x")[0][1:])
    groups[(r["taskset"], r["method"], K)].append(r)


def pct(v):
    return f"{100 * np.mean(v):.0f} ± {100 * np.std(v):.0f}%"


NAMES = {"latent_cem": "plain latent CEM", "latent_cem_subgoal": "subgoals, pick within horizon",
         "latent_cem_subgoal_beyond": "subgoals, pick at/after horizon"}
for ts in ("test", "far", "tune"):
    print(f"\n### {ts} tasks\n")
    print("| Planner | Horizon (env steps) | Success | Same room | Other room | Steps (successes) | Per-seed success | ms/replan |")
    print("|---|---|---|---|---|---|---|---|")
    for (t, method, K), rs in sorted(groups.items(), key=lambda kv: (kv[0][2], list(NAMES).index(kv[0][1]))):
        if t != ts:
            continue
        rs = sorted(rs, key=lambda r: r["seed"])
        s = [float(r["success"]) for r in rs]
        name = NAMES[method]
        same = "-" if ts == "far" else pct([float(r["success_same"]) for r in rs])
        print(f"| {name} | K={K} ({4 * K}) | {pct(s)} | {same} | {pct([float(r['success_cross']) for r in rs])} | "
              f"{np.mean([float(r['mean_steps_success']) for r in rs]):.1f} | {', '.join(f'{x:.2f}' for x in s)} | "
              f"{1000 * np.mean([float(r['sec_per_replan']) for r in rs]):.0f} |")

# graph stats and GPU memory from the subgoal JSON files
print()
for p in sorted(RES.glob("eval/subgoal*_K*/*.json")):
    d = json.loads(p.read_text())
    print(p.parent.name, p.name, d["graph"], d["graph_diag"], f"gpu {d['gpu_peak_gb']:.2f} GB",
          f"rss {d['host_peak_rss_gb']:.2f} GB")
