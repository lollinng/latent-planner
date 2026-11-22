"""Usage: python scripts/push_oracle_horizon.py KNOTS HOLD SAMPLES ITERS [BUDGET]
Oracle CEM (true push simulator, state-distance cost) on the push tuning tasks at several horizons / budgets.
Question: is the long-push gap a planning problem (oracle fails too) or a model problem?"""
import json
import sys
import time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from latent_planner import eval_push as EPU
from latent_planner.planner import CEMConfig

K, H, S, I = (int(x) for x in sys.argv[1:5])
budget = int(sys.argv[5]) if len(sys.argv) > 5 else EPU.BUDGET
tasks = EPU.load_tasks(ROOT / "results" / "push_tasks_tune.json", 100, 4040)
cfg = CEMConfig(knots=K, hold=H, samples=S, iters=I, running_w=1.0)
t0 = time.time()
r = EPU.run_episodes(EPU.OracleCEM(tasks, cfg, 0), tasks, budget=budget)
print(json.dumps({"cfg": f"K{K}xh{H} S{S} I{I} budget{budget}", "success": r["success"], "short": r["success_short"],
                  "long": r["success_long"], "steps": r["mean_steps_success"], "min": round((time.time() - t0) / 60, 1)}),
      flush=True)
