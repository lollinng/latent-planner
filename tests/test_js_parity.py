"""The browser env (web/src/env.js) must produce exactly the same positions and pixels as env.py."""
import base64
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from latent_planner import env

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_env_matches_python(tmp_path):
    rng = np.random.default_rng(3)
    pos = [env.sample_free(rng) for _ in range(400)]
    # wall-hugging states and out-of-range actions too
    pos += [np.array([env.WALL_X0 - env.AGENT_R, 20.0]), np.array([env.WALL_X1 + env.AGENT_R, 45.5]),
            np.array([env.BORDER + env.AGENT_R, env.SIZE - env.BORDER - env.AGENT_R])]
    act = rng.uniform(-1.5, 1.5, (len(pos), 2))
    inp = tmp_path / "in.json"
    inp.write_text(json.dumps({"pos": [p.tolist() for p in pos], "act": act.tolist()}))
    out = json.loads(subprocess.run(["node", str(ROOT / "tests/js_parity.mjs"), str(inp)], check=True,
                                    capture_output=True, text=True).stdout)
    for k, p in enumerate(pos):
        assert out["next"][k] == env.step_pos(p, act[k]).tolist(), k  # exact float64 equality
        js = np.frombuffer(base64.b64decode(out["frames"][k]), dtype=np.uint8).reshape(64, 64, 3)
        assert np.array_equal(js, env.render(p)), k


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_push_env_matches_python(tmp_path):
    from latent_planner import env_push as P

    rng = np.random.default_rng(4)
    states = []
    for _ in range(60):  # states from push episodes, so many are in contact or against walls
        S, _ = P.rollout(rng, 30, "push")
        states += [S[k] for k in (5, 15, 30)]
    states += [np.array([P.HI_B - P.BLOCK_H - P.AGENT_R, 30.0, P.HI_B, 30.0])]  # pinned against the right wall
    act = rng.uniform(-1.5, 1.5, (len(states), 2))
    inp = tmp_path / "in.json"
    inp.write_text(json.dumps({"states": [s.tolist() for s in states], "act": act.tolist()}))
    out = json.loads(subprocess.run(["node", str(ROOT / "tests/js_parity_push.mjs"), str(inp)], check=True,
                                    capture_output=True, text=True).stdout)
    for k, s in enumerate(states):
        assert out["next"][k] == P.step_state(s, act[k]).tolist(), k
        js = np.frombuffer(base64.b64decode(out["frames"][k]), dtype=np.uint8).reshape(64, 64, 3)
        assert np.array_equal(js, P.render(s)), k
