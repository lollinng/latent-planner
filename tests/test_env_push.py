import numpy as np

from latent_planner import env_push as P


def _ok(s):
    ax, ay, bx, by = s
    sep = max(abs(ax - bx), abs(ay - by)) >= P.AGENT_R + P.BLOCK_H - 1e-9  # touching allowed, never overlapping
    inside_a = P.LO_A - 1e-9 <= ax <= P.HI_A + 1e-9 and P.LO_A - 1e-9 <= ay <= P.HI_A + 1e-9
    inside_b = P.LO_B - 1e-9 <= bx <= P.HI_B + 1e-9 and P.LO_B - 1e-9 <= by <= P.HI_B + 1e-9
    return sep and inside_a and inside_b


def test_push_moves_block_and_walls_stop_it():
    s = np.array([20.0, 32.0, 30.0, 32.0])  # agent left of block, same row
    for _ in range(40):
        s = P.step_state(s, np.array([1.0, 0.0]))
    assert s[2] == P.HI_B  # block pinned against the right wall
    assert s[0] == s[2] - P.BLOCK_H - P.AGENT_R  # agent stopped touching it
    assert s[3] == 32.0  # no sideways drift


def test_invariants_under_random_and_push_policies():
    rng = np.random.default_rng(0)
    for pol in ("wander", "push"):
        for _ in range(30):
            S, _ = P.rollout(rng, 60, pol)
            assert all(_ok(s) for s in S)


def test_batch_dynamics_match_scalar():
    rng = np.random.default_rng(1)
    S = np.stack([P.rollout(rng, 20, "push")[0][-1] for _ in range(300)])
    for _ in range(25):
        act = rng.uniform(-1.3, 1.3, (len(S), 2))
        nb = P.step_state_batch(S, act)
        ns = np.stack([P.step_state(S[k], act[k]) for k in range(len(S))])
        assert np.array_equal(nb, ns)
        S = nb


def test_push_policy_moves_the_block():
    rng = np.random.default_rng(2)
    moved = [np.linalg.norm(S[-1, 2:] - S[0, 2:]) for S, _ in (P.rollout(rng, 48, "push") for _ in range(50))]
    assert np.mean(np.array(moved) > 5) > 0.5
