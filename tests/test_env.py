import numpy as np

from latent_planner import env


def test_render_batch_matches_render():
    rng = np.random.default_rng(0)
    pos = np.stack([env.sample_free(rng) for _ in range(300)])
    # include states pressed against walls and corners, where the 10x10 window is clipped
    edge = np.array([[env.BORDER + env.AGENT_R, env.BORDER + env.AGENT_R],
                     [env.SIZE - env.BORDER - env.AGENT_R, env.SIZE - env.BORDER - env.AGENT_R],
                     [env.WALL_X0 - env.AGENT_R, 10.0], [env.WALL_X1 + env.AGENT_R, 50.0]])
    pos = np.concatenate([pos, edge])
    batch = env.render_batch(pos)
    for k in range(pos.shape[0]):
        assert np.array_equal(batch[k], env.render(pos[k])), k


def test_walls_block_and_door_passes():
    p = np.array([20.0, 20.0])
    for _ in range(20):
        p = env.step_pos(p, np.array([1.0, 0.0]))
    assert p[0] == env.WALL_X0 - env.AGENT_R  # stopped by the middle wall
    p = np.array([20.0, 49.0])  # door band
    for _ in range(20):
        p = env.step_pos(p, np.array([1.0, 0.0]))
    assert env.room_of(p[0]) == 1 and env.is_free(*p)


def test_dynamics_stay_in_free_space():
    rng = np.random.default_rng(1)
    p = env.sample_free(rng)
    for _ in range(2000):
        p = env.step_pos(p, rng.uniform(-1, 1, 2))
        assert env.is_free(p[0], p[1])


def test_step_pos_batch_matches_scalar():
    rng = np.random.default_rng(2)
    pos = np.stack([env.sample_free(rng) for _ in range(500)])
    for _ in range(30):
        act = rng.uniform(-1.3, 1.3, (500, 2))
        nb = env.step_pos_batch(pos, act)
        ns = np.stack([env.step_pos(pos[k], act[k]) for k in range(500)])
        assert np.array_equal(nb, ns)
        pos = nb
