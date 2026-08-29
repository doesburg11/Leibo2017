import numpy as np

from leibo2017.envs.gathering import STAND_STILL, GatheringConfig, GatheringEnv
from leibo2017.envs.grid_utils import NUM_ACTIONS, OBS_SHAPE


def test_reset_shapes():
    env = GatheringEnv(GatheringConfig(episode_length=10), rng=np.random.default_rng(0))
    obs = env.reset()
    assert len(obs) == 2
    for o in obs:
        assert o.shape == OBS_SHAPE
        assert o.dtype == np.uint8


def test_episode_ends_at_episode_length():
    cfg = GatheringConfig(episode_length=5)
    env = GatheringEnv(cfg, rng=np.random.default_rng(0))
    env.reset()
    done = False
    steps = 0
    while not done:
        _, _, done, _ = env.step([STAND_STILL, STAND_STILL])
        steps += 1
        assert steps <= 5
    assert steps == 5


def test_walking_onto_apple_gives_reward_and_removes_it():
    env = GatheringEnv(GatheringConfig(n_apple=50), rng=np.random.default_rng(0))
    env.reset()
    site = env._apple_sites[0]
    p = env.players[0]
    p.row, p.col, p.orientation = site[0], site[1] - 1, 1  # one cell west of the apple, facing east
    assert env._apple_present[site] is True
    _, rewards, _, _ = env.step([0, 7])  # player 0 steps forward onto the apple; player 1 stands still
    assert rewards[0] == 1.0
    assert env._apple_present[site] is False


def test_two_beam_hits_tags_and_removes_player():
    env = GatheringEnv(GatheringConfig(n_tagged=7), rng=np.random.default_rng(0))
    env.reset()
    p0, p1 = env.players
    p0.row, p0.col, p0.orientation = 6, 5, 1  # facing east
    p1.row, p1.col = 6, 8  # directly ahead of p0's beam
    env.step([6, 7])  # USE_BEAM, STAND_STILL
    assert p1.hits_taken == 1
    assert p1.tagged_timer == 0
    env.step([6, 7])
    assert p1.tagged_timer == 7
    assert (p1.row, p1.col) == (p1.start_row, p1.start_col)


def test_action_space_matches_paper():
    assert NUM_ACTIONS == 8
