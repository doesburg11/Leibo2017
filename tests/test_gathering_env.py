import numpy as np

from leibo2017.envs.gathering import STAND_STILL, USE_BEAM, GatheringConfig, GatheringEnv
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


def test_two_beam_hits_tags_player():
    env = GatheringEnv(GatheringConfig(n_tagged=7), rng=np.random.default_rng(0))
    env.reset()
    p0, p1 = env.players
    p0.row, p0.col, p0.orientation = 6, 5, 1  # facing east
    p1.row, p1.col = 6, 8  # directly ahead of p0's beam
    env.step([USE_BEAM, STAND_STILL])
    assert p1.hits_taken == 1
    assert p1.tagged_timer == 0
    env.step([USE_BEAM, STAND_STILL])
    assert p1.tagged_timer == 7


def test_tagged_player_is_removed_for_exactly_n_tagged_steps_then_reactivates():
    """Regression test (Codex review, 2026-08-29): the cooldown used to be
    decremented in the same step it was set, and the reactivated player was
    teleported straight to start_row/start_col without checking whether the
    other player was standing there. This checks both: the player misses
    exactly n_tagged subsequent action opportunities (not n_tagged - 1),
    and reactivates onto a genuinely free cell."""
    env = GatheringEnv(GatheringConfig(n_tagged=3), rng=np.random.default_rng(0))
    env.reset()
    p0, p1 = env.players
    p0.row, p0.col, p0.orientation = 6, 5, 1
    p1.row, p1.col = 6, 8
    env.step([USE_BEAM, STAND_STILL])
    env.step([USE_BEAM, STAND_STILL])  # second hit -> tagged now, timer = 3
    assert p1.tagged_timer == 3
    apples_before = p1.apples_collected
    for expected_timer_after_step in (2, 1, 0):
        env.step([STAND_STILL, STAND_STILL])
        assert p1.tagged_timer == expected_timer_after_step
        assert p1.apples_collected == apples_before  # still removed: can't act, can't score
    assert (p1.row, p1.col) == (p1.start_row, p1.start_col)  # start cell was free -> used directly


def test_reactivation_falls_back_to_a_free_cell_if_start_cell_is_occupied():
    env = GatheringEnv(GatheringConfig(n_tagged=1), rng=np.random.default_rng(0))
    env.reset()
    p0, p1 = env.players
    p0.row, p0.col, p0.orientation = 6, 5, 1
    p1.row, p1.col = 6, 8
    env.step([USE_BEAM, STAND_STILL])
    env.step([USE_BEAM, STAND_STILL])  # tagged, timer = 1
    p0.row, p0.col = p1.start_row, p1.start_col  # occupy p1's start cell while it's away
    env.step([STAND_STILL, STAND_STILL])  # timer 1 -> 0: reactivation must avoid p0's cell
    assert p1.tagged_timer == 0
    assert (p1.row, p1.col) != (p0.row, p0.col)


def test_action_space_matches_paper():
    assert NUM_ACTIONS == 8
