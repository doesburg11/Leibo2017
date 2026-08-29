"""Tests for the optional RLlib wrapper. Skipped entirely if `ray` isn't
installed, since it's an optional extra (requirements-rllib.txt), not part
of the base install."""
import numpy as np
import pytest

ray = pytest.importorskip("ray")

from leibo2017.envs.grid_utils import NUM_ACTIONS, OBS_SHAPE  # noqa: E402
from leibo2017.envs.rllib_wrappers import (  # noqa: E402
    AGENT_IDS,
    make_gathering_rllib_env,
    make_wolfpack_rllib_env,
)

FLAT_DIM = int(np.prod(OBS_SHAPE))


@pytest.mark.parametrize("factory", [make_gathering_rllib_env, make_wolfpack_rllib_env])
def test_reset_and_step_shapes(factory):
    env = factory({"episode_length": 5})
    obs, info = env.reset()
    assert set(obs.keys()) == set(AGENT_IDS)
    for aid in AGENT_IDS:
        assert obs[aid].shape == (FLAT_DIM,)
        assert obs[aid].dtype == np.float32
        assert obs[aid].min() >= 0.0 and obs[aid].max() <= 1.0

    action_dict = {aid: 0 for aid in AGENT_IDS}
    obs, rewards, terminations, truncations, infos = env.step(action_dict)
    assert set(rewards.keys()) == set(AGENT_IDS)
    assert "__all__" in terminations and "__all__" in truncations


@pytest.mark.parametrize("factory", [make_gathering_rllib_env, make_wolfpack_rllib_env])
def test_episode_length_drives_truncation(factory):
    env = factory({"episode_length": 3})
    env.reset()
    action_dict = {aid: 0 for aid in AGENT_IDS}
    for _ in range(2):
        _, _, terminations, truncations, _ = env.step(action_dict)
        assert terminations["__all__"] is False
        assert truncations["__all__"] is False
    _, _, terminations, truncations, _ = env.step(action_dict)
    assert truncations["__all__"] is True


def test_observation_space_matches_flattened_obs_shape():
    env = make_gathering_rllib_env()
    assert env.observation_space.shape == (FLAT_DIM,)
    assert env.action_space.n == NUM_ACTIONS


def _rollout_obs_trajectory(factory, seed, n_steps=15):
    """Run STAND_STILL for both agents and collect observations -- with
    both wolves standing still, Wolfpack's only source of variation is the
    scripted prey's own flee/wander RNG draws, which is what this is
    actually probing."""
    env = factory({"seed": seed, "episode_length": n_steps + 1})
    env.reset()
    action_dict = {aid: 7 for aid in AGENT_IDS}  # STAND_STILL
    traj = []
    for _ in range(n_steps):
        obs, _, _, _, _ = env.step(action_dict)
        traj.append(obs[AGENT_IDS[0]].copy())
    return traj


def test_construction_time_seed_via_env_config_is_reproducible():
    """Regression test (Codex review, 2026-08-29): env_config={"seed": ...}
    used to be silently dropped -- both make_*_rllib_env factories stripped
    "seed" out of the kwargs passed to GatheringConfig/WolfpackConfig
    without ever using it, so two envs built with the same seed still got
    independent, unseeded RNG streams."""
    traj_a = _rollout_obs_trajectory(make_wolfpack_rllib_env, seed=123)
    traj_b = _rollout_obs_trajectory(make_wolfpack_rllib_env, seed=123)
    for a, b in zip(traj_a, traj_b):
        assert np.array_equal(a, b)

    traj_c = _rollout_obs_trajectory(make_wolfpack_rllib_env, seed=456)
    assert any(not np.array_equal(a, c) for a, c in zip(traj_a, traj_c))


def test_reset_seed_reseeds_the_same_env_instance():
    """Gymnasium/RLlib's per-episode reset(seed=...) contract: calling
    reset with a seed should make everything from that point on
    deterministic, regardless of whatever randomness happened before."""
    env1 = make_wolfpack_rllib_env({"episode_length": 20})
    env1.reset()
    for _ in range(3):  # burn some unseeded random steps first
        env1.step({aid: 7 for aid in AGENT_IDS})
    env1.reset(seed=999)
    traj1 = [env1.step({aid: 7 for aid in AGENT_IDS})[0][AGENT_IDS[0]].copy() for _ in range(10)]

    env2 = make_wolfpack_rllib_env({"episode_length": 20, "seed": 42})  # different initial seed
    env2.reset(seed=999)
    traj2 = [env2.step({aid: 7 for aid in AGENT_IDS})[0][AGENT_IDS[0]].copy() for _ in range(10)]

    for a, b in zip(traj1, traj2):
        assert np.array_equal(a, b)


def test_gathering_beam_use_rate_is_split_per_agent_not_duplicated_as_a_list():
    """Regression test (Codex review, 2026-08-29): info dicts used to be
    copied wholesale to both agent ids, so both RLlib agents' info
    contained the *same 2-element* beam_use_rate list rather than each
    agent getting its own scalar."""
    env = make_gathering_rllib_env({"episode_length": 5})
    env.reset()
    action_dict = {"player-0": 6, "player-1": 7}  # USE_BEAM, STAND_STILL
    _, _, _, _, infos = env.step(action_dict)
    for aid in AGENT_IDS:
        assert isinstance(infos[aid]["beam_use_rate"], float)
    assert infos["player-0"]["beam_use_rate"] != infos["player-1"]["beam_use_rate"]


def test_wolfpack_global_stats_are_shared_across_both_agents():
    env = make_wolfpack_rllib_env({"episode_length": 5})
    env.reset()
    _, _, _, _, infos = env.step({aid: 7 for aid in AGENT_IDS})
    assert infos["player-0"]["captures"] == infos["player-1"]["captures"]
