import numpy as np

from leibo2017.envs.grid_utils import OBS_SHAPE, STAND_STILL
from leibo2017.envs.wolfpack import WolfpackConfig, WolfpackEnv


def test_reset_shapes():
    env = WolfpackEnv(WolfpackConfig(episode_length=10), rng=np.random.default_rng(0))
    obs = env.reset()
    assert len(obs) == 2
    for o in obs:
        assert o.shape == OBS_SHAPE


def test_episode_ends_at_episode_length():
    env = WolfpackEnv(WolfpackConfig(episode_length=5), rng=np.random.default_rng(0))
    env.reset()
    done = False
    steps = 0
    while not done:
        _, _, done, _ = env.step([STAND_STILL, STAND_STILL])
        steps += 1
    assert steps == 5


def test_solo_capture_gives_r_lone_only_to_capturer():
    env = WolfpackEnv(WolfpackConfig(capture_radius=0, r_team=5.0, r_lone=1.0), rng=np.random.default_rng(0))
    env.reset()
    env.wolves[0]["row"], env.wolves[0]["col"] = 5, 5
    env.wolves[1]["row"], env.wolves[1]["col"] = 15, 15  # far away, outside any radius
    env.prey["row"], env.prey["col"] = 5, 6
    env.wolves[0]["orientation"] = 1  # facing east, one step from the prey
    obs, rewards, done, info = env.step([0, STAND_STILL])  # wolf 0 steps forward onto the prey
    assert rewards[0] == 1.0
    assert rewards[1] == 0.0
    assert info["captures"] == 1


def test_joint_capture_within_radius_gives_r_team_to_both():
    env = WolfpackEnv(WolfpackConfig(capture_radius=3, r_team=5.0, r_lone=1.0), rng=np.random.default_rng(0))
    env.reset()
    env.wolves[0]["row"], env.wolves[0]["col"] = 5, 5
    env.wolves[1]["row"], env.wolves[1]["col"] = 5, 7  # within capture_radius=3 of the prey
    env.prey["row"], env.prey["col"] = 5, 6
    env.wolves[0]["orientation"] = 1
    obs, rewards, done, info = env.step([0, STAND_STILL])
    assert rewards[0] == 5.0
    assert rewards[1] == 5.0
