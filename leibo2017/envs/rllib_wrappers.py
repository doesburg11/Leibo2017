"""Optional Ray RLlib adapter for GatheringEnv/WolfpackEnv.

This is a separate, additive training backend, not a replacement for
`leibo2017/agents/dqn.py` and `leibo2017/training/loop.py`: those implement
the paper's own independent-DQN method literally and are what every
`run_*.py` script in this repo uses by default. This module exists so the
exact same paper-faithful environments can optionally be trained with
RLlib's standard algorithms (PPO, DQN, ...) instead, e.g. to check whether
a more capable/standard implementation reaches qualitatively different
conclusions than the paper's own simple setup. Requires `ray[rllib]`,
which is *not* in `requirements.txt` -- see `requirements-rllib.txt` and
the README's "Optional: RLlib backend" section.

Both `GatheringEnv` and `WolfpackEnv` share the same minimal, list-indexed
API (`reset() -> (obs0, obs1)`, `step([a0, a1]) -> (obs_tuple, rewards,
done, info)`); this wrapper adapts that directly to RLlib's dict-keyed
`MultiAgentEnv` API (new API stack, RLlib >= 2.x) without changing either
environment. Both games always keep exactly two agents present for the
whole episode (a "tagged"/removed player still receives an observation and
a turn each step inside the wrapped env -- see gathering.py/wolfpack.py),
so this wrapper never needs to add or remove agents mid-episode.
"""

from __future__ import annotations

import numpy as np
from gymnasium.spaces import Box, Discrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from leibo2017.envs.grid_utils import NUM_ACTIONS, OBS_SHAPE

AGENT_IDS = ["player-0", "player-1"]
_FLAT_OBS_DIM = int(np.prod(OBS_SHAPE))


def _split_info(info: dict) -> list[dict]:
    """Turn one step's env-level info dict into one dict per agent.

    A value that's a list/tuple with one entry per agent (e.g. Gathering's
    `beam_use_rate`, indexed [player0, player1]) is unpacked to each
    agent's own scalar; anything else (e.g. Wolfpack's genuinely global
    `captures`/`avg_wolves_per_capture`) is copied identically to every
    agent, since it isn't agent-specific to begin with.
    """
    per_agent = [{} for _ in AGENT_IDS]
    for key, value in info.items():
        if isinstance(value, (list, tuple)) and len(value) == len(AGENT_IDS):
            for i in range(len(AGENT_IDS)):
                per_agent[i][key] = value[i]
        else:
            for i in range(len(AGENT_IDS)):
                per_agent[i][key] = value
    return per_agent


class TwoPlayerRLlibEnv(MultiAgentEnv):
    """Wraps a 2-agent, list-indexed env (GatheringEnv or WolfpackEnv) for RLlib.

    `inner_env_factory` takes a single `seed` argument (may be None) and
    returns a freshly-constructed inner env -- this is what lets both
    RLlib's `env_config={"seed": ...}` (construction time) and
    `reset(seed=...)` (per-episode, standard Gymnasium contract) actually
    affect the wrapped env's RNG instead of silently doing nothing.
    """

    def __init__(self, inner_env_factory, seed=None):
        super().__init__()
        self._inner_env_factory = inner_env_factory
        self._env = inner_env_factory(seed)
        self.possible_agents = list(AGENT_IDS)
        self.agents = list(AGENT_IDS)
        # Flattened, not (3, 16, 21): RLlib's default Catalog treats any 3D
        # Box as an image and tries to pick a default CNN, which has no
        # preset for this shape (raises ValueError). Flattening sidesteps
        # that and matches this repo's own literal reading of Sec. 4 ("two
        # hidden layers with 32 units" -- an MLP, not a described conv
        # stack) -- see agents/dqn.py's docstring for the same choice.
        # float32 in [0, 1], not raw uint8: RLlib's default FC encoder
        # expects a float input dtype (it does not auto-normalize uint8
        # Box spaces the way an image-specific pipeline would), and
        # normalizing to [0, 1] matches this repo's own DQN input scaling
        # (see agents/dqn.py, which divides by 255.0 the same way).
        obs_space = Box(low=0.0, high=1.0, shape=(_FLAT_OBS_DIM,), dtype="float32")
        act_space = Discrete(NUM_ACTIONS)
        self.observation_spaces = {aid: obs_space for aid in AGENT_IDS}
        self.action_spaces = {aid: act_space for aid in AGENT_IDS}
        self.observation_space = obs_space
        self.action_space = act_space

    @staticmethod
    def _flatten(obs: np.ndarray) -> np.ndarray:
        return (obs.reshape(-1).astype(np.float32)) / 255.0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            # Reseed the same env instance's RNG rather than rebuilding it,
            # honoring Gymnasium/RLlib's per-episode reset(seed=...) contract.
            self._env.rng = np.random.default_rng(seed)
        obs = self._env.reset()
        obs_dict = {aid: self._flatten(o) for aid, o in zip(AGENT_IDS, obs)}
        return obs_dict, {aid: {} for aid in AGENT_IDS}

    def step(self, action_dict):
        actions = [action_dict[aid] for aid in AGENT_IDS]
        obs, rewards, done, info = self._env.step(actions)
        obs_dict = {aid: self._flatten(o) for aid, o in zip(AGENT_IDS, obs)}
        reward_dict = {aid: r for aid, r in zip(AGENT_IDS, rewards)}
        terminations = {aid: False for aid in AGENT_IDS}
        terminations["__all__"] = False
        truncations = {aid: bool(done) for aid in AGENT_IDS}
        truncations["__all__"] = bool(done)
        info_dict = {aid: d for aid, d in zip(AGENT_IDS, _split_info(info))}
        return obs_dict, reward_dict, terminations, truncations, info_dict


def make_gathering_rllib_env(config: dict | None = None) -> TwoPlayerRLlibEnv:
    from leibo2017.envs.gathering import GatheringConfig, GatheringEnv

    config = dict(config or {})
    seed = config.pop("seed", None)
    return TwoPlayerRLlibEnv(
        lambda s: GatheringEnv(GatheringConfig(**config), rng=np.random.default_rng(s)), seed
    )


def make_wolfpack_rllib_env(config: dict | None = None) -> TwoPlayerRLlibEnv:
    from leibo2017.envs.wolfpack import WolfpackConfig, WolfpackEnv

    config = dict(config or {})
    seed = config.pop("seed", None)
    return TwoPlayerRLlibEnv(
        lambda s: WolfpackEnv(WolfpackConfig(**config), rng=np.random.default_rng(s)), seed
    )
