"""Train a pair of independent DQN agents on Gathering, sweeping abundance
(`n_apple`) and conflict-cost (`n_tagged`) -- Sec. 5.1 / Fig. 4 (top)."""

from __future__ import annotations

import numpy as np

from leibo2017.agents.dqn import DQNAgent, DQNConfig
from leibo2017.envs.gathering import GatheringConfig, GatheringEnv
from leibo2017.training.loop import train_independent_dqn


def run_gathering_training(
    n_apple: int,
    n_tagged: int,
    total_steps: int,
    dqn_kwargs: dict | None = None,
    seed: int | None = None,
    tail_episodes: int = 10,
) -> dict:
    """Train, then report mean beam-use rate over the last `tail_episodes`
    episodes -- the paper's aggressiveness / social-behavior metric
    (Sec. 5.1: "counted the number of beam actions ... normalized ... by
    the amount of time in which both agents were playing")."""
    dqn_kwargs = dict(dqn_kwargs or {})
    rng = np.random.default_rng(seed)

    def env_factory():
        cfg = GatheringConfig(n_apple=n_apple, n_tagged=n_tagged)
        return GatheringEnv(cfg, rng=np.random.default_rng(rng.integers(2**31)))

    agents = [
        DQNAgent(DQNConfig(seed=None if seed is None else seed + i, **dqn_kwargs))
        for i in range(2)
    ]
    result = train_independent_dqn(lambda: env_factory(), agents, total_steps)

    tail = result.info_history[-tail_episodes:] if result.info_history else []
    beam_rates = [np.mean(info["beam_use_rate"]) for info in tail if "beam_use_rate" in info]
    aggressiveness = float(np.mean(beam_rates)) if beam_rates else float("nan")
    return {
        "n_apple": n_apple,
        "n_tagged": n_tagged,
        "aggressiveness": aggressiveness,
        "agents": agents,
        "episode_returns": result.episode_returns,
    }
