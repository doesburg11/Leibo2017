"""Train a pair of independent DQN agents on Wolfpack, sweeping capture
radius and group-capture bonus (`r_team`/`r_lone`) -- Sec. 5.2 / Fig. 4
(bottom)."""

from __future__ import annotations

import numpy as np

from leibo2017.agents.dqn import DQNAgent, DQNConfig
from leibo2017.envs.wolfpack import WolfpackConfig, WolfpackEnv
from leibo2017.training.loop import train_independent_dqn


def run_wolfpack_training(
    capture_radius: float,
    r_team: float,
    total_steps: int,
    r_lone: float = 1.0,
    dqn_kwargs: dict | None = None,
    seed: int | None = None,
    tail_episodes: int = 10,
) -> dict:
    """Train, then report mean wolves-per-capture over the last
    `tail_episodes` episodes -- the paper's cooperation metric (Sec. 5.2,
    Fig. 4: "two minus the average number of wolves per capture"; we report
    the raw average-wolves-per-capture in [1, 2] and let plotting code apply
    the "two minus" transform, since that's a display choice, not a
    property of the environment)."""
    dqn_kwargs = dict(dqn_kwargs or {})
    rng = np.random.default_rng(seed)

    def env_factory():
        cfg = WolfpackConfig(capture_radius=capture_radius, r_team=r_team, r_lone=r_lone)
        return WolfpackEnv(cfg, rng=np.random.default_rng(rng.integers(2**31)))

    agents = [
        DQNAgent(DQNConfig(seed=None if seed is None else seed + i, **dqn_kwargs))
        for i in range(2)
    ]
    result = train_independent_dqn(lambda: env_factory(), agents, total_steps)

    tail = result.info_history[-tail_episodes:] if result.info_history else []
    rates = [info["avg_wolves_per_capture"] for info in tail if info.get("captures", 0) > 0]
    wolves_per_capture = float(np.mean(rates)) if rates else float("nan")
    return {
        "capture_radius": capture_radius,
        "r_team": r_team,
        "r_lone": r_lone,
        "wolves_per_capture": wolves_per_capture,
        "agents": agents,
        "episode_returns": result.episode_returns,
    }
