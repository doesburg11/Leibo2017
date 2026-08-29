"""Shared independent-DQN training loop, used by both Gathering and Wolfpack.

Both games expose the same minimal env interface (`reset()`, `step(actions)`
-> (obs, rewards, done, info)) and both are trained the same way in the
paper: two independent DQN learners, each training on its own experience
each step, no communication. This loop is the one piece of machinery shared
between them; everything game-specific (map, reward, social metric) lives
in the env classes themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from leibo2017.agents.dqn import DQNAgent


@dataclass
class TrainingResult:
    agents: list[DQNAgent]
    episode_returns: list[list[float]] = field(default_factory=list)  # per-episode, per-agent
    info_history: list[dict] = field(default_factory=list)  # info dict at the end of each episode


def train_independent_dqn(env_factory, agents: list[DQNAgent], total_steps: int, log_every_episode: bool = True) -> TrainingResult:
    """Run `total_steps` environment steps of independent-DQN training.

    `env_factory()` must return a fresh env with `.reset()`/`.step()`. A
    fresh env instance is created for every episode (Sec. 4: "Each episode
    lasted for 1,000 steps"), reusing the same agents/buffers across
    episodes, matching the paper's continual online training.
    """
    result = TrainingResult(agents=agents)
    steps_done = 0
    while steps_done < total_steps:
        env = env_factory()
        obs = env.reset()
        done = False
        ep_returns = [0.0] * len(agents)
        info = {}
        while not done and steps_done < total_steps:
            actions = [agent.act(o) for agent, o in zip(agents, obs)]
            next_obs, rewards, done, info = env.step(actions)
            for i, agent in enumerate(agents):
                agent.observe(obs[i], actions[i], rewards[i], next_obs[i], done)
                agent.train_step()
                ep_returns[i] += rewards[i]
            obs = next_obs
            steps_done += 1
        if log_every_episode:
            result.episode_returns.append(ep_returns)
            result.info_history.append(info)
    return result
