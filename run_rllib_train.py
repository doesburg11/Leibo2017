#!/usr/bin/env python3
"""Optional: train Gathering/Wolfpack with Ray RLlib instead of the paper's
own independent-DQN method.

This is purely additive -- see the README's "Optional: RLlib backend"
section. `leibo2017/agents/dqn.py` and every other `run_*.py` script are
untouched; this script exists to answer a different question ("does a
standard, well-tuned RL library reach different conclusions than the
paper's own simple setup on the exact same environments?"), not to
reproduce Fig. 4/6/7 -- it reports raw per-agent episode return, not the
paper's beam-use-rate / wolves-per-capture social-behavior metrics (doing
that from RLlib's training loop would need a custom callback, which isn't
built here).

Requires `ray[rllib]` (`pip install -r requirements-rllib.txt`), not part
of the base install.
"""
import argparse

import ray
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env

from leibo2017.envs.rllib_wrappers import AGENT_IDS, make_gathering_rllib_env, make_wolfpack_rllib_env

ENV_FACTORIES = {"gathering": make_gathering_rllib_env, "wolfpack": make_wolfpack_rllib_env}


def build_config(game: str, algo: str, episode_length: int, hidden_size: int):
    env_name = f"{game}_env"
    register_env(env_name, ENV_FACTORIES[game])
    probe_env = ENV_FACTORIES[game]()
    policies = {
        aid: PolicySpec(observation_space=probe_env.observation_space, action_space=probe_env.action_space, config={})
        for aid in AGENT_IDS
    }

    config_cls = {"PPO": PPOConfig, "DQN": DQNConfig}[algo]
    config = (
        config_cls()
        .framework("torch")
        .environment(env=env_name, env_config={"episode_length": episode_length})
        .env_runners(num_env_runners=0, rollout_fragment_length=episode_length)
        .multi_agent(policies=policies, policy_mapping_fn=lambda agent_id, *a, **kw: agent_id)
        .rl_module(model_config={"fcnet_hiddens": [hidden_size, hidden_size]})
    )
    if algo == "PPO":
        config = config.training(train_batch_size=max(200, 4 * episode_length), minibatch_size=64)
    else:
        config = config.training(
            train_batch_size=32,
            replay_buffer_config={"type": "MultiAgentEpisodeReplayBuffer", "capacity": 100_000},
        )
    return config


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--game", choices=["gathering", "wolfpack"], default="gathering")
    ap.add_argument("--algo", choices=["PPO", "DQN"], default="PPO")
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--episode-length", type=int, default=200)
    ap.add_argument("--hidden-size", type=int, default=32, help="Matches the paper's Sec. 4 default (32 units).")
    args = ap.parse_args()

    ray.init(include_dashboard=False, logging_level="ERROR")
    try:
        config = build_config(args.game, args.algo, args.episode_length, args.hidden_size)
        algo = config.build()
        for i in range(1, args.iterations + 1):
            result = algo.train()
            env_runners = result.get("env_runners", {})
            per_agent = env_runners.get("agent_episode_returns_mean", {})
            print(f"iter {i:3d}  episode_return_mean={env_runners.get('episode_return_mean'):.3f}  "
                  f"per_agent={per_agent}")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
