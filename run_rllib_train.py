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
from pathlib import Path

import ray
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env

from leibo2017.envs.rllib_wrappers import AGENT_IDS, make_gathering_rllib_env, make_wolfpack_rllib_env

ENV_FACTORIES = {"gathering": make_gathering_rllib_env, "wolfpack": make_wolfpack_rllib_env}


def build_config(game: str, algo: str, episode_length: int, hidden_size: int, num_env_runners: int = 0, num_gpus_per_learner: int = 0):
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
        .env_runners(num_env_runners=num_env_runners, rollout_fragment_length=episode_length)
        .multi_agent(policies=policies, policy_mapping_fn=lambda agent_id, *a, **kw: agent_id)
        .rl_module(model_config={"fcnet_hiddens": [hidden_size, hidden_size]})
        .learners(num_gpus_per_learner=num_gpus_per_learner)
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
    ap.add_argument("--num-env-runners", type=int, default=0, help="Parallel remote rollout workers (each its own env instance + CPU). 0 = single local process (default).")
    ap.add_argument("--gpu", action="store_true", help="Train the (tiny) network on GPU instead of CPU.")
    ap.add_argument("--checkpoint-dir", type=str, default=None,
                     help="Where to save the trained policies (algo.save()) when training finishes. "
                          "Defaults to output/rllib_checkpoints/<game>_<algo>/. "
                          "Ray's own ~/ray_results/<timestamp>/ trial dir is NOT a checkpoint -- "
                          "it stays empty unless you go through this (or a ray.tune.Tuner run).")
    args = ap.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir or f"output/rllib_checkpoints/{args.game}_{args.algo.lower()}").resolve()

    ray.init(include_dashboard=False, logging_level="ERROR")
    algo = None
    try:
        config = build_config(
            args.game, args.algo, args.episode_length, args.hidden_size,
            num_env_runners=args.num_env_runners, num_gpus_per_learner=1 if args.gpu else 0,
        )
        algo = config.build()
        for i in range(1, args.iterations + 1):
            result = algo.train()
            env_runners = result.get("env_runners", {})
            per_agent = env_runners.get("agent_episode_returns_mean", {})
            print(f"iter {i:3d}  episode_return_mean={env_runners.get('episode_return_mean'):.3f}  "
                  f"per_agent={per_agent}")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        algo.save(checkpoint_dir=str(checkpoint_dir))
        print(f"Saved checkpoint to {checkpoint_dir}")
    finally:
        # Releases the algorithm's remote env-runner/learner actors and GPU
        # state gracefully -- with --num-env-runners/--gpu it owns more than
        # just local-process memory, so this matters beyond ray.shutdown()
        # alone (which force-tears-down anything still attached).
        if algo is not None:
            algo.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
