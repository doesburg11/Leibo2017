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

# RLlib's own DQNConfig defaults for `epsilon`'s decay-to-floor timestep and
# `target_network_update_freq` -- both counted in *global* env steps sampled
# across every env runner combined (ray.rllib.env.multi_agent_env_runner
# passes a `global_env_steps_lifetime` into exploration, and
# NUM_ENV_STEPS_SAMPLED_LIFETIME into the target-update check). Left as
# RLlib's raw constants (10_000, 500), both implicitly assume the
# single-env-runner steps-per-`algo.train()` rate; --num-env-runners changes
# that rate, so unscaled they complete faster (in iteration terms) than
# RLlib intended once parallel env runners are turned on.
#
# The naive fix is scaling by num_env_runners directly, but that overshoots:
# DQN's own `min_sample_timesteps_per_iteration=1000` means one
# `algo.train()` call keeps sampling rollout_fragment_length-sized rounds
# from every env runner until at least 1000 global steps are collected, not
# exactly `num_env_runners * episode_length`. E.g. at episode_length=200,
# num_env_runners=0/1 collects 1000 steps/iteration already (5 rounds of
# 200, to clear the 1000-step floor) -- not 200 -- and num_env_runners=30
# collects 6000 steps/iteration (one round of 30*200, comfortably over the
# floor) -- not 30x that 1000, only 6x (measured empirically: 1000 vs 6000
# steps after one algo.train() call at episode_length=200). `_steps_per_dqn_iteration()`
# below reproduces that same ceil-to-the-sampling-floor arithmetic so the
# scale factor matches actual steps-per-iteration instead of num_env_runners.
_DQN_DEFAULT_EPSILON_TIMESTEPS = 10_000
_DQN_DEFAULT_TARGET_NETWORK_UPDATE_FREQ = 500
_DQN_MIN_SAMPLE_TIMESTEPS_PER_ITERATION = 1_000  # ray.rllib.algorithms.dqn.dqn.DQNConfig's own default


def _steps_per_dqn_iteration(num_env_runners: int, episode_length: int) -> int:
    per_round = max(num_env_runners, 1) * episode_length
    rounds = -(-_DQN_MIN_SAMPLE_TIMESTEPS_PER_ITERATION // per_round)  # ceil division
    return rounds * per_round


# With training_intensity left at its default (None), DQN's calculate_rr_weights()
# (ray/rllib/algorithms/dqn/dqn.py) always returns [1, 1]: exactly ONE
# replay-sample-and-gradient-update per algo.train() call, regardless of
# train_batch_size, episode_length, or num_env_runners -- confirmed
# empirically (num_module_steps_trained_lifetime == train_batch_size after
# one algo.train() call). To get more than one update per algo.train()
# call, training_intensity must be set explicitly to
# `updates_per_iteration * native_ratio`, where native_ratio is RLlib's own
# train_batch_size-vs-collected-data ratio (see calculate_rr_weights).
#
# A prior version of this code tried to additionally divide by
# `ceil(min_sample_timesteps_per_iteration / (episode_length *
# max(num_env_runners+1, 1)))`, reasoning that Algorithm._run_one_training_
# iteration() calling training_step() repeatedly until that 1000-step
# floor is met (5x at num_env_runners=0, episode_length=200) would apply
# the same training_intensity-derived update count on each call, inflating
# the total. That reasoning turned out to be wrong: calculate_rr_weights
# is effectively evaluated once for the whole algo.train() call (verified
# empirically -- at num_env_runners=0, updates_per_iteration=10 produced
# num_module_steps_trained_lifetime of exactly 10*train_batch_size, not 2x
# or 50x it), so no such adjustment is needed; the plain formula below
# already gives an exact updates_per_iteration-many gradient updates per
# algo.train() call, at any num_env_runners.
def _dqn_training_intensity(updates_per_iteration: int, train_batch_size: int, episode_length: int, num_env_runners: int) -> float:
    native_ratio = train_batch_size / (episode_length * max(num_env_runners + 1, 1))
    return updates_per_iteration * native_ratio


def build_config(game: str, algo: str, episode_length: int, hidden_size: int, num_env_runners: int = 0, num_gpus_per_learner: int = 0, train_batch_size: int | None = None, epsilon_timesteps: int | None = None, target_network_update_freq: int | None = None, updates_per_iteration: int | None = None):
    if updates_per_iteration is not None and updates_per_iteration < 1:
        raise ValueError(f"updates_per_iteration must be >= 1, got {updates_per_iteration}")
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
        config = config.training(
            train_batch_size=train_batch_size or max(200, 4 * episode_length), minibatch_size=64
        )
    else:
        runner_scale = _steps_per_dqn_iteration(num_env_runners, episode_length) / _steps_per_dqn_iteration(0, episode_length)
        eps_timesteps = epsilon_timesteps if epsilon_timesteps is not None else round(_DQN_DEFAULT_EPSILON_TIMESTEPS * runner_scale)
        tnuf = target_network_update_freq if target_network_update_freq is not None else round(_DQN_DEFAULT_TARGET_NETWORK_UPDATE_FREQ * runner_scale)
        actual_train_batch_size = train_batch_size or 32
        training_intensity = None
        if updates_per_iteration is not None:
            training_intensity = _dqn_training_intensity(updates_per_iteration, actual_train_batch_size, episode_length, num_env_runners)
        config = config.training(
            train_batch_size=actual_train_batch_size,
            replay_buffer_config={"type": "MultiAgentEpisodeReplayBuffer", "capacity": 100_000},
            epsilon=[(0, 1.0), (eps_timesteps, 0.05)],
            target_network_update_freq=tnuf,
            training_intensity=training_intensity,
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
    ap.add_argument("--train-batch-size", type=int, default=None,
                     help="Samples per gradient update. Defaults to 32 for DQN, max(200, 4*episode_length) "
                          "for PPO. DQN's default of 32 is a low update-to-data ratio once "
                          "--num-env-runners collects far more env steps per iteration than that.")
    ap.add_argument("--epsilon-timesteps", type=int, default=None,
                     help="DQN only: global env steps over which epsilon decays 1.0 -> 0.05. Defaults to "
                          "10,000 scaled by the actual steps-per-algo.train()-iteration ratio vs. "
                          "--num-env-runners=0 (not num_env_runners itself -- DQN's own "
                          "min_sample_timesteps_per_iteration=1000 means e.g. 30 env runners collects "
                          "6000 steps/iteration, a 6x ratio, not 30x), preserving RLlib's own "
                          "single-runner decay cadence in iteration terms.")
    ap.add_argument("--target-network-update-freq", type=int, default=None,
                     help="DQN only: global env steps between target-network hard updates. Defaults to "
                          "500 scaled the same way as --epsilon-timesteps.")
    ap.add_argument("--updates-per-iteration", type=int, default=None,
                     help="DQN only: gradient updates per algo.train() call (int >= 1), via "
                          "training_intensity. RLlib's own default (unset) always does exactly 1 "
                          "update/iteration REGARDLESS of --train-batch-size, --episode-length, or "
                          "--num-env-runners -- e.g. 1000 iterations trains the network on only 1000 "
                          "total batches no matter how much data got collected. Set this explicitly to "
                          "actually use collected data for training instead of leaving most of it "
                          "unreplayed each iteration. Large values raise the replay ratio (samples "
                          "reused vs. freshly collected) -- can overfit/destabilize DQN if pushed too "
                          "high, same as any replay-ratio hyperparameter.")
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
            train_batch_size=args.train_batch_size,
            epsilon_timesteps=args.epsilon_timesteps, target_network_update_freq=args.target_network_update_freq,
            updates_per_iteration=args.updates_per_iteration,
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
