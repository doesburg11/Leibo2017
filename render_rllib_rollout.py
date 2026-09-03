#!/usr/bin/env python3
"""Evaluate and visualize a policy trained by `run_rllib_train.py`.

Trains the same PPO/DQN config on Gathering/Wolfpack (see `build_config` in
`run_rllib_train.py`), then rolls out one greedy (`explore=False`) episode
using the trained `RLModule`s directly -- the new API stack has no working
`Algorithm.compute_single_action` for multi-agent envs (it raises
`AttributeError` on `MultiAgentEnvRunner`), so actions are computed via
`RLModule.forward_inference` + the module's own action-distribution class
instead. Prints per-agent return and the env's social-behavior info
(`captures`/`avg_wolves_per_capture` for Wolfpack, `beam_use_rate` for
Gathering) for the rendered episode, and writes it to a GIF.

Requires `ray[rllib]` -- see `requirements-rllib.txt`.
"""
import argparse
from pathlib import Path

import numpy as np
import ray
import torch

from leibo2017.envs.rllib_wrappers import AGENT_IDS
from leibo2017.plotting.video import save_rollout_gif
from run_rllib_train import ENV_FACTORIES, build_config


def greedy_actions(module_dict, obs_dict):
    """Compute one greedy action per agent from its trained `RLModule`.

    DQN's `forward_inference` already returns the exploit (argmax-Q) action
    directly under `Columns.ACTIONS` -- no distribution involved. PPO (and
    other policy-gradient modules) instead return `action_dist_inputs`
    (logits), which must be turned into a distribution and made
    deterministic before sampling. Branching on which key is present covers
    both `--algo` choices this script exposes.
    """
    actions = {}
    for aid, obs in obs_dict.items():
        module = module_dict[aid]
        batch = {"obs": torch.from_numpy(np.asarray(obs, dtype=np.float32)[None, :])}
        fwd_out = module.forward_inference(batch)
        if "actions" in fwd_out:
            actions[aid] = int(fwd_out["actions"][0])
        else:
            dist_cls = module.get_inference_action_dist_cls()
            dist = dist_cls.from_logits(fwd_out["action_dist_inputs"])
            actions[aid] = int(dist.to_deterministic().sample()[0])
    return actions


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--game", choices=["gathering", "wolfpack"], default="wolfpack")
    ap.add_argument("--algo", choices=["PPO", "DQN"], default="PPO")
    ap.add_argument("--iterations", type=int, default=10, help="Training iterations before the eval rollout.")
    ap.add_argument("--episode-length", type=int, default=200, help="Training episode length.")
    ap.add_argument("--render-episode-length", type=int, default=None,
                     help="Steps rendered into the gif; defaults to --episode-length.")
    ap.add_argument("--hidden-size", type=int, default=32)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--scale", type=int, default=8, help="Pixel upscale factor per grid cell.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default="output/render_rllib_rollout")
    args = ap.parse_args()

    render_episode_length = args.render_episode_length or args.episode_length
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ray.init(include_dashboard=False, logging_level="ERROR")
    try:
        config = build_config(args.game, args.algo, args.episode_length, args.hidden_size)
        algo = config.build()
        for i in range(1, args.iterations + 1):
            result = algo.train()
            env_runners = result.get("env_runners", {})
            print(f"iter {i:3d}  episode_return_mean={env_runners.get('episode_return_mean'):.3f}")

        module_dict = algo.env_runner.module

        env = ENV_FACTORIES[args.game]({"episode_length": render_episode_length, "seed": args.seed})
        obs_dict, _ = env.reset(seed=args.seed)
        frames = [env.render()]
        per_agent_return = {aid: 0.0 for aid in AGENT_IDS}
        last_info = {}
        done = False
        while not done:
            action_dict = greedy_actions(module_dict, obs_dict)
            obs_dict, reward_dict, terminations, truncations, info_dict = env.step(action_dict)
            for aid in AGENT_IDS:
                per_agent_return[aid] += reward_dict[aid]
            last_info = {aid: info_dict[aid] for aid in AGENT_IDS}
            frames.append(env.render())
            done = terminations["__all__"] or truncations["__all__"]
    finally:
        ray.shutdown()

    out_path = out_dir / f"{args.game}_{args.algo.lower()}_rollout.gif"
    save_rollout_gif(frames, str(out_path), fps=args.fps, scale=args.scale)
    print(f"Wrote {out_path} ({len(frames)} frames)")
    print(f"eval episode return per_agent={per_agent_return}  info={last_info}")


if __name__ == "__main__":
    main()
