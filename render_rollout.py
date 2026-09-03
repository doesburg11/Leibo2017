#!/usr/bin/env python3
"""Render a played-out Gathering or Wolfpack episode to an animated GIF --
the graphics-utility counterpart to the sibling SequentialSocialDilemmas
repo's episode-rollout video rendering (see README's "Does this have the
same graphics utilities as the original?" discussion).

By default this trains briefly (--total-steps, much smaller than the
run_experiment*.py smoke-test budgets) then rolls out one greedy episode
and renders it; pass --random-policy to skip training entirely and just
render a random rollout, useful as a fast sanity check that the env/
rendering pipeline works.
"""
import argparse
from pathlib import Path

import numpy as np

from leibo2017.agents.dqn import DQNAgent, DQNConfig
from leibo2017.envs.gathering import GatheringConfig, GatheringEnv
from leibo2017.envs.wolfpack import WolfpackConfig, WolfpackEnv
from leibo2017.plotting.video import save_rollout_gif
from leibo2017.training.loop import train_independent_dqn


def make_env(game: str, episode_length: int, rng: np.random.Generator):
    if game == "gathering":
        return GatheringEnv(GatheringConfig(episode_length=episode_length), rng=rng)
    return WolfpackEnv(WolfpackConfig(episode_length=episode_length), rng=rng)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--game", choices=["gathering", "wolfpack"], default="gathering")
    ap.add_argument("--total-steps", type=int, default=5_000, help="Training budget before rollout; ignored with --random-policy.")
    ap.add_argument("--episode-length", type=int, default=200, help="Steps rendered into the gif (independent of training's own 1000-step episodes).")
    ap.add_argument("--random-policy", action="store_true", help="Skip training; render a random-action rollout instead.")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--scale", type=int, default=8, help="Pixel upscale factor per grid cell.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default="output/render_rollout")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    agents = None
    if not args.random_policy:
        def env_factory():
            return make_env(args.game, 1000, np.random.default_rng(rng.integers(2**31)))

        agents = [DQNAgent(DQNConfig(seed=args.seed + i)) for i in range(2)]
        train_independent_dqn(env_factory, agents, args.total_steps)

    env = make_env(args.game, args.episode_length, np.random.default_rng(rng.integers(2**31)))
    action_rng = np.random.default_rng(args.seed)
    obs = env.reset()
    frames = [env.render()]
    done = False
    while not done:
        if agents is None:
            actions = action_rng.integers(env.num_actions, size=env.num_agents)
        else:
            actions = [agent.act(o, greedy=True) for agent, o in zip(agents, obs)]
        obs, rewards, done, info = env.step(actions)
        frames.append(env.render())

    out_path = out_dir / f"{args.game}_rollout.gif"
    save_rollout_gif(frames, str(out_path), fps=args.fps, scale=args.scale)
    print(f"Wrote {out_path} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
