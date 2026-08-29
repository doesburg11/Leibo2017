#!/usr/bin/env python3
"""Empirical game-theoretic analysis for Wolfpack (Sec. 5.2, Fig. 5-6).

Mirrors run_egta_gathering.py: Pi^C = pool trained at high capture-radius /
high group-bonus, Pi^D = pool trained at low radius / low bonus. See that
script's docstring (and the README "Blind spots" section) for the same
caveat about `--sweep`'s per-condition scatter being an explicit,
disclosed interpretation rather than a verified reproduction.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from leibo2017.analysis.egta import PayoffEstimate, estimate_payoff_matrix, play_episode
from leibo2017.envs.wolfpack import WolfpackConfig, WolfpackEnv
from leibo2017.plotting.figures import plot_egta_scatter
from leibo2017.training.train_wolfpack import run_wolfpack_training


def env_factory():
    return WolfpackEnv(WolfpackConfig())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool-size", type=int, default=4)
    ap.add_argument("--total-steps", type=int, default=20_000, help="Paper uses 40,000,000 per policy.")
    ap.add_argument("--n-egta-samples", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default="output/egta_wolfpack")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--radius-values", type=float, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--bonus-values", type=float, nargs="+", default=[1, 2, 4, 8, 16])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("Training Pi^C pool (high radius / high group bonus) ...")
    pool_c = [
        run_wolfpack_training(capture_radius=5.0, r_team=16.0, total_steps=args.total_steps,
                               seed=int(rng.integers(2**31)))["agents"][0]
        for _ in range(args.pool_size)
    ]
    print("Training Pi^D pool (low radius / low group bonus) ...")
    pool_d = [
        run_wolfpack_training(capture_radius=1.0, r_team=1.0, total_steps=args.total_steps,
                               seed=int(rng.integers(2**31)))["agents"][0]
        for _ in range(args.pool_size)
    ]

    estimate = estimate_payoff_matrix(env_factory, pool_c, pool_d, args.n_egta_samples, rng)
    print(f"R={estimate.R:.2f} P={estimate.P:.2f} S={estimate.S:.2f} T={estimate.T:.2f} "
          f"fear={estimate.fear:.2f} greed={estimate.greed:.2f} -> {estimate.classify()}")

    summary = {"R": estimate.R, "P": estimate.P, "S": estimate.S, "T": estimate.T,
               "fear": estimate.fear, "greed": estimate.greed, "classification": estimate.classify()}
    points = [(estimate.fear, estimate.greed, estimate.classify())]

    if args.sweep:
        print("Sweeping per-condition points for the Fig. 6-style scatter (interpretation; see docstring) ...")
        sweep_records = []
        for radius in args.radius_values:
            for bonus in args.bonus_values:
                cell = run_wolfpack_training(capture_radius=radius, r_team=bonus, total_steps=args.total_steps,
                                              seed=int(rng.integers(2**31)))
                own = cell["agents"]
                r1, r2 = play_episode(env_factory(), (own[0], own[1]))
                d = pool_d[rng.integers(len(pool_d))]
                p1, p2 = play_episode(env_factory(), (d, d))
                s1, t2 = play_episode(env_factory(), (own[0], d))
                t1, s2 = play_episode(env_factory(), (d, own[1]))
                est = PayoffEstimate(R=np.mean([r1, r2]), P=np.mean([p1, p2]), S=np.mean([s1, s2]), T=np.mean([t1, t2]))
                points.append((est.fear, est.greed, est.classify()))
                sweep_records.append({"radius": radius, "r_team": bonus, "fear": est.fear,
                                       "greed": est.greed, "classification": est.classify()})
        summary["sweep"] = sweep_records

    with open(out_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    plot_egta_scatter(points, title="Wolfpack empirical payoff matrices (Fig. 6 right)",
                       out_path=str(out_dir / "fig6_wolfpack_scatter.png"))
    print(f"Wrote {out_dir / 'fig6_wolfpack_scatter.png'} and {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
