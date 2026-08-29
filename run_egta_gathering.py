#!/usr/bin/env python3
"""Empirical game-theoretic analysis for Gathering (Sec. 5.1, Fig. 5-6).

The paper's clearly-specified procedure: train a pool of policies at a
high-abundance/low-conflict-cost setting (Pi^C) and a pool at a
low-abundance/high-conflict-cost setting (Pi^D), then sample pairs from
each to estimate one aggregate (R, P, S, T) and classify it.

What Fig. 6 additionally shows -- one scattered point per (N_apple,
N_tagged) condition swept in Experiment 1, spread across all dilemma
types -- is NOT precisely specified: the paper never states how a full
(R, P, S, T) tuple, rather than a single scalar aggressiveness, is derived
for each individual swept condition. `--sweep` reproduces the visual
*shape* of Fig. 6 using an explicit, disclosed extension of the paper's
own method: for each condition, that condition's own trained pair supplies
R (self-play) while a fixed reference Pi^D pool (trained at the sweep's
most defecting corner) supplies P/S/T by cross-play. This is an
interpretation, not a verified reproduction of DeepMind's exact procedure
-- see the top-level README's "Blind spots" section.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from leibo2017.analysis.egta import PayoffEstimate, estimate_payoff_matrix, play_episode
from leibo2017.envs.gathering import GatheringConfig, GatheringEnv
from leibo2017.plotting.figures import plot_egta_scatter
from leibo2017.training.train_gathering import run_gathering_training


def env_factory():
    return GatheringEnv(GatheringConfig())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool-size", type=int, default=4, help="Trained policies per pool (Pi^C / Pi^D).")
    ap.add_argument("--total-steps", type=int, default=20_000, help="Paper uses 40,000,000 per policy.")
    ap.add_argument("--n-egta-samples", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default="output/egta_gathering")
    ap.add_argument("--sweep", action="store_true", help="Also produce the Fig. 6-style multi-point scatter.")
    ap.add_argument("--n-apple-values", type=int, nargs="+", default=[5, 10, 20, 40, 80])
    ap.add_argument("--n-tagged-values", type=int, nargs="+", default=[5, 10, 25, 50, 100])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("Training Pi^C pool (high abundance / low conflict-cost) ...")
    pool_c = [
        run_gathering_training(n_apple=80, n_tagged=100, total_steps=args.total_steps, seed=int(rng.integers(2**31)))["agents"][0]
        for _ in range(args.pool_size)
    ]
    print("Training Pi^D pool (low abundance / high conflict-cost) ...")
    pool_d = [
        run_gathering_training(n_apple=5, n_tagged=5, total_steps=args.total_steps, seed=int(rng.integers(2**31)))["agents"][0]
        for _ in range(args.pool_size)
    ]

    estimate = estimate_payoff_matrix(env_factory, pool_c, pool_d, args.n_egta_samples, rng)
    print(f"R={estimate.R:.2f} P={estimate.P:.2f} S={estimate.S:.2f} T={estimate.T:.2f} "
          f"fear={estimate.fear:.2f} greed={estimate.greed:.2f} -> {estimate.classify()}")

    summary = {"R": estimate.R, "P": estimate.P, "S": estimate.S, "T": estimate.T,
               "fear": estimate.fear, "greed": estimate.greed, "classification": estimate.classify()}

    points = [(estimate.fear, estimate.greed, estimate.classify())]

    if args.sweep:
        print("Sweeping per-condition points for the Fig. 6-style scatter (see docstring: this is an "
              "explicitly-flagged interpretation, not a verified reproduction of the paper's exact method) ...")
        sweep_records = []
        for n_apple in args.n_apple_values:
            for n_tagged in args.n_tagged_values:
                cell = run_gathering_training(n_apple=n_apple, n_tagged=n_tagged, total_steps=args.total_steps,
                                               seed=int(rng.integers(2**31)))
                own = cell["agents"]
                r1, r2 = play_episode(env_factory(), (own[0], own[1]))
                d = pool_d[rng.integers(len(pool_d))]
                p1, p2 = play_episode(env_factory(), (d, d))
                s1, t2 = play_episode(env_factory(), (own[0], d))
                t1, s2 = play_episode(env_factory(), (d, own[1]))
                est = PayoffEstimate(R=np.mean([r1, r2]), P=np.mean([p1, p2]), S=np.mean([s1, s2]), T=np.mean([t1, t2]))
                points.append((est.fear, est.greed, est.classify()))
                sweep_records.append({"n_apple": n_apple, "n_tagged": n_tagged, "fear": est.fear,
                                       "greed": est.greed, "classification": est.classify()})
        summary["sweep"] = sweep_records

    with open(out_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    plot_egta_scatter(points, title="Gathering empirical payoff matrices (Fig. 6 left)",
                       out_path=str(out_dir / "fig6_gathering_scatter.png"))
    print(f"Wrote {out_dir / 'fig6_gathering_scatter.png'} and {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
