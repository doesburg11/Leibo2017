#!/usr/bin/env python3
"""Experiment 3 (Sec. 5.3, Fig. 7): how DQN hyperparameters shift the
emergence of defection, for both games. Three factors, each compared at
two levels, against a single-axis sweep (scarcity for Gathering, group
benefit for Wolfpack):

  - discount:     0.99 vs 0.995              (Sec. 4 text default: 0.99)
  - batch size:   1e4 vs 1e5                 (Fig. 7 legend values; NOTE
                  Sec. 3.1's text instead says "batch sizes of 1e5 (our
                  default) and 1e6" -- an actual inconsistency between the
                  paper's own text and its Fig. 7 legend. We reproduce
                  Fig. 7's literal legend values here; see README.)
  - network size: 16 vs 64 hidden units      (Fig. 7 legend values; the
                  Sec. 4 text default of 32 is not one of the two compared
                  values, which is also just how the paper set up this
                  particular ablation.)

Other factors are held at the paper's stated defaults (discount 0.99 for
the batch-size/network-size panels, batch size 1e5 for the discount/
network-size panels, network size 32 for the discount/batch-size panels).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from leibo2017.plotting.figures import plot_ablation_lines
from leibo2017.training.train_gathering import run_gathering_training
from leibo2017.training.train_wolfpack import run_wolfpack_training

DEFAULT_DISCOUNT = 0.99
DEFAULT_BATCH = 100_000
DEFAULT_HIDDEN = 32


def sweep_gathering(x_values, factor: str, levels: tuple, total_steps: int, seed: int) -> dict:
    series = {}
    for level in levels:
        ys = []
        for x in x_values:
            n_apple = x  # "scarcity" axis: smaller n_apple = more abundant; we sweep n_apple directly
            kwargs = dict(discount=DEFAULT_DISCOUNT, batch_capacity=DEFAULT_BATCH, hidden_size=DEFAULT_HIDDEN)
            kwargs[factor] = level
            result = run_gathering_training(
                n_apple=n_apple, n_tagged=25, total_steps=total_steps, dqn_kwargs=kwargs, seed=seed,
            )
            ys.append(result["aggressiveness"])
        series[f"{factor}={level}"] = ys
    return series


def sweep_wolfpack(x_values, factor: str, levels: tuple, total_steps: int, seed: int) -> dict:
    series = {}
    for level in levels:
        ys = []
        for x in x_values:
            r_team = x  # "group benefit" axis
            kwargs = dict(discount=DEFAULT_DISCOUNT, batch_capacity=DEFAULT_BATCH, hidden_size=DEFAULT_HIDDEN)
            kwargs[factor] = level
            result = run_wolfpack_training(
                capture_radius=3.0, r_team=r_team, r_lone=1.0, total_steps=total_steps,
                dqn_kwargs=kwargs, seed=seed,
            )
            ys.append(2.0 - result["wolves_per_capture"])
        series[f"{factor}={level}"] = ys
    return series


FACTORS = {
    "discount": (0.99, 0.995),
    "batch_capacity": (10_000, 100_000),  # Fig. 7 legend: "Batch size: 1e+04" vs "1e+05"
    "hidden_size": (16, 64),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-points", type=int, default=5)
    ap.add_argument("--gathering-scarcity", type=int, nargs="+", default=[5, 15, 30, 60, 100])
    ap.add_argument("--wolfpack-benefit", type=float, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--total-steps", type=int, default=20_000, help="Paper uses 40,000,000 per curve point.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default="output/experiment3_agent_params")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for factor, levels in FACTORS.items():
        print(f"=== Gathering: {factor} ablation ===")
        g_series = sweep_gathering(args.gathering_scarcity, factor, levels, args.total_steps, args.seed)
        plot_ablation_lines(
            args.gathering_scarcity, g_series, x_name="N_apple (scarcity)", y_name="Aggressiveness",
            title=f"Gathering: {factor} ablation (Fig. 7 top)",
            out_path=str(out_dir / f"fig7_gathering_{factor}.png"),
        )
        all_results[f"gathering_{factor}"] = g_series

        print(f"=== Wolfpack: {factor} ablation ===")
        w_series = sweep_wolfpack(args.wolfpack_benefit, factor, levels, args.total_steps, args.seed)
        plot_ablation_lines(
            args.wolfpack_benefit, w_series, x_name="r_team (group benefit)", y_name="Lone-wolf capture rate",
            title=f"Wolfpack: {factor} ablation (Fig. 7 bottom)",
            out_path=str(out_dir / f"fig7_wolfpack_{factor}.png"),
        )
        all_results[f"wolfpack_{factor}"] = w_series

    with open(out_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Wrote figures and results.json to {out_dir}")


if __name__ == "__main__":
    main()
