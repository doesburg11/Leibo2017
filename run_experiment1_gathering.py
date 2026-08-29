#!/usr/bin/env python3
"""Experiment 1 (Sec. 5.1, Fig. 4 top): sweep Gathering's abundance
(n_apple) x conflict-cost (n_tagged) and plot the resulting aggressiveness
(beam-use rate) heatmap.

The paper trains each grid cell for 40,000,000 steps. That is not
reproduced at that scale here -- see the top-level README's "Running at
paper scale vs. as a smoke test" section. Defaults below run a fast
smoke test; pass --total-steps 40000000 (and expect it to take a very
long time on a single machine) to attempt the paper's own scale.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from leibo2017.plotting.figures import plot_heatmap
from leibo2017.training.train_gathering import run_gathering_training


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-apple-values", type=int, nargs="+", default=[5, 10, 20, 40, 80])
    ap.add_argument("--n-tagged-values", type=int, nargs="+", default=[5, 10, 25, 50, 100])
    ap.add_argument("--total-steps", type=int, default=20_000, help="Paper uses 40,000,000 per cell.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default="output/experiment1_gathering")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = np.zeros((len(args.n_apple_values), len(args.n_tagged_values)))
    records = []
    start = time.time()
    for i, n_apple in enumerate(args.n_apple_values):
        for j, n_tagged in enumerate(args.n_tagged_values):
            result = run_gathering_training(
                n_apple=n_apple, n_tagged=n_tagged, total_steps=args.total_steps,
                seed=args.seed + i * 100 + j,
            )
            grid[i, j] = result["aggressiveness"]
            records.append({"n_apple": n_apple, "n_tagged": n_tagged, "aggressiveness": result["aggressiveness"]})
            print(f"n_apple={n_apple:4d} n_tagged={n_tagged:4d} -> aggressiveness={result['aggressiveness']:.4f} "
                  f"(elapsed {time.time()-start:.0f}s)")

    with open(out_dir / "results.json", "w") as f:
        json.dump(records, f, indent=2)

    plot_heatmap(
        grid, x_labels=args.n_tagged_values, y_labels=args.n_apple_values,
        x_name="N_tagged (conflict-cost)", y_name="N_apple (abundance)",
        cbar_label="Aggressiveness (beam-use rate)",
        title="Gathering: aggressiveness vs. abundance/conflict-cost (Fig. 4 top)",
        out_path=str(out_dir / "fig4_gathering_heatmap.png"),
    )
    print(f"Wrote {out_dir / 'fig4_gathering_heatmap.png'} and {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
