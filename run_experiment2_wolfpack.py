#!/usr/bin/env python3
"""Experiment 2 (Sec. 5.2, Fig. 4 bottom): sweep Wolfpack's capture radius x
group-capture bonus (r_team / r_lone) and plot the resulting
"two minus average wolves per capture" heatmap (higher = more cooperative).

As in Experiment 1, the paper's own 40,000,000-steps-per-cell scale is not
attempted by default -- see README. Pass --total-steps to scale up.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from leibo2017.plotting.figures import plot_heatmap
from leibo2017.training.train_wolfpack import run_wolfpack_training


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--radius-values", type=float, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--bonus-values", type=float, nargs="+", default=[1, 2, 4, 8, 16],
                     help="r_team values; r_lone is fixed at 1.0, so this is the r_team/r_lone ratio.")
    ap.add_argument("--total-steps", type=int, default=20_000, help="Paper uses 40,000,000 per cell.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default="output/experiment2_wolfpack")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = np.zeros((len(args.radius_values), len(args.bonus_values)))
    records = []
    start = time.time()
    for i, radius in enumerate(args.radius_values):
        for j, bonus in enumerate(args.bonus_values):
            result = run_wolfpack_training(
                capture_radius=radius, r_team=bonus, r_lone=1.0, total_steps=args.total_steps,
                seed=args.seed + i * 100 + j,
            )
            lone_wolf_rate = 2.0 - result["wolves_per_capture"]  # paper's Fig. 4 quantity, in [0, 1]
            grid[i, j] = lone_wolf_rate
            records.append({"radius": radius, "r_team": bonus, "wolves_per_capture": result["wolves_per_capture"],
                             "lone_wolf_rate": lone_wolf_rate})
            print(f"radius={radius:4.1f} r_team={bonus:5.1f} -> wolves/capture={result['wolves_per_capture']:.3f} "
                  f"(elapsed {time.time()-start:.0f}s)")

    with open(out_dir / "results.json", "w") as f:
        json.dump(records, f, indent=2)

    plot_heatmap(
        grid, x_labels=args.bonus_values, y_labels=args.radius_values,
        x_name="r_team (group capture benefit)", y_name="capture radius",
        cbar_label="Lone-wolf capture rate (2 - avg wolves/capture)",
        title="Wolfpack: cooperation vs. radius/group-benefit (Fig. 4 bottom)",
        out_path=str(out_dir / "fig4_wolfpack_heatmap.png"),
    )
    print(f"Wrote {out_dir / 'fig4_wolfpack_heatmap.png'} and {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
