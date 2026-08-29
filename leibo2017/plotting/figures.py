"""Reproduce the paper's figure types: heatmaps (Fig. 4), EGTA scatter
(Fig. 6), and parameter-ablation line plots (Fig. 7)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_heatmap(values: np.ndarray, x_labels, y_labels, x_name: str, y_name: str, cbar_label: str, title: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(values, origin="lower", aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_egta_scatter(points: list[tuple[float, float, str]], title: str, out_path: str) -> None:
    """`points` is a list of (fear, greed, class_label). Matches Fig. 6's
    fear=P-S (x) vs greed=T-R (y) quadrant scatter."""
    colors = {
        "Prisoner's Dilemma": "red",
        "Chicken": "blue",
        "Stag Hunt": "green",
        "Non-SSD (R<P)": "gray",
        "Non-SSD (R>P)": "black",
    }
    markers = {
        "Prisoner's Dilemma": "D",
        "Chicken": "s",
        "Stag Hunt": "o",
        "Non-SSD (R<P)": ".",
        "Non-SSD (R>P)": "*",
    }
    fig, ax = plt.subplots(figsize=(5, 5))
    for label in colors:
        xs = [f for f, g, lab in points if lab == label]
        ys = [g for f, g, lab in points if lab == label]
        if xs:
            ax.scatter(xs, ys, c=colors[label], marker=markers[label], label=label, alpha=0.7)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axvline(0, color="k", linewidth=0.5)
    ax.set_xlabel("fear = P - S")
    ax.set_ylabel("greed = T - R")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ablation_lines(x_values, series: dict[str, list[float]], x_name: str, y_name: str, title: str, out_path: str) -> None:
    """Fig. 7 style: y-metric vs. x_values, one line per condition in `series`."""
    fig, ax = plt.subplots(figsize=(5, 4))
    for label, ys in series.items():
        ax.plot(x_values, ys, marker="o", label=label)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
