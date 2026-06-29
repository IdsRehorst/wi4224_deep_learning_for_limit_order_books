# plot_avg_movement_heatmaps.py

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def to_grid(avg_probs, K):
    side = 2 * K + 1

    if avg_probs.shape[0] != side * side:
        raise ValueError(
            f"Expected vector of length {side * side} for K={K}, "
            f"but got length {avg_probs.shape[0]}."
        )

    return avg_probs.reshape(side, side)


def load_grid(path, K):
    avg_probs = np.load(path)
    return to_grid(avg_probs, K)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--standard-avg", required=True)
    parser.add_argument("--reduced-spatial-avg", required=True)
    parser.add_argument("--full-spatial-avg", required=True)

    parser.add_argument(
        "--output",
        default="figures/kghm_movement_predicted_mass_full.png",
    )

    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--stock", type=str, default="KGHM")

    args = parser.parse_args()

    standard = load_grid(args.standard_avg, args.K)
    reduced_spatial = load_grid(args.reduced_spatial_avg, args.K)
    full_spatial = load_grid(args.full_spatial_avg, args.K)

    grids = [standard, reduced_spatial, full_spatial]

    titles = [
        "Standard NN",
        "Reduced spatial NN",
        "Full local spatial NN",
    ]

    positive_values = np.concatenate([g[g > 0].ravel() for g in grids])

    if positive_values.size == 0:
        raise ValueError("All heatmap values are zero. Cannot use log scale.")

    norm = LogNorm(
        vmin=max(positive_values.min(), 1e-8),
        vmax=max(g.max() for g in grids),
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 5.5),
        constrained_layout=True,
    )

    for ax, grid, title in zip(axes, grids, titles):
        im = ax.imshow(
            grid,
            origin="lower",
            extent=[
                -args.K - 0.5,
                args.K + 0.5,
                -args.K - 0.5,
                args.K + 0.5,
            ],
            norm=norm,
            aspect="equal",
        )

        ax.set_title(title, fontsize=14)
        ax.set_xlabel(r"$\Delta B_t$")
        ax.set_ylabel(r"$\Delta A_t$")

        ax.set_xticks(range(-args.K, args.K + 1, 5))
        ax.set_yticks(range(-args.K, args.K + 1, 5))

        # Mark the no-movement class.
        ax.scatter([0], [0], marker="x", s=90, linewidths=2)
        ax.text(
            0.5,
            0.5,
            r"$(0,0)$",
            fontsize=10,
            ha="left",
            va="bottom",
        )

    cbar = fig.colorbar(im, ax=axes, shrink=0.9)
    cbar.set_label("Average predicted probability, log scale")

    fig.suptitle(
        f"Average predicted distribution on {args.stock} movement observations",
        fontsize=15,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved to {output}")


if __name__ == "__main__":
    main()