"""Generate the figures referenced by the README and write-up."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

FIG_DIR = config.ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False})


def fig_business_tradeoff(sweep: pd.DataFrame):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))

    ax[0].plot(sweep["threshold"], sweep["net_value"], lw=2, color="#1f77b4")
    best = sweep.loc[sweep["net_value"].idxmax()]
    ax[0].axvline(best["threshold"], ls="--", color="#d62728", lw=1)
    ax[0].annotate(f"optimum\n{best['threshold']:.2f}",
                   (best["threshold"], best["net_value"]),
                   textcoords="offset points", xytext=(-45, -25), color="#d62728")
    ax[0].set_xlabel("fraud score threshold")
    ax[0].set_ylabel("net value (USD)")
    ax[0].set_title("Sensitivity vs business impact")

    ax[1].plot(sweep["threshold"], sweep["precision"], label="precision", lw=2)
    ax[1].plot(sweep["threshold"], sweep["recall"], label="recall", lw=2)
    ax[1].plot(sweep["threshold"], sweep["legit_blocked_rate"],
               label="legit blocked", lw=2, ls=":")
    ax[1].set_xlabel("fraud score threshold")
    ax[1].set_title("Precision / recall / collateral")
    ax[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "business_tradeoff.png", bbox_inches="tight")
    plt.close(fig)


def fig_score_distribution(scored: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist(scored["fraud_score"], bins=60, color="#4c72b0")
    ax.set_yscale("log")
    ax.set_xlabel("fraud score")
    ax.set_ylabel("entities (log scale)")
    ax.set_title("Score distribution: thin high-score tail")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "score_distribution.png", bbox_inches="tight")
    plt.close(fig)


def fig_detector_agreement(scored: pd.DataFrame):
    cols = ["clustering_score", "distribution_score", "cross_population_score"]
    corr = scored[cols].corr()

    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    im = ax.imshow(corr, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3), ["cluster", "distrib", "cross-pop"], rotation=20)
    ax.set_yticks(range(3), ["cluster", "distrib", "cross-pop"])
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    color="white" if corr.iloc[i, j] > 0.6 else "black")
    ax.set_title("Detector correlation\n(low = complementary coverage)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "detector_agreement.png", bbox_inches="tight")
    plt.close(fig)


def main():
    scored = pd.read_parquet(config.OUT_DIR / "scored_entities.parquet")
    sweep = pd.read_csv(config.OUT_DIR / "threshold_sweep.csv")

    fig_business_tradeoff(sweep)
    fig_score_distribution(scored)
    fig_detector_agreement(scored)

    corr = scored[["clustering_score", "distribution_score",
                   "cross_population_score"]].corr()
    print("Detector correlations:\n", corr.round(3))
    print(f"\nFigures -> {FIG_DIR}")


if __name__ == "__main__":
    main()
