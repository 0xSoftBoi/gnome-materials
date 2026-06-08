"""Plotting and metrics for WBM active learning campaign."""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_daf(n_stable_found: int, budget: int, n_stable_total: int, n_pool: int) -> float:
    """Discovery Acceleration Factor for active learning.

    DAF = precision_at_budget / prevalence
        = (n_stable_found / budget) / (n_stable_total / n_pool)

    A random strategy achieves DAF ≈ 1.0.
    A perfect oracle achieves DAF = n_pool / n_stable_total.

    Args:
        n_stable_found: Stable structures found within the labeled set.
        budget: Total structures labeled (= len(labeled_set)).
        n_stable_total: Total stable structures in the full pool.
        n_pool: Total pool size.
    """
    if budget == 0:
        return float("nan")
    prevalence = n_stable_total / n_pool
    return (n_stable_found / budget) / prevalence


def plot_wbm_al_results(
    histories: list[dict],
    strategy_names: list[str],
    n_stable_total: int,
    n_pool: int,
    output_path: str = "results/wbm_al_results.png",
) -> None:
    """Plot DAF curves for each strategy over the active learning campaign.

    Args:
        histories: List of history dicts from WBMALLoop.run().
        strategy_names: Display names for each strategy.
        n_stable_total: Number of truly stable structures in the full WBM pool.
        n_pool: Total WBM pool size.
        output_path: Where to save the figure.
    """
    prevalence = n_stable_total / n_pool
    perfect_daf = 1.0 / prevalence

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Active Learning on WBM (256K structures)\n"
        "CHGNet Surrogate + MC Dropout Uncertainty",
        fontsize=13,
        fontweight="bold",
    )

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    # Subplot 1: DAF vs. budget (number of labeled structures)
    ax = axes[0]
    for i, (hist, name) in enumerate(zip(histories, strategy_names)):
        ax.plot(
            hist["n_labeled"],
            hist["daf"],
            marker="o",
            markersize=4,
            label=name,
            color=colors[i % len(colors)],
            linewidth=2,
        )
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="Random baseline (DAF=1)")
    ax.axhline(perfect_daf, color="black", linestyle=":", linewidth=1,
               label=f"Perfect oracle (DAF={perfect_daf:.1f})")
    ax.set_xlabel("Labeled structures (budget)")
    ax.set_ylabel("Discovery Acceleration Factor (DAF)")
    ax.set_title("DAF vs. Labeling Budget")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Subplot 2: Cumulative stable structures found
    ax = axes[1]
    for i, (hist, name) in enumerate(zip(histories, strategy_names)):
        ax.plot(
            hist["n_labeled"],
            hist["n_stable_found"],
            marker="s",
            markersize=4,
            label=name,
            color=colors[i % len(colors)],
            linewidth=2,
        )
    # Random baseline
    budgets = np.array(histories[0]["n_labeled"])
    random_stable = budgets * prevalence
    ax.plot(budgets, random_stable, color="gray", linestyle="--", linewidth=1,
            label="Random baseline")
    ax.set_xlabel("Labeled structures (budget)")
    ax.set_ylabel("Stable structures found")
    ax.set_title("Cumulative Stable Structures Found")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Subplot 3: Precision over iterations
    ax = axes[2]
    for i, (hist, name) in enumerate(zip(histories, strategy_names)):
        iters = list(range(len(hist["precision"])))
        ax.plot(iters, hist["precision"], marker="^", markersize=4,
                label=name, color=colors[i % len(colors)], linewidth=2)
    ax.axhline(prevalence, color="gray", linestyle="--", linewidth=1,
               label=f"Prevalence ({prevalence:.1%})")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Precision (stable / labeled)")
    ax.set_title("Precision per Iteration")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nWBM plot saved to {output_path}")
    plt.close()


def print_wbm_summary(
    histories: list[dict],
    strategy_names: list[str],
    n_stable_total: int,
    n_pool: int,
) -> None:
    """Print final metrics for each strategy."""
    prevalence = n_stable_total / n_pool
    perfect_daf = 1.0 / prevalence

    print("\n" + "=" * 65)
    print("WBM ACTIVE LEARNING SUMMARY")
    print(f"Pool: {n_pool:,}  |  Stable: {n_stable_total:,}  |  "
          f"Prevalence: {prevalence:.1%}  |  Max DAF: {perfect_daf:.2f}")
    print("=" * 65)

    for hist, name in zip(histories, strategy_names):
        final_daf = hist["daf"][-1]
        final_stable = hist["n_stable_found"][-1]
        final_budget = hist["n_labeled"][-1]
        final_precision = hist["precision"][-1]
        improvement = final_daf / 1.0  # vs. random

        print(f"\n{name}:")
        print(f"  Budget used:      {final_budget:,}")
        print(f"  Stable found:     {final_stable:,} / {n_stable_total:,} "
              f"({100*final_stable/n_stable_total:.1f}%)")
        print(f"  Precision:        {final_precision:.1%}")
        print(f"  Final DAF:        {final_daf:.3f}x  ({improvement:.2f}x vs random)")

    print("\n" + "=" * 65)
