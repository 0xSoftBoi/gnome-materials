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


def aggregate_seeds(json_paths: list[str]) -> dict:
    """Load per-seed JSON result files and compute mean ± std across seeds.

    Args:
        json_paths: Paths to JSON files written by main_wbm() per seed.

    Returns:
        Dict with keys 'strategy_names', 'n_labeled', 'n_pool', 'n_stable',
        and per-strategy arrays 'daf_mean', 'daf_std', 'stable_mean', 'stable_std'.
    """
    import json
    runs = [json.load(open(p)) for p in json_paths]
    strategy_names = runs[0]['strategy_names']
    n_labeled = runs[0]['histories'][0]['n_labeled']

    result = {
        'strategy_names': strategy_names,
        'n_labeled': n_labeled,
        'n_pool': runs[0]['n_pool'],
        'n_stable': runs[0]['n_stable'],
        'seeds': [r['seed'] for r in runs],
    }

    for si in range(len(strategy_names)):
        dafs = np.array([r['histories'][si]['daf'] for r in runs])
        stables = np.array([r['histories'][si]['n_stable_found'] for r in runs])
        result[f'daf_mean_{si}'] = dafs.mean(axis=0).tolist()
        result[f'daf_std_{si}'] = dafs.std(axis=0).tolist()
        result[f'stable_mean_{si}'] = stables.mean(axis=0).tolist()
        result[f'stable_std_{si}'] = stables.std(axis=0).tolist()

    return result


def plot_multiseed_results(
    agg: dict,
    output_path: str = "results/wbm_multiseed.png",
) -> None:
    """Plot multi-seed DAF curves with ±1σ shaded bands."""
    strategy_names = agg['strategy_names']
    n_labeled = np.array(agg['n_labeled'])
    n_pool = agg['n_pool']
    n_stable = agg['n_stable']
    prevalence = n_stable / n_pool
    perfect_daf = 1.0 / prevalence
    n_seeds = len(agg['seeds'])

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"WBM Active Learning — {n_seeds} seeds (mean ± 1σ)\n"
        "CHGNet Surrogate + MC Dropout",
        fontsize=13, fontweight="bold",
    )

    ax = axes[0]
    for si, name in enumerate(strategy_names):
        mean = np.array(agg[f'daf_mean_{si}'])
        std = np.array(agg[f'daf_std_{si}'])
        c = colors[si % len(colors)]
        ax.plot(n_labeled, mean, color=c, linewidth=2, label=name)
        ax.fill_between(n_labeled, mean - std, mean + std, color=c, alpha=0.2)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="Random (DAF=1)")
    ax.axhline(perfect_daf, color="black", linestyle=":", linewidth=1,
               label=f"Oracle (DAF={perfect_daf:.1f})")
    ax.set_xlabel("Labeled structures (budget)")
    ax.set_ylabel("DAF")
    ax.set_title("Discovery Acceleration Factor")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for si, name in enumerate(strategy_names):
        mean = np.array(agg[f'stable_mean_{si}'])
        std = np.array(agg[f'stable_std_{si}'])
        c = colors[si % len(colors)]
        ax.plot(n_labeled, mean, color=c, linewidth=2, label=name)
        ax.fill_between(n_labeled, mean - std, mean + std, color=c, alpha=0.2)
    ax.plot(n_labeled, n_labeled * prevalence, color="gray", linestyle="--",
            linewidth=1, label="Random baseline")
    ax.set_xlabel("Labeled structures (budget)")
    ax.set_ylabel("Stable structures found")
    ax.set_title("Cumulative Stable Found")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Multi-seed plot saved to {output_path}")
    plt.close()


def print_multiseed_summary(agg: dict) -> None:
    """Print mean ± std DAF for each strategy across seeds."""
    strategy_names = agg['strategy_names']
    n_pool = agg['n_pool']
    n_stable = agg['n_stable']
    prevalence = n_stable / n_pool
    seeds = agg['seeds']

    print("\n" + "=" * 65)
    print("WBM MULTI-SEED SUMMARY")
    print(f"Pool: {n_pool:,}  |  Stable: {n_stable:,}  |  "
          f"Prevalence: {prevalence:.1%}  |  Seeds: {seeds}")
    print("=" * 65)

    for si, name in enumerate(strategy_names):
        final_daf_mean = agg[f'daf_mean_{si}'][-1]
        final_daf_std = agg[f'daf_std_{si}'][-1]
        final_stable_mean = agg[f'stable_mean_{si}'][-1]
        final_stable_std = agg[f'stable_std_{si}'][-1]
        print(f"\n{name}:")
        print(f"  Final DAF:     {final_daf_mean:.3f} ± {final_daf_std:.3f}")
        print(f"  Stable found:  {final_stable_mean:.1f} ± {final_stable_std:.1f}")

    print("\n" + "=" * 65)
