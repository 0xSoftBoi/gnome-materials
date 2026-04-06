import matplotlib.pyplot as plt
import numpy as np


def plot_comparison(histories, strategy_names, output_path='results/comparison.png'):
    """Plot comparison of strategies.

    Args:
        histories: list of history dicts from ActiveLearningLoop
        strategy_names: list of strategy names
        output_path: where to save the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Active Learning Comparison: GNN Materials Discovery', fontsize=14, fontweight='bold')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # Plot 1: Best value found over iterations
    ax = axes[0]
    for i, (history, name) in enumerate(zip(histories, strategy_names)):
        best_found = history['best_found']
        ax.plot(range(1, len(best_found) + 1), best_found, marker='o', label=name, color=colors[i], linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Best Formation Energy')
    ax.set_title('Best Value Found vs Iterations')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Top-10 discovery efficiency
    ax = axes[1]
    for i, (history, name) in enumerate(zip(histories, strategy_names)):
        top10 = history['top10_efficiency']
        ax.plot(range(1, len(top10) + 1), top10, marker='s', label=name, color=colors[i], linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Top-10 Discovered (Fraction)')
    ax.set_title('Top-10 Discovery Efficiency')
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Sample efficiency comparison
    ax = axes[2]
    n_iters = len(histories[0]['best_found'])
    initial_train = 50
    k_per_iter = 20
    total_samples_per_iter = [initial_train + k_per_iter * (i + 1) for i in range(n_iters)]

    for i, (history, name) in enumerate(zip(histories, strategy_names)):
        best_found = history['best_found']
        ax.scatter(total_samples_per_iter, best_found, s=100, label=name, color=colors[i], alpha=0.7)
        ax.plot(total_samples_per_iter, best_found, color=colors[i], linewidth=2, alpha=0.5)

    ax.set_xlabel('Total Labeled Samples')
    ax.set_ylabel('Best Formation Energy')
    ax.set_title('Sample Efficiency')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")
    plt.close()


def print_summary(histories, strategy_names):
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("ACTIVE LEARNING SUMMARY")
    print("=" * 60)

    for history, name in zip(histories, strategy_names):
        best = min(history['best_found'])
        final = history['best_found'][-1]
        top10_final = history['top10_efficiency'][-1]

        print(f"\n{name}:")
        print(f"  Final best value: {final:.4f}")
        print(f"  Overall best found: {best:.4f}")
        print(f"  Top-10 discovered: {top10_final:.1%}")

    print("\n" + "=" * 60)
