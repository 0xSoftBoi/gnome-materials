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


def plot_lambda_tuning(lambda_histories, lambda_values, output_path='results/lambda_tuning_analysis.png'):
    """Compare UCB performance across different lambda values.

    Args:
        lambda_histories: dict of {lambda: [history_random, history_greedy, history_ucb]}
        lambda_values: list of lambda values tested
        output_path: where to save the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Lambda Tuning: Exploration-Exploitation Tradeoff', fontsize=14, fontweight='bold')

    colors = plt.cm.viridis(np.linspace(0, 1, len(lambda_values)))

    # Extract UCB performance for each lambda
    ucb_best_found = []
    ucb_top10 = []

    for lam in lambda_values:
        histories = lambda_histories[lam]
        # Index 2 is UCB strategy
        history = histories[2]
        ucb_best_found.append(history['best_found'])
        ucb_top10.append(history['top10_efficiency'])

    # Plot 1: Best value trajectory for each lambda
    ax = axes[0]
    for i, (lam, best_found, color) in enumerate(zip(lambda_values, ucb_best_found, colors)):
        ax.plot(range(1, len(best_found) + 1), best_found, marker='o', label=f'λ={lam}',
                color=color, linewidth=2, markersize=6)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Best Formation Energy')
    ax.set_title('Best Value Found vs Iteration')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 2: Top-10 discovery for each lambda
    ax = axes[1]
    for i, (lam, top10, color) in enumerate(zip(lambda_values, ucb_top10, colors)):
        ax.plot(range(1, len(top10) + 1), top10, marker='s', label=f'λ={lam}',
                color=color, linewidth=2, markersize=6)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Top-10 Discovered (Fraction)')
    ax.set_title('Top-10 Discovery Efficiency vs Iteration')
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 3: Final performance comparison
    ax = axes[2]
    final_best = [best_found[-1] for best_found in ucb_best_found]
    final_top10 = [top10[-1] for top10 in ucb_top10]

    x_pos = np.arange(len(lambda_values))
    bar_width = 0.35

    # Normalize final_best to [0, 1] for visualization alongside top10
    final_best_normalized = (np.array(final_best) - min(final_best)) / (max(final_best) - min(final_best))

    ax.bar(x_pos - bar_width/2, final_top10, bar_width, label='Top-10 Discovered', alpha=0.8, color='#1f77b4')
    ax.bar(x_pos + bar_width/2, final_best_normalized, bar_width, label='Best Value (normalized)', alpha=0.8, color='#ff7f0e')

    ax.set_xlabel('Lambda Value')
    ax.set_ylabel('Performance (normalized)')
    ax.set_title('Final Performance vs Lambda')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{lam}' for lam in lambda_values])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nLambda tuning plot saved to {output_path}")
    plt.close()

    # Print lambda summary
    print("\n" + "=" * 60)
    print("LAMBDA TUNING SUMMARY")
    print("=" * 60)
    print("\nUCB Performance by Lambda:")
    print(f"{'Lambda':<8} {'Best Found':<15} {'Top-10 %':<12}")
    print("-" * 35)
    for lam, best, top10 in zip(lambda_values, final_best, final_top10):
        print(f"{lam:<8} {best:<15.4f} {top10*100:<12.1f}%")
    print("\n" + "=" * 60)


def plot_scaling_analysis(all_histories, pool_sizes, output_path='results/scaling_analysis.png'):
    """Compare strategy performance across different pool sizes (scaling experiment).

    Args:
        all_histories: dict of {exp_name: [history1, history2, history3]}
        pool_sizes: list of pool sizes corresponding to experiments
        output_path: where to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Scaling Analysis: Strategy Performance vs Pool Size', fontsize=14, fontweight='bold')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    strategy_names = ['Random', 'Greedy (μ)', 'UCB (μ - λσ)']

    # Collect final performance per strategy across pool sizes
    final_best_values = {name: [] for name in strategy_names}
    top10_discoveries = {name: [] for name in strategy_names}

    for (exp_name, histories), pool_size in zip(sorted(all_histories.items()), pool_sizes):
        for i, (history, strategy_name) in enumerate(zip(histories, strategy_names)):
            final_best = history['best_found'][-1]
            top10_final = history['top10_efficiency'][-1]

            final_best_values[strategy_name].append(final_best)
            top10_discoveries[strategy_name].append(top10_final)

    # Plot 1: Best value found vs pool size
    ax = axes[0]
    for i, strategy_name in enumerate(strategy_names):
        ax.plot(pool_sizes, final_best_values[strategy_name], marker='o', label=strategy_name,
                color=colors[i], linewidth=2.5, markersize=8)
    ax.set_xlabel('Candidate Pool Size', fontsize=11)
    ax.set_ylabel('Best Formation Energy', fontsize=11)
    ax.set_title('Best Value Found vs Pool Size')
    ax.set_xscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 2: Top-10 discovery efficiency vs pool size
    ax = axes[1]
    for i, strategy_name in enumerate(strategy_names):
        ax.plot(pool_sizes, [x*100 for x in top10_discoveries[strategy_name]], marker='s',
                label=strategy_name, color=colors[i], linewidth=2.5, markersize=8)
    ax.set_xlabel('Candidate Pool Size', fontsize=11)
    ax.set_ylabel('Top-10 Discovered (%)', fontsize=11)
    ax.set_title('Top-10 Discovery Efficiency vs Pool Size')
    ax.set_xscale('log')
    ax.set_ylim([0, 105])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nScaling plot saved to {output_path}")
    plt.close()

    # Print scaling summary
    print("\n" + "=" * 60)
    print("SCALING ANALYSIS SUMMARY")
    print("=" * 60)
    for strategy_name in strategy_names:
        print(f"\n{strategy_name}:")
        for pool_size, best, top10 in zip(pool_sizes, final_best_values[strategy_name],
                                           top10_discoveries[strategy_name]):
            print(f"  Pool={pool_size:5d}: best={best:.4f}, top-10={top10:.1%}")
    print("\n" + "=" * 60)
