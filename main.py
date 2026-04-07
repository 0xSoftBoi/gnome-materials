#!/usr/bin/env python3
"""
GNN Materials Discovery Pipeline with Active Learning
Simplified GNoME-style system with uncertainty-guided candidate selection.
"""

import os
import sys
import torch
import numpy as np
import argparse

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Add project to path
sys.path.insert(0, os.path.dirname(__file__))

from data.dataset import load_dataset, SyntheticCrystalDataset, generate_candidates, ELEMENTS
from data.mp_dataset import MPCrystalDataset, MP_IN_CHANNELS
from model.gnn import GNNRegressor
from active_learning.strategies import RandomStrategy, GreedyStrategy, UCBStrategy
from active_learning.loop import ActiveLearningLoop
from evaluation.metrics import plot_comparison, print_summary, plot_scaling_analysis, plot_lambda_tuning


def run_al_experiment(dataset, initial_train_indices, candidate_indices, strategies, device='cpu', experiment_name='v2', in_channels=26):
    """Run a single AL experiment with given strategies."""
    histories = []

    for strategy_name, strategy in strategies:
        print(f"\n    --- {strategy_name} ---")

        # Create fresh model for each strategy
        model = GNNRegressor(in_channels=in_channels, hidden_dim=128, dropout_p=0.3)

        # Create AL loop
        loop = ActiveLearningLoop(
            dataset=dataset,
            model=model,
            strategy=strategy,
            train_indices=initial_train_indices.copy(),
            candidate_indices=candidate_indices.copy(),
            device=device
        )

        # Run AL loop
        loop.run(
            n_iters=5,
            k_per_iter=20,
            epochs_per_iter=30,
            batch_size=16,
            lr=1e-3
        )

        history = loop.get_history()
        histories.append(history)

    return histories


def main_scaling():
    """Scaling experiment: test pool size impact on strategy performance."""
    print("\n" + "=" * 80)
    print("EXPERIMENT A: SCALING ANALYSIS")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    pool_configs = [
        (2000, 'v2'),
        (10000, 'v3-small'),
        (50000, 'v3-large'),
    ]

    all_histories = {}
    pool_sizes = []

    for n_samples, exp_name in pool_configs:
        print(f"\n{'='*60}\nPool Size: {n_samples} ({exp_name})\n{'='*60}")

        # Load dataset
        print(f"[1] Loading dataset: {n_samples} samples, 26-dim features...")
        dataset = SyntheticCrystalDataset(n_samples=n_samples, seed=42)

        # Setup split
        print(f"[2] Setting up labeled/candidate pools...")
        all_indices = list(range(len(dataset)))
        np.random.seed(42)
        np.random.shuffle(all_indices)

        initial_train_size = 100
        initial_train_indices = all_indices[:initial_train_size]
        candidate_indices = all_indices[initial_train_size:]

        budget_pct = (10 * 30 / len(candidate_indices)) * 100
        print(f"    Initial training: {len(initial_train_indices)}")
        print(f"    Candidate pool: {len(candidate_indices)}")
        print(f"    Budget coverage: {budget_pct:.1f}%")

        # Setup strategies
        print(f"[3] Setting up strategies...")
        strategies = [
            ('Random', RandomStrategy()),
            ('Greedy (μ)', GreedyStrategy()),
            ('UCB (μ - λσ)', UCBStrategy(lambda_=1.0))
        ]

        # Run experiment
        print(f"[4] Running active learning...")
        histories = run_al_experiment(dataset, initial_train_indices, candidate_indices, strategies, device, exp_name)

        # Save results
        print(f"[5] Saving results...")
        strategy_names = [name for name, _ in strategies]
        output_file = f'results/comparison_{exp_name}.png'
        plot_comparison(histories, strategy_names, output_path=output_file)
        print_summary(histories, strategy_names)

        all_histories[exp_name] = histories
        pool_sizes.append(n_samples)

    # Cross-pool scaling plot
    print(f"\n[6] Creating scaling analysis plot...")
    plot_scaling_analysis(all_histories, pool_sizes, output_path='results/scaling_analysis.png')

    return all_histories


def main_lambda_tuning():
    """Lambda tuning experiment: explore exploration-exploitation tradeoff."""
    print("\n" + "=" * 80)
    print("EXPERIMENT B: LAMBDA TUNING (Exploration-Exploitation Tradeoff)")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    n_samples = 2000
    print(f"\n[1] Loading dataset: {n_samples} samples, 26-dim features...")
    dataset = SyntheticCrystalDataset(n_samples=n_samples, seed=42)

    # Setup split
    print(f"[2] Setting up labeled/candidate pools...")
    all_indices = list(range(len(dataset)))
    np.random.seed(42)
    np.random.shuffle(all_indices)

    initial_train_size = 100
    initial_train_indices = all_indices[:initial_train_size]
    candidate_indices = all_indices[initial_train_size:]

    print(f"    Initial training: {len(initial_train_indices)}")
    print(f"    Candidate pool: {len(candidate_indices)}")

    # Test lambda values
    lambda_values = [0.0, 0.5, 1.0, 2.0, 5.0]
    print(f"\n[3] Testing lambda values: {lambda_values}")

    lambda_histories = {}

    for lam in lambda_values:
        print(f"\n{'='*60}\nLambda = {lam}\n{'='*60}")

        # Strategies: Random + Greedy + UCB with different lambdas
        strategies = [
            ('Random', RandomStrategy()),
            ('Greedy (μ)', GreedyStrategy()),
            (f'UCB (λ={lam})', UCBStrategy(lambda_=lam))
        ]

        # Run experiment
        print(f"[4] Running active learning...")
        histories = run_al_experiment(dataset, initial_train_indices, candidate_indices, strategies, device, f'lambda_{lam}')

        # Save results
        print(f"[5] Saving results...")
        strategy_names = [name for name, _ in strategies]
        output_file = f'results/lambda_tuning_{lam}.png'
        plot_comparison(histories, strategy_names, output_path=output_file)
        print_summary(histories, strategy_names)

        lambda_histories[lam] = histories

    # Lambda comparison plot
    print(f"\n[6] Creating lambda tuning analysis plot...")
    plot_lambda_tuning(lambda_histories, lambda_values, output_path='results/lambda_tuning_analysis.png')

    return lambda_histories


def main_mp():
    """Materials Project experiment: run AL on real DFT-computed structures."""
    print("\n" + "=" * 80)
    print("EXPERIMENT C: MATERIALS PROJECT (Real DFT Data)")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load MP dataset
    print(f"\n[1] Loading Materials Project dataset...")
    try:
        dataset = MPCrystalDataset(max_structures=None)  # Download all
    except EnvironmentError as e:
        print(f"ERROR: {e}")
        return None

    print(f"    Total structures: {len(dataset)}")

    # Setup split
    print(f"[2] Setting up labeled/candidate pools...")
    all_indices = list(range(len(dataset)))
    np.random.seed(42)
    np.random.shuffle(all_indices)

    initial_train_size = 100
    initial_train_indices = all_indices[:initial_train_size]
    candidate_indices = all_indices[initial_train_size:]

    budget_pct = (10 * 30 / len(candidate_indices)) * 100
    print(f"    Initial training: {len(initial_train_indices)}")
    print(f"    Candidate pool: {len(candidate_indices)}")
    print(f"    Budget coverage: {budget_pct:.1f}%")

    # Setup strategies
    print(f"[3] Setting up strategies...")
    strategies = [
        ('Random', RandomStrategy()),
        ('Greedy (μ)', GreedyStrategy()),
        ('UCB (μ - λσ)', UCBStrategy(lambda_=1.0))
    ]

    # Run experiment
    print(f"[4] Running active learning on real MP data...")
    histories = run_al_experiment(
        dataset,
        initial_train_indices,
        candidate_indices,
        strategies,
        device,
        'mp_v4',
        in_channels=MP_IN_CHANNELS
    )

    if histories:
        # Save results
        print(f"[5] Saving results...")
        strategy_names = [name for name, _ in strategies]
        plot_comparison(histories, strategy_names, output_path='results/comparison_mp_v4.png')
        print_summary(histories, strategy_names)

    return histories


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GNN Materials Discovery Experiments')
    parser.add_argument('--experiment', choices=['scaling', 'lambda', 'mp', 'all'], default='all',
                       help='Which experiment to run')
    args = parser.parse_args()

    if args.experiment in ('scaling', 'all'):
        print("\nRunning scaling experiment...")
        scaling_results = main_scaling()

    if args.experiment in ('lambda', 'all'):
        print("\n\nRunning lambda tuning experiment...")
        lambda_results = main_lambda_tuning()

    if args.experiment in ('mp', 'all'):
        print("\n\nRunning Materials Project experiment...")
        mp_results = main_mp()

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 80)
    print("\nResults saved to results/:")
    if args.experiment in ('scaling', 'all'):
        print("  - Scaling: comparison_v2.png, comparison_v3-small.png, comparison_v3-large.png, scaling_analysis.png")
    if args.experiment in ('lambda', 'all'):
        print("  - Lambda: lambda_tuning_0.0.png ... lambda_tuning_5.0.png, lambda_tuning_analysis.png")
    if args.experiment in ('mp', 'all'):
        print("  - Materials Project: comparison_mp_v4.png")
