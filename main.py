#!/usr/bin/env python3
"""
GNN Materials Discovery Pipeline with Active Learning
Simplified GNoME-style system with uncertainty-guided candidate selection.
"""

import os
import sys
import torch
import numpy as np

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Add project to path
sys.path.insert(0, os.path.dirname(__file__))

from data.dataset import load_dataset, SyntheticCrystalDataset, generate_candidates, ELEMENTS
from model.gnn import GNNRegressor
from active_learning.strategies import RandomStrategy, GreedyStrategy, UCBStrategy
from active_learning.loop import ActiveLearningLoop
from evaluation.metrics import plot_comparison, print_summary


def main():
    print("=" * 60)
    print("GNN Materials Discovery with Active Learning")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # ========== 1. Load dataset ==========
    print("\n[1] Loading dataset...")
    dataset = SyntheticCrystalDataset(n_samples=500, seed=42)
    print(f"    Dataset size: {len(dataset)} samples")
    print(f"    Element vocabulary: {len(ELEMENTS)} ({', '.join(ELEMENTS[:10])}...)")

    # ========== 2. Setup initial train/candidate split ==========
    print("\n[2] Setting up initial labeled and candidate pools...")
    all_indices = list(range(len(dataset)))
    np.random.seed(42)
    np.random.shuffle(all_indices)

    initial_train_size = 50
    initial_train_indices = all_indices[:initial_train_size]
    candidate_indices = all_indices[initial_train_size:]

    print(f"    Initial training set: {len(initial_train_indices)} samples")
    print(f"    Candidate pool: {len(candidate_indices)} samples")

    # ========== 3. Setup strategies ==========
    print("\n[3] Setting up selection strategies...")
    strategies = [
        ('Random', RandomStrategy()),
        ('Greedy (μ)', GreedyStrategy()),
        ('UCB (μ - λσ)', UCBStrategy(lambda_=1.0))
    ]
    print(f"    Strategies: {[name for name, _ in strategies]}")

    # ========== 4. Run active learning for each strategy ==========
    print("\n[4] Running active learning loops...")
    histories = []

    for strategy_name, strategy in strategies:
        print(f"\n    --- {strategy_name} ---")

        # Create fresh model for each strategy
        in_channels = len(ELEMENTS)
        model = GNNRegressor(in_channels=in_channels, hidden_dim=64, dropout_p=0.3)

        # Create AL loop
        loop = ActiveLearningLoop(
            dataset=dataset,
            model=model,
            strategy=strategy,
            train_indices=initial_train_indices,
            candidate_indices=candidate_indices.copy(),
            device=device
        )

        # Run AL loop
        loop.run(
            n_iters=5,
            k_per_iter=20,
            epochs_per_iter=20,
            batch_size=32,
            lr=1e-3
        )

        history = loop.get_history()
        histories.append(history)

    # ========== 5. Evaluate and plot ==========
    print("\n[5] Evaluating results...")
    strategy_names = [name for name, _ in strategies]

    plot_comparison(histories, strategy_names, output_path='results/comparison.png')
    print_summary(histories, strategy_names)

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
