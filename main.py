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


def main_chgnet():
    """CHGNet pre-trained surrogate experiment on real MP data."""
    print("\n" + "=" * 80)
    print("EXPERIMENT D: CHGNet SURROGATE (Pre-trained + MC Dropout)")
    print("=" * 80)

    import numpy as np
    from data.mp_dataset_chgnet import MPStructureDataset
    from model.chgnet_surrogate import CHGNetSurrogate
    from active_learning.loop_chgnet import CHGNetALLoop
    from active_learning.strategies import RandomStrategy, GreedyStrategy, UCBStrategy
    from evaluation.metrics import plot_comparison, print_summary

    # Load dataset (downloads raw structures if not cached)
    print("\n[1] Loading raw MP structures...")
    dataset = MPStructureDataset(max_structures=2000)
    print(f"    Dataset size: {len(dataset)}")

    # Split
    print("[2] Setting up pools...")
    all_indices = list(range(len(dataset)))
    np.random.seed(42)
    np.random.shuffle(all_indices)

    initial_train_size = 100
    train_indices = all_indices[:initial_train_size]
    candidate_indices = all_indices[initial_train_size:]

    budget = 10 * 30  # 10 iters × 30 candidates
    coverage = budget / len(candidate_indices) * 100
    print(f"    Initial labeled: {len(train_indices)}")
    print(f"    Candidate pool: {len(candidate_indices)}")
    print(f"    Budget coverage: {coverage:.1f}%")

    strategies = [
        ('Random', RandomStrategy()),
        ('Greedy (μ)', GreedyStrategy()),
        ('UCB (μ - λσ)', UCBStrategy(lambda_=1.0)),
    ]

    # Pre-compute all graphs ONCE and share across strategies
    print("\n[3] Pre-computing graphs (shared across strategies)...")
    _surrogate_for_graphs = CHGNetSurrogate(dropout_p=0.3, freeze_backbone=True)
    all_indices = sorted(set(train_indices) | set(candidate_indices))
    all_structures = dataset.get_structures(all_indices)
    raw_graphs = _surrogate_for_graphs.precompute_graphs(all_structures)
    shared_graph_cache = {idx: g for idx, g in zip(all_indices, raw_graphs)}
    print(f"    Graph cache: {len(shared_graph_cache)} entries")
    del _surrogate_for_graphs

    histories = []
    print("\n[4] Running AL strategies...")
    for strategy_name, strategy in strategies:
        print(f"\n    --- {strategy_name} ---")

        surrogate = CHGNetSurrogate(dropout_p=0.3, freeze_backbone=True)

        loop = CHGNetALLoop(
            dataset=dataset,
            surrogate=surrogate,
            strategy=strategy,
            train_indices=train_indices.copy(),
            candidate_indices=candidate_indices.copy(),
            graph_cache=shared_graph_cache,
        )

        loop.run(
            n_iters=10,
            k_per_iter=30,
            epochs_per_iter=20,
            lr=1e-3,
            n_mc_passes=10,
        )
        histories.append(loop.get_history())

    # Save results
    print("\n[5] Saving results...")
    strategy_names = [name for name, _ in strategies]

    # Adapt histories for plot_comparison (expects 'top10_efficiency' key)
    adapted = []
    for h in histories:
        adapted.append({
            'best_found': h['best_found'],
            'top10_efficiency': h['top100_efficiency'],  # top-100 recall
            'val_mae': [0.0] * len(h['best_found']),
        })

    plot_comparison(adapted, strategy_names, output_path='results/comparison_chgnet.png',
                    top_k=100, initial_train=100, k_per_iter=30)
    print_summary(adapted, strategy_names, top_k=100)

    return histories


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


def main_wbm(
    summary_path='data/wbm_summary.csv.gz',
    structs_path='data/wbm_init_structs.json.bz2',
    max_structures=None,  # None = full 256K; set to e.g. 20000 for demo
    initial_labeled=200,
    n_iters=20,
    k_per_iter=100,
    shortlist_size=5000,
    mc_passes=10,
    epochs_per_iter=0,    # 0 = no fine-tuning (recommended for WBM)
    output_path='results/wbm_al_results.png',
):
    """Active learning campaign on WBM (Matbench Discovery test set).

    Simulates iterative materials discovery on 256K WBM crystal structures.
    Each iteration fine-tunes a CHGNet surrogate on the current labeled set,
    runs MC Dropout uncertainty estimation on a shortlist of candidates, and
    selects the most promising structures for "DFT verification" (i.e., looks
    up their DFT e_above_hull from the WBM summary).

    The primary metric is the Discovery Acceleration Factor (DAF):
        DAF = (stable found / budget) / (stable total / pool size)
    A random strategy achieves DAF ≈ 1.0; perfect oracle ≈ 6x for WBM.

    Key finding: epochs_per_iter=0 (no fine-tuning) outperforms fine-tuning
    for this task. CHGNet was trained on 700K MP structures and is already
    well-calibrated for WBM-style crystals. Fine-tuning on a small biased
    initial set causes catastrophic forgetting.

    Note: this is an iterative AL simulation, not the one-shot Matbench
    Discovery leaderboard task (which ranks by a single model inference pass).
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT E: WBM ACTIVE LEARNING CAMPAIGN")
    print("=" * 80)

    import os
    import numpy as np
    from data.wbm_dataset import WBMDataset
    from model.chgnet_surrogate import CHGNetSurrogate
    from active_learning.loop_wbm import WBMALLoop
    from active_learning.strategies import RandomStrategy, GreedyStrategy, UCBStrategy
    from evaluation.wbm_metrics import plot_wbm_al_results, print_wbm_summary

    # Download if not cached
    if not os.path.exists(summary_path):
        WBMDataset.download_summary(summary_path)
    if not os.path.exists(structs_path):
        WBMDataset.download_structures(structs_path)

    print(f"\n[1] Loading WBM dataset...")
    dataset = WBMDataset(
        summary_path=summary_path,
        structs_path=structs_path,
        max_structures=max_structures,
    )
    print(f"    Pool: {len(dataset):,}  |  Stable: {dataset.n_stable:,}  |  "
          f"Prevalence: {dataset.prevalence:.1%}")

    # Random initial labeled set
    np.random.seed(42)
    all_ids = dataset.material_ids
    initial_ids = list(np.random.choice(all_ids, size=initial_labeled, replace=False))
    print(f"    Initial labeled: {initial_labeled}")
    print(f"    Budget per iter: {k_per_iter}  |  Iters: {n_iters}")
    print(f"    Total budget: {initial_labeled + n_iters * k_per_iter:,} "
          f"({100*(initial_labeled + n_iters*k_per_iter)/len(dataset):.1f}% of pool)")

    strategies = [
        ('Random', RandomStrategy()),
        ('Greedy (μ)', GreedyStrategy()),
        ('UCB (μ - λσ)', UCBStrategy(lambda_=1.0)),
    ]

    histories = []
    print("\n[2] Running AL strategies...")
    for strategy_name, strategy in strategies:
        print(f"\n    --- {strategy_name} ---")

        surrogate = CHGNetSurrogate(dropout_p=0.3, freeze_backbone=True)

        loop = WBMALLoop(
            dataset=dataset,
            surrogate=surrogate,
            strategy=strategy,
            initial_ids=initial_ids,
            shortlist_size=shortlist_size,
        )

        loop.run(
            n_iters=n_iters,
            k_per_iter=k_per_iter,
            mc_passes=mc_passes,
            epochs_per_iter=epochs_per_iter,
            lr=1e-3,
        )
        histories.append(loop.history)

    # Save results
    print("\n[3] Saving results...")
    os.makedirs('results', exist_ok=True)
    strategy_names = [name for name, _ in strategies]
    plot_wbm_al_results(
        histories, strategy_names,
        n_stable_total=dataset.n_stable,
        n_pool=len(dataset),
        output_path=output_path,
    )
    print_wbm_summary(histories, strategy_names, dataset.n_stable, len(dataset))

    return histories


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GNN Materials Discovery Experiments')
    parser.add_argument('--experiment',
                        choices=['scaling', 'lambda', 'mp', 'chgnet', 'wbm', 'all'],
                        default='all',
                        help='Which experiment to run')
    parser.add_argument('--wbm-max', type=int, default=None,
                        help='Limit WBM pool size for quick demo (e.g. 20000)')
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

    if args.experiment in ('chgnet',):
        print("\n\nRunning CHGNet surrogate experiment...")
        chgnet_results = main_chgnet()

    if args.experiment in ('wbm',):
        print("\n\nRunning WBM active learning campaign...")
        wbm_results = main_wbm(max_structures=args.wbm_max)

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
