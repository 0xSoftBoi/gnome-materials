"""Active learning loop using CHGNet surrogate on raw pymatgen structures."""

import sys
import numpy as np
from active_learning.strategies import RandomStrategy


class CHGNetALLoop:
    """
    AL loop for structure datasets with CHGNet surrogate.

    Pre-computes all graphs once at init — avoids repeated graph_converter calls
    (which dominate runtime at ~60ms/structure × pool size).
    Uses top-100 recall instead of top-10 (more meaningful at sparse coverage).
    """

    def __init__(self, dataset, surrogate, strategy,
                 train_indices, candidate_indices, graph_cache=None):
        self.dataset = dataset
        self.surrogate = surrogate
        self.strategy = strategy
        self.train_indices = list(train_indices)
        self.candidate_indices = list(candidate_indices)

        self.history = {
            'best_found': [],
            'top100_efficiency': [],
        }

        # Pre-compute top-100 true indices for recall tracking
        all_energies = np.array([dataset.get_energy(i) for i in range(len(dataset))])
        self.top100_indices = set(np.argsort(all_energies)[:100].tolist())

        if graph_cache is not None:
            self.graph_cache = graph_cache
        else:
            all_indices = sorted(set(train_indices) | set(candidate_indices))
            all_structures = dataset.get_structures(all_indices)
            print(f"  Pre-computing {len(all_structures)} graphs (one-time cost)...")
            sys.stdout.flush()
            raw_graphs = surrogate.precompute_graphs(all_structures)
            self.graph_cache = {idx: g for idx, g in zip(all_indices, raw_graphs)}
            print(f"  Graph cache ready ({len(self.graph_cache)} entries)")
            sys.stdout.flush()

    def run(self, n_iters=10, k_per_iter=30, epochs_per_iter=20,
            lr=1e-3, n_mc_passes=10):
        is_random = isinstance(self.strategy, RandomStrategy)

        for iteration in range(n_iters):
            print(f"\n=== Iteration {iteration + 1}/{n_iters} ===")
            sys.stdout.flush()

            if not self.candidate_indices:
                print("  No more candidates!")
                break

            if not is_random:
                # Fine-tune surrogate on labeled set
                labeled_graphs = [self.graph_cache[i] for i in self.train_indices]
                labeled_energies = self.dataset.get_energies(self.train_indices)
                print(f"  Fine-tuning on {len(labeled_graphs)} structures...")
                sys.stdout.flush()
                self.surrogate.fine_tune(
                    labeled_graphs, labeled_energies,
                    epochs=epochs_per_iter, lr=lr,
                )

                # MC dropout predictions over candidates
                candidate_graphs = [self.graph_cache[i] for i in self.candidate_indices]
                print(f"  Running {n_mc_passes} MC passes on {len(candidate_graphs)} candidates...")
                sys.stdout.flush()
                means, stds = self.surrogate.predict_with_uncertainty(
                    candidate_graphs, n_passes=n_mc_passes
                )
            else:
                n = len(self.candidate_indices)
                means = np.zeros(n)
                stds = np.zeros(n)

            # Select top-k via strategy
            selected_local = self.strategy.select(None, means, stds, k=k_per_iter)
            selected_global = [self.candidate_indices[i] for i in selected_local]

            # Update pools
            self.train_indices.extend(selected_global)
            selected_set = set(selected_global)
            self.candidate_indices = [i for i in self.candidate_indices if i not in selected_set]

            # Track metrics
            train_energies = self.dataset.get_energies(self.train_indices)
            best_found = min(train_energies)
            self.history['best_found'].append(best_found)

            top100_in_train = sum(1 for idx in self.train_indices
                                  if idx in self.top100_indices)
            self.history['top100_efficiency'].append(top100_in_train / 100.0)

            print(f"  Best found: {best_found:.4f} eV/atom")
            print(f"  Top-100 discovered: {top100_in_train}/100")
            print(f"  Training set size: {len(self.train_indices)}")
            sys.stdout.flush()

    def get_history(self):
        return self.history
