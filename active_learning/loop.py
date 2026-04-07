import sys
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_absolute_error
import numpy as np


class ActiveLearningLoop:
    """Active Learning loop: train → predict → select → label → retrain."""

    def __init__(self, dataset, model, strategy, train_indices, candidate_indices, device='cpu'):
        """
        Args:
            dataset: full dataset (train + candidates)
            model: GNN model
            strategy: SelectionStrategy instance
            train_indices: initial training set indices
            candidate_indices: indices of unlabeled candidates
            device: 'cpu' or 'cuda'
        """
        self.dataset = dataset
        self.model = model
        self.strategy = strategy
        self.train_indices = list(train_indices)
        self.candidate_indices = list(candidate_indices)
        self.device = device

        self.history = {
            'best_found': [],
            'top10_efficiency': [],
            'val_mae': []
        }

    def run(self, n_iters=10, k_per_iter=20, epochs_per_iter=20, batch_size=32, lr=1e-3):
        """Run active learning loop.

        Args:
            n_iters: number of AL iterations
            k_per_iter: candidates to select per iteration
            epochs_per_iter: training epochs per iteration
            batch_size: batch size for training
            lr: learning rate
        """
        from model.gnn import train_model

        for iteration in range(n_iters):
            print(f"\n=== Iteration {iteration + 1}/{n_iters} ===")
            sys.stdout.flush()

            # Train on labeled set
            train_data = [self.dataset[i] for i in self.train_indices]
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

            self.model = train_model(
                self.model, train_loader, train_loader, device=self.device,
                epochs=epochs_per_iter, lr=lr
            )

            # Predict on remaining candidates
            if not self.candidate_indices:
                print("No more candidates!")
                break

            candidate_data = [self.dataset[i] for i in self.candidate_indices]
            candidate_loader = DataLoader(candidate_data, batch_size=batch_size, shuffle=False)

            means, stds = self.model.predict_with_uncertainty(candidate_loader, n_passes=20, device=self.device)

            # Select top-k via strategy
            selected_local_indices = self.strategy.select(candidate_data, means, stds, k=k_per_iter)
            selected_global_indices = [self.candidate_indices[i] for i in selected_local_indices]

            # Add to training set
            self.train_indices.extend(selected_global_indices)
            for idx in selected_global_indices:
                self.candidate_indices.remove(idx)

            # Track metrics
            train_y = torch.cat([self.dataset[i].y for i in self.train_indices])
            best_found = train_y.min().item()
            self.history['best_found'].append(best_found)

            # Top-10 efficiency: how many of top 10 best are in training set?
            all_y = torch.tensor([self.dataset[i].y.item() for i in range(len(self.dataset))])
            top10_indices = torch.argsort(all_y)[:10]
            top10_in_train = sum(1 for idx in top10_indices if idx in self.train_indices)
            self.history['top10_efficiency'].append(top10_in_train / 10.0)

            print(f"  Best found: {best_found:.4f}")
            print(f"  Top-10 discovered: {top10_in_train}/10")
            print(f"  Training set size: {len(self.train_indices)}")

    def get_history(self):
        return self.history
