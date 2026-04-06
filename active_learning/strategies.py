import numpy as np


class SelectionStrategy:
    """Base class for candidate selection strategies."""

    def select(self, candidates, means, stds, k=20):
        """Select top-k candidates.

        Args:
            candidates: list of candidate Data objects
            means: predicted means (μ) for candidates
            stds: predicted std devs (σ) for candidates
            k: number of candidates to select

        Returns:
            indices of selected candidates
        """
        raise NotImplementedError


class RandomStrategy(SelectionStrategy):
    """Select candidates uniformly at random."""

    def select(self, candidates, means, stds, k=20):
        n = len(candidates)
        k = min(k, n)
        indices = np.random.choice(n, size=k, replace=False)
        return indices


class GreedyStrategy(SelectionStrategy):
    """Select candidates with lowest predicted energy (best μ)."""

    def select(self, candidates, means, stds, k=20):
        n = len(candidates)
        k = min(k, n)
        # Lower energy = better (minimize)
        indices = np.argsort(means)[:k]
        return indices


class UCBStrategy(SelectionStrategy):
    """Upper Confidence Bound: select by μ - λ*σ (uncertainty-aware minimization)."""

    def __init__(self, lambda_=1.0):
        self.lambda_ = lambda_

    def select(self, candidates, means, stds, k=20):
        n = len(candidates)
        k = min(k, n)
        # Score: μ - λ*σ (lower is better for minimization)
        # λ weights how much we explore uncertain regions
        scores = means - self.lambda_ * stds
        indices = np.argsort(scores)[:k]
        return indices
