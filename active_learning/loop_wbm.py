"""Active learning loop for the WBM dataset using Discovery Acceleration Factor (DAF)."""

from __future__ import annotations

import random
import sys

import numpy as np


class WBMALLoop:
    """Iterative active learning campaign on WBM 256K crystal structures.

    Each iteration:
    1. Fine-tune the CHGNet surrogate on the current labeled set
    2. Draw a random shortlist from the unlabeled pool
    3. Run MC Dropout on the shortlist to get (μ, σ) per structure
    4. Select the top-k by the acquisition function
    5. Look up oracle labels (DFT e_above_hull from WBM summary)
    6. Track the Discovery Acceleration Factor (DAF)

    The DAF at budget B is:
        precision_at_B / prevalence
      = (n_stable_found / B) / (n_stable_total / N)

    A random strategy achieves DAF ≈ 1.0; a perfect oracle achieves
    DAF = 1/prevalence ≈ 6.0 for WBM (16.7% prevalence).

    Args:
        dataset: WBMDataset instance with loaded summary and structures.
        surrogate: CHGNetSurrogate instance.
        strategy: Acquisition strategy (RandomStrategy/GreedyStrategy/UCBStrategy).
        initial_ids: material_ids in the initial labeled set.
        shortlist_size: Number of candidates to evaluate per iteration with
            MC Dropout. Must convert their graphs each iteration.
        rng_seed: Random seed for shortlist sampling.
    """

    def __init__(
        self,
        dataset,
        surrogate,
        strategy,
        initial_ids: list[str],
        shortlist_size: int = 5000,
        rng_seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.surrogate = surrogate
        self.strategy = strategy
        self.shortlist_size = shortlist_size
        self.rng = random.Random(rng_seed)

        all_ids = set(dataset.material_ids)
        self.labeled_ids: list[str] = list(initial_ids)
        self.unlabeled_ids: list[str] = [mid for mid in dataset.material_ids
                                          if mid not in set(initial_ids)]

        # Oracle labels are looked up lazily
        self._labeled_stable: dict[str, bool] = {
            mid: dataset.is_stable(mid) for mid in self.labeled_ids
        }

        self.history: dict[str, list] = {
            "n_labeled": [],
            "n_stable_found": [],
            "daf": [],
            "precision": [],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _n_stable_found(self) -> int:
        return sum(self._labeled_stable.values())

    def _compute_daf(self) -> float:
        n_labeled = len(self.labeled_ids)
        n_stable_found = self._n_stable_found()
        if n_labeled == 0:
            return float("nan")
        precision = n_stable_found / n_labeled
        return precision / self.dataset.prevalence

    def _record(self) -> None:
        n_stable = self._n_stable_found()
        n_labeled = len(self.labeled_ids)
        precision = n_stable / n_labeled if n_labeled > 0 else 0.0
        self.history["n_labeled"].append(n_labeled)
        self.history["n_stable_found"].append(n_stable)
        self.history["daf"].append(precision / self.dataset.prevalence)
        self.history["precision"].append(precision)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        n_iters: int = 20,
        k_per_iter: int = 100,
        mc_passes: int = 10,
        epochs_per_iter: int = 20,
        lr: float = 1e-3,
    ) -> dict:
        """Run the active learning loop.

        Args:
            n_iters: Number of iterations.
            k_per_iter: Structures to "query" (label) per iteration.
            mc_passes: Number of stochastic forward passes for uncertainty.
            epochs_per_iter: Fine-tuning epochs per iteration.
            lr: Learning rate for the surrogate's optimizer.

        Returns:
            History dict with keys: n_labeled, n_stable_found, daf, precision.
        """
        from active_learning.strategies import RandomStrategy

        is_random = isinstance(self.strategy, RandomStrategy)

        # Record baseline (before any AL)
        self._record()
        print(
            f"Initial: {len(self.labeled_ids)} labeled, "
            f"{self._n_stable_found()} stable found, "
            f"DAF={self._compute_daf():.3f}"
        )
        sys.stdout.flush()

        for iteration in range(n_iters):
            print(f"\n=== Iteration {iteration + 1}/{n_iters} ===")
            sys.stdout.flush()

            if len(self.unlabeled_ids) < k_per_iter:
                print("  Pool exhausted — stopping early.")
                break

            if not is_random:
                if epochs_per_iter > 0:
                    # Fine-tune surrogate on labeled set.
                    # Note: for WBM this often hurts because CHGNet is already
                    # well-calibrated from 700K MP training. Skip with epochs_per_iter=0.
                    labeled_e_forms = self.dataset.get_e_form(self.labeled_ids)
                    labeled_structs = self.dataset.load_structures(self.labeled_ids)
                    labeled_graphs = self.surrogate.precompute_graphs(labeled_structs)
                    print(f"  Fine-tuning on {len(labeled_graphs)} structures...")
                    sys.stdout.flush()
                    self.surrogate.fine_tune(
                        labeled_graphs, labeled_e_forms,
                        epochs=epochs_per_iter, lr=lr,
                    )

                # Two-stage shortlist:
                # Stage 1 — cheap point estimate on a large random sample of candidates
                #   to focus the shortlist on the most promising (low μ) region.
                # Stage 2 — full MC Dropout on the top-shortlist_size by Stage 1 μ.
                stage1_n = min(self.shortlist_size * 5, len(self.unlabeled_ids))

                stage1_ids = self.rng.sample(self.unlabeled_ids, stage1_n)
                stage1_structs = self.dataset.load_structures(stage1_ids)
                stage1_graphs = self.surrogate.precompute_graphs(stage1_structs)
                stage1_means = self.surrogate.predict_point(stage1_graphs)

                # Pick top-shortlist_size by predicted e_form (lowest = most stable)
                actual_shortlist = min(self.shortlist_size, stage1_n)
                top_idx = np.argsort(stage1_means)[:actual_shortlist]
                shortlist_ids = [stage1_ids[i] for i in top_idx]
                shortlist_graphs = [stage1_graphs[i] for i in top_idx]

                print(
                    f"  Stage1: {stage1_n} candidates → top {actual_shortlist} shortlist, "
                    f"then {mc_passes} MC passes..."
                )
                sys.stdout.flush()
                means, stds = self.surrogate.predict_with_uncertainty(
                    shortlist_graphs, n_passes=mc_passes
                )
            else:
                # Random strategy: select directly from the full unlabeled pool
                shortlist_ids = self.unlabeled_ids
                means = np.zeros(len(shortlist_ids))
                stds = np.zeros(len(shortlist_ids))

            # Select top-k by acquisition function
            actual_k = min(k_per_iter, len(shortlist_ids))
            selected_local = self.strategy.select(None, means, stds, k=actual_k)
            selected_ids = [shortlist_ids[i] for i in selected_local]

            # Update pools
            self.labeled_ids.extend(selected_ids)
            self._labeled_stable.update(
                {mid: self.dataset.is_stable(mid) for mid in selected_ids}
            )
            selected_set = set(selected_ids)
            self.unlabeled_ids = [mid for mid in self.unlabeled_ids
                                   if mid not in selected_set]

            # Track metrics
            self._record()
            print(
                f"  Labeled: {len(self.labeled_ids)}, "
                f"Stable found: {self._n_stable_found()}, "
                f"DAF={self._compute_daf():.3f}"
            )
            sys.stdout.flush()

        return self.history
