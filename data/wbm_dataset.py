"""WBM dataset loader for active learning on the Matbench Discovery benchmark."""

from __future__ import annotations

import bz2
import os
import urllib.request

import numpy as np
import pandas as pd
from pymatgen.core import Structure


STRUCTS_FIGSHARE_ID = 40344466
SUMMARY_FIGSHARE_ID = 44225498


class WBMDataset:
    """WBM test set for active learning simulation.

    Provides:
    - Oracle stability labels (DFT e_above_hull via MP convex hull)
    - Formation energies for surrogate fine-tuning targets
    - Lazy structure loading from the WBM init-structs JSON

    The oracle uses ``e_above_hull_mp2020_corrected_ppd_mp`` — the same
    column Matbench Discovery uses for all model comparisons. A structure
    is considered stable if this value is ≤ 0 eV/atom.

    Args:
        summary_path: Path to ``wbm-summary.csv.gz``.
        structs_path: Path to ``wbm-init-structs.json.bz2``. Required for
            structure-based operations (graph conversion). Pass ``None`` to
            skip structure loading (summary-only mode).
        max_structures: Limit dataset to first N structures (for quick demos).
    """

    STABILITY_COL = "e_above_hull_mp2020_corrected_ppd_mp"
    E_FORM_COL = "e_form_per_atom_mp2020_corrected"
    STABILITY_THRESHOLD = 0.0  # eV/atom

    def __init__(
        self,
        summary_path: str,
        structs_path: str | None = None,
        max_structures: int | None = None,
    ) -> None:
        self.df = pd.read_csv(summary_path)
        if "material_id" not in self.df.columns:
            raise ValueError("summary CSV must have a 'material_id' column")
        self.df = self.df.set_index("material_id")
        if max_structures is not None:
            self.df = self.df.iloc[:max_structures]

        self._structs_path = structs_path
        self._structs_df: pd.DataFrame | None = None  # lazy

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------

    @property
    def material_ids(self) -> list[str]:
        return list(self.df.index)

    def __len__(self) -> int:
        return len(self.df)

    @property
    def n_stable(self) -> int:
        return int((self.df[self.STABILITY_COL] <= self.STABILITY_THRESHOLD).sum())

    @property
    def prevalence(self) -> float:
        return self.n_stable / len(self.df)

    # ------------------------------------------------------------------
    # Oracle interface
    # ------------------------------------------------------------------

    def is_stable(self, material_id: str) -> bool:
        return bool(self.df.loc[material_id, self.STABILITY_COL] <= self.STABILITY_THRESHOLD)

    def are_stable(self, material_ids: list[str]) -> np.ndarray:
        return (self.df.loc[material_ids, self.STABILITY_COL].values <= self.STABILITY_THRESHOLD)

    def get_e_above_hull(self, material_ids: list[str]) -> np.ndarray:
        return self.df.loc[material_ids, self.STABILITY_COL].values.astype(float)

    # Fine-tuning target: MP2020-corrected formation energy (consistent with convex hull)
    def get_e_form(self, material_ids: list[str]) -> np.ndarray:
        return self.df.loc[material_ids, self.E_FORM_COL].values.astype(float)

    # ------------------------------------------------------------------
    # Structure loading (lazy)
    # ------------------------------------------------------------------

    def _ensure_structs_loaded(self) -> None:
        if self._structs_df is not None:
            return
        if self._structs_path is None:
            raise RuntimeError(
                "structs_path not provided. Download wbm-init-structs.json.bz2 with "
                "WBMDataset.download_structures() first."
            )
        print(f"Loading WBM structures from {self._structs_path}...")
        # File is pandas-style columnar JSON: {col: {int_idx: val, ...}, ...}
        # Columns: material_id, formula_from_cse, initial_structure
        self._structs_df = pd.read_json(self._structs_path).set_index("material_id")
        print(f"  Loaded {len(self._structs_df)} structures.")

    def _get_struct_dict(self, material_id: str) -> dict:
        self._ensure_structs_loaded()
        return self._structs_df.loc[material_id, "initial_structure"]

    def load_structures(self, material_ids: list[str]) -> list[Structure]:
        """Load pymatgen Structures for the given material IDs."""
        return [Structure.from_dict(self._get_struct_dict(mid)) for mid in material_ids]

    # ------------------------------------------------------------------
    # Download helpers
    # ------------------------------------------------------------------

    @staticmethod
    def download_summary(target_path: str) -> None:
        url = f"https://api.figshare.com/v2/file/download/{SUMMARY_FIGSHARE_ID}"
        print(f"Downloading WBM summary → {target_path}")
        urllib.request.urlretrieve(url, target_path)

    @staticmethod
    def download_structures(target_path: str) -> None:
        url = f"https://api.figshare.com/v2/file/download/{STRUCTS_FIGSHARE_ID}"
        print(f"Downloading WBM structures ({url}) → {target_path}")
        urllib.request.urlretrieve(url, target_path)
