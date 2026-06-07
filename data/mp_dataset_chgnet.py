"""Materials Project dataset storing raw pymatgen structures for CHGNet fine-tuning."""

import os
from pathlib import Path
import torch
import numpy as np

DEFAULT_RAW_CACHE = Path(__file__).parent / 'mp_cache' / 'structures_raw.pt'
E_ABOVE_HULL_THRESHOLD = 0.1
MAX_ATOMS = 50


class MPStructureDataset:
    """Stores raw pymatgen structures + formation energies from Materials Project."""

    def __init__(self, cache_path=None, force_download=False, max_structures=None):
        self.cache_path = Path(cache_path or DEFAULT_RAW_CACHE)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_structures = max_structures
        self.data = self._load_or_download(force_download)

    def _load_or_download(self, force_download):
        if self.cache_path.exists() and not force_download:
            print(f"Loading cached raw structures from {self.cache_path}...")
            data = torch.load(self.cache_path, weights_only=False)
            print(f"  Loaded {len(data)} (structure, energy) pairs")
        else:
            print("Downloading raw structures from Materials Project...")
            data = self._download()
            torch.save(data, self.cache_path)
            print(f"  Cached {len(data)} structures to {self.cache_path}")

        if self.max_structures and len(data) > self.max_structures:
            np.random.seed(42)
            indices = np.random.choice(len(data), self.max_structures, replace=False)
            data = [data[i] for i in sorted(indices)]
            print(f"  Sampled {len(data)} structures (max_structures={self.max_structures})")

        return data

    def _download(self):
        from mp_api.client import MPRester

        api_key = os.getenv('MP_API_KEY')
        if not api_key:
            raise EnvironmentError(
                "MP_API_KEY not set. Get key at https://materialsproject.org/api"
            )

        data = []
        with MPRester(api_key) as mpr:
            print(f"  Querying MP (energy_above_hull <= {E_ABOVE_HULL_THRESHOLD})...")
            docs = mpr.materials.summary.search(
                energy_above_hull=(0, E_ABOVE_HULL_THRESHOLD),
                fields=["structure", "formation_energy_per_atom"],
            )
            for i, doc in enumerate(docs):
                if i % 10000 == 0:
                    print(f"  Processing {i}...")
                if doc.structure is None:
                    continue
                if doc.structure.num_sites > MAX_ATOMS:
                    continue
                if doc.formation_energy_per_atom is None:
                    continue
                try:
                    data.append((doc.structure, float(doc.formation_energy_per_atom)))
                except Exception:
                    continue

        print(f"  Total valid: {len(data)}")
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return [self.data[i] for i in idx]
        return self.data[idx]

    def get_structure(self, idx):
        return self.data[idx][0]

    def get_energy(self, idx):
        return self.data[idx][1]

    def get_structures(self, indices):
        return [self.data[i][0] for i in indices]

    def get_energies(self, indices):
        return [self.data[i][1] for i in indices]
