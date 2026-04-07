"""Materials Project dataset for real DFT-computed crystal structures."""

import os
from pathlib import Path
import torch
from torch_geometric.data import Data
import numpy as np


# Constants
MP_IN_CHANNELS = 95  # 89 one-hot + 6 physical properties
NEIGHBOR_CUTOFF_ANGSTROM = 5.0
MAX_ATOMS_PER_CELL = 50
E_ABOVE_HULL_THRESHOLD = 0.1  # eV/atom (stability criterion)
DEFAULT_CACHE_FILE = Path(__file__).parent / 'mp_cache' / 'structures.pt'

# 89-element vocabulary (H through Ac)
MP_ELEMENTS = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac'
]
MP_ELEMENT_TO_IDX = {e: i for i, e in enumerate(MP_ELEMENTS)}

# Physical properties for all 89 elements (same 6 as dataset.py, expanded)
# [atomic_num/118, electronegativity/4, atomic_radius/300, valence/8, period/7, group/18]
MP_ELEMENT_PROPERTIES = {
    'H': [1/118, 2.20/4, 53/300, 1/8, 1/7, 1/18],
    'He': [2/118, 0.0/4, 31/300, 0/8, 1/7, 18/18],
    'Li': [3/118, 0.98/4, 167/300, 1/8, 2/7, 1/18],
    'Be': [4/118, 1.57/4, 112/300, 2/8, 2/7, 2/18],
    'B': [5/118, 2.04/4, 82/300, 3/8, 2/7, 13/18],
    'C': [6/118, 2.55/4, 77/300, 4/8, 2/7, 14/18],
    'N': [7/118, 3.04/4, 71/300, 3/8, 2/7, 15/18],
    'O': [8/118, 3.44/4, 66/300, 2/8, 2/7, 16/18],
    'F': [9/118, 3.98/4, 64/300, 1/8, 2/7, 17/18],
    'Ne': [10/118, 0.0/4, 62/300, 0/8, 2/7, 18/18],
    'Na': [11/118, 0.93/4, 186/300, 1/8, 3/7, 1/18],
    'Mg': [12/118, 1.31/4, 160/300, 2/8, 3/7, 2/18],
    'Al': [13/118, 1.61/4, 143/300, 3/8, 3/7, 13/18],
    'Si': [14/118, 1.90/4, 118/300, 4/8, 3/7, 14/18],
    'P': [15/118, 2.19/4, 110/300, 3/8, 3/7, 15/18],
    'S': [16/118, 2.58/4, 104/300, 2/8, 3/7, 16/18],
    'Cl': [17/118, 3.16/4, 99/300, 1/8, 3/7, 17/18],
    'Ar': [18/118, 0.0/4, 97/300, 0/8, 3/7, 18/18],
    'K': [19/118, 0.82/4, 227/300, 1/8, 4/7, 1/18],
    'Ca': [20/118, 1.00/4, 197/300, 2/8, 4/7, 2/18],
    'Sc': [21/118, 1.36/4, 162/300, 3/8, 4/7, 3/18],
    'Ti': [22/118, 1.54/4, 147/300, 4/8, 4/7, 4/18],
    'V': [23/118, 1.63/4, 134/300, 5/8, 4/7, 5/18],
    'Cr': [24/118, 1.66/4, 128/300, 6/8, 4/7, 6/18],
    'Mn': [25/118, 1.55/4, 127/300, 7/8, 4/7, 7/18],
    'Fe': [26/118, 1.83/4, 126/300, 8/8, 4/7, 8/18],
    'Co': [27/118, 1.88/4, 125/300, 7/8, 4/7, 9/18],
    'Ni': [28/118, 1.91/4, 124/300, 8/8, 4/7, 10/18],
    'Cu': [29/118, 1.90/4, 128/300, 1/8, 4/7, 11/18],
    'Zn': [30/118, 1.65/4, 134/300, 2/8, 4/7, 12/18],
    'Ga': [31/118, 1.81/4, 141/300, 3/8, 4/7, 13/18],
    'Ge': [32/118, 2.01/4, 122/300, 4/8, 4/7, 14/18],
    'As': [33/118, 2.18/4, 121/300, 3/8, 4/7, 15/18],
    'Se': [34/118, 2.55/4, 117/300, 2/8, 4/7, 16/18],
    'Br': [35/118, 2.96/4, 114/300, 1/8, 4/7, 17/18],
    'Kr': [36/118, 0.0/4, 110/300, 0/8, 4/7, 18/18],
    'Rb': [37/118, 0.82/4, 248/300, 1/8, 5/7, 1/18],
    'Sr': [38/118, 0.95/4, 215/300, 2/8, 5/7, 2/18],
    'Y': [39/118, 1.22/4, 180/300, 3/8, 5/7, 3/18],
    'Zr': [40/118, 1.33/4, 160/300, 4/8, 5/7, 4/18],
    'Nb': [41/118, 1.60/4, 146/300, 5/8, 5/7, 5/18],
    'Mo': [42/118, 2.16/4, 139/300, 6/8, 5/7, 6/18],
    'Tc': [43/118, 1.90/4, 136/300, 7/8, 5/7, 7/18],
    'Ru': [44/118, 2.20/4, 134/300, 8/8, 5/7, 8/18],
    'Rh': [45/118, 2.28/4, 134/300, 8/8, 5/7, 9/18],
    'Pd': [46/118, 2.20/4, 137/300, 8/8, 5/7, 10/18],
    'Ag': [47/118, 1.93/4, 144/300, 1/8, 5/7, 11/18],
    'Cd': [48/118, 1.69/4, 144/300, 2/8, 5/7, 12/18],
    'In': [49/118, 1.78/4, 167/300, 3/8, 5/7, 13/18],
    'Sn': [50/118, 1.96/4, 140/300, 4/8, 5/7, 14/18],
    'Sb': [51/118, 2.05/4, 140/300, 3/8, 5/7, 15/18],
    'Te': [52/118, 2.10/4, 138/300, 2/8, 5/7, 16/18],
    'I': [53/118, 2.66/4, 139/300, 1/8, 5/7, 17/18],
    'Xe': [54/118, 0.0/4, 140/300, 0/8, 5/7, 18/18],
    'Cs': [55/118, 0.79/4, 265/300, 1/8, 6/7, 1/18],
    'Ba': [56/118, 0.89/4, 222/300, 2/8, 6/7, 2/18],
    'La': [57/118, 1.10/4, 187/300, 3/8, 6/7, 3/18],
    'Ce': [58/118, 1.12/4, 182/300, 3/8, 6/7, 3/18],
    'Pr': [59/118, 1.13/4, 182/300, 3/8, 6/7, 3/18],
    'Nd': [60/118, 1.14/4, 181/300, 3/8, 6/7, 3/18],
    'Pm': [61/118, 1.13/4, 181/300, 3/8, 6/7, 3/18],
    'Sm': [62/118, 1.17/4, 180/300, 3/8, 6/7, 3/18],
    'Eu': [63/118, 1.20/4, 199/300, 3/8, 6/7, 3/18],
    'Gd': [64/118, 1.20/4, 179/300, 3/8, 6/7, 3/18],
    'Tb': [65/118, 1.10/4, 177/300, 3/8, 6/7, 3/18],
    'Dy': [66/118, 1.22/4, 177/300, 3/8, 6/7, 3/18],
    'Ho': [67/118, 1.23/4, 176/300, 3/8, 6/7, 3/18],
    'Er': [68/118, 1.24/4, 175/300, 3/8, 6/7, 3/18],
    'Tm': [69/118, 1.25/4, 174/300, 3/8, 6/7, 3/18],
    'Yb': [70/118, 1.10/4, 194/300, 3/8, 6/7, 3/18],
    'Lu': [71/118, 1.27/4, 173/300, 3/8, 6/7, 3/18],
    'Hf': [72/118, 1.30/4, 159/300, 4/8, 6/7, 4/18],
    'Ta': [73/118, 1.50/4, 146/300, 5/8, 6/7, 5/18],
    'W': [74/118, 2.36/4, 139/300, 6/8, 6/7, 6/18],
    'Re': [75/118, 1.90/4, 137/300, 7/8, 6/7, 7/18],
    'Os': [76/118, 2.20/4, 135/300, 8/8, 6/7, 8/18],
    'Ir': [77/118, 2.20/4, 136/300, 8/8, 6/7, 9/18],
    'Pt': [78/118, 2.28/4, 139/300, 8/8, 6/7, 10/18],
    'Au': [79/118, 2.54/4, 144/300, 1/8, 6/7, 11/18],
    'Hg': [80/118, 2.00/4, 151/300, 2/8, 6/7, 12/18],
    'Tl': [81/118, 1.62/4, 171/300, 3/8, 6/7, 13/18],
    'Pb': [82/118, 2.33/4, 175/300, 4/8, 6/7, 14/18],
    'Bi': [83/118, 2.02/4, 182/300, 3/8, 6/7, 15/18],
    'Po': [84/118, 2.0/4, 167/300, 2/8, 6/7, 16/18],
    'At': [85/118, 2.2/4, 140/300, 1/8, 6/7, 17/18],
    'Rn': [86/118, 0.0/4, 120/300, 0/8, 6/7, 18/18],
    'Fr': [87/118, 0.7/4, 348/300, 1/8, 7/7, 1/18],
    'Ra': [88/118, 0.9/4, 283/300, 2/8, 7/7, 2/18],
    'Ac': [89/118, 1.1/4, 186/300, 3/8, 7/7, 3/18],
}


class MPCrystalDataset:
    """Real Materials Project crystal structures with DFT formation energies."""

    def __init__(self, cache_path=None, force_download=False, max_structures=None):
        """
        Args:
            cache_path: path to cache file (default: data/mp_cache/structures.pt)
            force_download: if True, re-download even if cache exists
            max_structures: max structures to load (slices after loading, won't corrupt cache)
        """
        self.cache_path = Path(cache_path or DEFAULT_CACHE_FILE)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_structures = max_structures
        self.structures = self._load_or_download(force_download)

    def _load_or_download(self, force_download=False):
        """Load from cache or download from MP."""
        if self.cache_path.exists() and not force_download:
            print(f"Loading cached structures from {self.cache_path}...")
            structures = torch.load(self.cache_path, weights_only=False)
            print(f"  Loaded {len(structures)} structures")
        else:
            print(f"Downloading structures from Materials Project...")
            structures = self._download_from_mp()
            print(f"Saving cache to {self.cache_path}...")
            torch.save(structures, self.cache_path)
            print(f"  Cached {len(structures)} structures")

        # Slice if max_structures specified (never corrupts cache)
        if self.max_structures and len(structures) > self.max_structures:
            structures = structures[:self.max_structures]
            print(f"  Sliced to {len(structures)} structures (max_structures={self.max_structures})")

        return structures

    def _download_from_mp(self):
        """Download all stable structures from Materials Project."""
        from mp_api.client import MPRester

        api_key = os.getenv('MP_API_KEY')
        if not api_key:
            raise EnvironmentError(
                "MP_API_KEY environment variable not set. "
                "Get your API key from https://materialsproject.org/api and set:\n"
                "  export MP_API_KEY='your-api-key'"
            )

        structures = []
        with MPRester(api_key) as mpr:
            print("Querying Materials Project (energy_above_hull <= 0.1 eV/atom)...")
            docs = mpr.materials.summary.search(
                energy_above_hull=(0, E_ABOVE_HULL_THRESHOLD),
                fields=["material_id", "structure", "formation_energy_per_atom"]
            )

            for i, doc in enumerate(docs):
                if i % 5000 == 0:
                    print(f"  Downloaded {i} structures...")

                if doc.structure is None:
                    continue

                try:
                    data = self._structure_to_data(doc.structure, doc.formation_energy_per_atom)
                    if data is not None:
                        structures.append(data)
                except Exception as e:
                    continue

        print(f"  Total valid structures: {len(structures)}")
        return structures

    def _structure_to_data(self, structure, formation_energy_per_atom):
        """Convert pymatgen Structure to torch_geometric Data."""
        from pymatgen.core import Structure

        # Skip large structures
        if structure.num_sites > MAX_ATOMS_PER_CELL:
            return None

        n_atoms = structure.num_sites

        # Node features: [89-dim one-hot, 6-dim physical properties]
        x = torch.zeros(n_atoms, MP_IN_CHANNELS, dtype=torch.float32)

        for i, site in enumerate(structure.sites):
            # Get element symbol (strip oxidation state: Fe2+ → Fe)
            symbol = site.specie.symbol
            if symbol not in MP_ELEMENT_TO_IDX:
                return None  # Unknown element

            # One-hot encoding
            elem_idx = MP_ELEMENT_TO_IDX[symbol]
            x[i, elem_idx] = 1.0

            # Physical properties
            if symbol in MP_ELEMENT_PROPERTIES:
                props = MP_ELEMENT_PROPERTIES[symbol]
                x[i, 89:95] = torch.tensor(props, dtype=torch.float32)

        # Edge list: structure neighbors within cutoff
        neighbor_list = structure.get_neighbor_list(r=NEIGHBOR_CUTOFF_ANGSTROM)
        if not neighbor_list[0]:  # No neighbors found
            return None

        center_indices, neighbor_indices, _, _ = neighbor_list
        edge_index = torch.tensor(
            [center_indices, neighbor_indices],
            dtype=torch.int64
        )

        # Target: formation energy
        y = torch.tensor([formation_energy_per_atom], dtype=torch.float32)

        return Data(x=x, edge_index=edge_index, y=y)

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        """Get single structure or list of structures."""
        if isinstance(idx, list):
            return [self.structures[i] for i in idx]
        return self.structures[idx]
