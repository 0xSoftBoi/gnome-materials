import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


ELEMENTS = ['H', 'C', 'N', 'O', 'Si', 'P', 'S', 'Cl', 'Fe', 'Cu',
            'Zn', 'Ag', 'Pt', 'Au', 'Li', 'Na', 'K', 'Ca', 'Al', 'Mg']
ELEMENT_TO_IDX = {e: i for i, e in enumerate(ELEMENTS)}

# Physical properties for each element (v2 enhancement)
ELEMENT_PROPERTIES = {
    'H':  {'atomic_num': 1,  'electronegativity': 2.20, 'atomic_radius': 53,  'valence': 1, 'period': 1, 'group': 1},
    'C':  {'atomic_num': 6,  'electronegativity': 2.55, 'atomic_radius': 77,  'valence': 4, 'period': 2, 'group': 14},
    'N':  {'atomic_num': 7,  'electronegativity': 3.04, 'atomic_radius': 75,  'valence': 5, 'period': 2, 'group': 15},
    'O':  {'atomic_num': 8,  'electronegativity': 3.44, 'atomic_radius': 73,  'valence': 6, 'period': 2, 'group': 16},
    'Si': {'atomic_num': 14, 'electronegativity': 1.90, 'atomic_radius': 111, 'valence': 4, 'period': 3, 'group': 14},
    'P':  {'atomic_num': 15, 'electronegativity': 2.19, 'atomic_radius': 106, 'valence': 5, 'period': 3, 'group': 15},
    'S':  {'atomic_num': 16, 'electronegativity': 2.58, 'atomic_radius': 103, 'valence': 6, 'period': 3, 'group': 16},
    'Cl': {'atomic_num': 17, 'electronegativity': 3.16, 'atomic_radius': 99,  'valence': 7, 'period': 3, 'group': 17},
    'Fe': {'atomic_num': 26, 'electronegativity': 1.83, 'atomic_radius': 126, 'valence': 2, 'period': 4, 'group': 8},
    'Cu': {'atomic_num': 29, 'electronegativity': 1.90, 'atomic_radius': 128, 'valence': 1, 'period': 4, 'group': 11},
    'Zn': {'atomic_num': 30, 'electronegativity': 1.65, 'atomic_radius': 122, 'valence': 2, 'period': 4, 'group': 12},
    'Ag': {'atomic_num': 47, 'electronegativity': 1.93, 'atomic_radius': 165, 'valence': 1, 'period': 5, 'group': 11},
    'Pt': {'atomic_num': 78, 'electronegativity': 2.28, 'atomic_radius': 139, 'valence': 2, 'period': 6, 'group': 10},
    'Au': {'atomic_num': 79, 'electronegativity': 2.54, 'atomic_radius': 144, 'valence': 1, 'period': 6, 'group': 11},
    'Li': {'atomic_num': 3,  'electronegativity': 0.98, 'atomic_radius': 167, 'valence': 1, 'period': 2, 'group': 1},
    'Na': {'atomic_num': 11, 'electronegativity': 0.93, 'atomic_radius': 186, 'valence': 1, 'period': 3, 'group': 1},
    'K':  {'atomic_num': 19, 'electronegativity': 0.82, 'atomic_radius': 227, 'valence': 1, 'period': 4, 'group': 1},
    'Ca': {'atomic_num': 20, 'electronegativity': 1.00, 'atomic_radius': 197, 'valence': 2, 'period': 4, 'group': 2},
    'Al': {'atomic_num': 13, 'electronegativity': 1.61, 'atomic_radius': 143, 'valence': 3, 'period': 3, 'group': 13},
    'Mg': {'atomic_num': 12, 'electronegativity': 1.31, 'atomic_radius': 160, 'valence': 2, 'period': 3, 'group': 2},
}


class SyntheticCrystalDataset:
    """Generate synthetic crystal graphs with formation energy labels."""

    def __init__(self, n_samples=500, seed=42):
        self.n_samples = n_samples
        self.seed = seed
        self.data_list = self._generate()

    def _generate(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        data_list = []
        for _ in range(self.n_samples):
            # v2: 4-20 atoms (wider range than v1's 5-15)
            n_atoms = np.random.randint(4, 21)
            atom_indices = np.random.choice(len(ELEMENTS), size=n_atoms, replace=True)
            atom_symbols = [ELEMENTS[i] for i in atom_indices]

            # v2: 26-dim features = 20-dim one-hot + 6 physical properties
            x = torch.zeros(n_atoms, 26, dtype=torch.float)

            # One-hot encoding (first 20 dims)
            for i, idx in enumerate(atom_indices):
                x[i, idx] = 1.0

            # Physical properties (next 6 dims, normalized)
            for i, symbol in enumerate(atom_symbols):
                props = ELEMENT_PROPERTIES[symbol]
                x[i, 20] = props['atomic_num'] / 118.0          # normalized
                x[i, 21] = props['electronegativity'] / 4.0
                x[i, 22] = props['atomic_radius'] / 300.0
                x[i, 23] = props['valence'] / 8.0
                x[i, 24] = props['period'] / 7.0
                x[i, 25] = props['group'] / 18.0

            # v2: Distance-based connectivity (more physically motivated)
            coords = np.random.uniform(0, 10, size=(n_atoms, 3))
            distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)

            # kNN-style: connect to k=4 nearest neighbors
            k = min(4, n_atoms - 1)
            edge_indices = []
            for i in range(n_atoms):
                nearest = np.argsort(distances[i])[1:k+1]  # exclude self
                for j in nearest:
                    if i < j:  # avoid duplicates
                        edge_indices.append([i, j])

            if not edge_indices:
                u, v = np.random.choice(n_atoms, 2, replace=False)
                edge_indices.append([min(u, v), max(u, v)])

            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

            # v2: Physics-grounded energy formula
            electronegativities = np.array([ELEMENT_PROPERTIES[s]['electronegativity'] for s in atom_symbols])
            valences = np.array([ELEMENT_PROPERTIES[s]['valence'] for s in atom_symbols])
            periods = np.array([ELEMENT_PROPERTIES[s]['period'] for s in atom_symbols])

            en_variance = np.std(electronegativities)      # ionic character
            avg_electronegativity = np.mean(electronegativities)
            avg_period = np.mean(periods)
            avg_valence = np.mean(valences)
            coordination = edge_index.shape[1] / n_atoms if edge_index.shape[1] > 0 else 0

            # Energy formula: lower = more stable
            energy = (-1.5 * en_variance                       # ionic bonds
                     - 0.8 * avg_electronegativity             # stability
                     + 0.4 * avg_period                        # heavier = less stable
                     - 0.3 * np.log(1.0 + avg_valence)         # bonding capacity
                     + 0.2 * coordination)                     # connectivity
            energy += np.random.normal(0, 0.05)               # reduced noise

            y = torch.tensor([energy], dtype=torch.float)

            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)

        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return [self.data_list[i] for i in idx]
        return self.data_list[idx]


def load_dataset(n_samples=500, batch_size=32, train_ratio=0.8, seed=42):
    """Load dataset and split into train/val/test."""
    dataset = SyntheticCrystalDataset(n_samples=n_samples, seed=seed)

    n_train = int(train_ratio * len(dataset))
    n_val = (len(dataset) - n_train) // 2

    torch.manual_seed(seed)
    indices = torch.randperm(len(dataset))

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]

    train_loader = DataLoader(dataset[train_idx.tolist()], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset[val_idx.tolist()], batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, dataset


def generate_candidates(dataset, n_candidates=200, seed=42):
    """Generate candidate pool via element substitution."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    candidates = []
    n_elements = len(ELEMENTS)

    for _ in range(n_candidates):
        # Pick a random sample and mutate
        base_idx = np.random.randint(0, len(dataset))
        base_data = dataset[base_idx]

        # Clone and mutate atoms
        x = base_data.x.clone()
        edge_index = base_data.edge_index.clone()

        # Substitute 1-3 random atoms
        n_mutations = np.random.randint(1, 4)
        n_atoms = x.shape[0]
        mutation_indices = np.random.choice(n_atoms, size=min(n_mutations, n_atoms), replace=False)

        for idx in mutation_indices:
            new_elem = np.random.randint(0, n_elements)
            x[idx, :] = 0
            x[idx, new_elem] = 1.0

        candidate = Data(x=x, edge_index=edge_index, y=torch.tensor([0.0]))
        candidates.append(candidate)

    return candidates
