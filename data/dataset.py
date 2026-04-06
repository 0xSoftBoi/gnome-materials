import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


ELEMENTS = ['H', 'C', 'N', 'O', 'Si', 'P', 'S', 'Cl', 'Fe', 'Cu',
            'Zn', 'Ag', 'Pt', 'Au', 'Li', 'Na', 'K', 'Ca', 'Al', 'Mg']
ELEMENT_TO_IDX = {e: i for i, e in enumerate(ELEMENTS)}


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
            # Random graph: 5-15 atoms per structure
            n_atoms = np.random.randint(5, 16)
            atom_indices = np.random.choice(len(ELEMENTS), size=n_atoms, replace=True)

            # Features: one-hot encode atoms
            x = torch.zeros(n_atoms, len(ELEMENTS), dtype=torch.float)
            for i, idx in enumerate(atom_indices):
                x[i, idx] = 1.0

            # Edges: random connectivity (sparse)
            n_edges = np.random.randint(n_atoms, min(n_atoms * 3, 50))
            edge_indices = []
            for _ in range(n_edges):
                u = np.random.randint(0, n_atoms)
                v = np.random.randint(0, n_atoms)
                if u != v:
                    edge_indices.append([u, v])

            if not edge_indices:
                # Ensure at least one edge
                u, v = np.random.choice(n_atoms, 2, replace=False)
                edge_indices.append([u, v])

            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

            # Ground truth: formation energy = f(composition, connectivity) + noise
            # Atoms with lower atomic number tend to be more stable
            composition_score = np.mean([atom_indices[i] / len(ELEMENTS) for i in range(n_atoms)])
            connectivity_score = edge_index.shape[1] / n_atoms  # edges per atom

            # Formation energy (lower = more stable)
            energy = -2.0 * (1.0 - composition_score) + 0.5 * connectivity_score
            energy += np.random.normal(0, 0.1)  # noise

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
