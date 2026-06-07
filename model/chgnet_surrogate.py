"""CHGNet pre-trained surrogate with MC dropout uncertainty for active learning."""

import random
import numpy as np
import torch
import torch.nn as nn
from chgnet.model import CHGNet


class CHGNetSurrogate:
    """
    Pre-trained CHGNet backbone + MC dropout for uncertainty-aware AL.

    Strategy:
      - Freeze atom/bond/angle conv layers (expensive to train, good priors)
      - Fine-tune site_wise, readout_norm, and final mlp each iteration
      - Enable p=0.3 dropout in mlp for MC dropout uncertainty estimation
    """

    def __init__(self, dropout_p=0.3, freeze_backbone=True):
        # Force CPU at load time: MPS and CPU have equal throughput for CHGNet
        # (small GNN), but MPS causes silent OOM kills on 16 GB unified memory.
        self.model = CHGNet.load(use_device='cpu')
        self.device = torch.device('cpu')
        self.dropout_p = dropout_p

        # Set dropout p in the final MLP (was 0 in pre-trained weights)
        for mod in self.model.mlp.modules():
            if isinstance(mod, nn.Dropout):
                mod.p = dropout_p

        if freeze_backbone:
            frozen_prefixes = ('atom_conv_layers', 'bond_conv_layers',
                               'angle_layers', 'atom_embedding',
                               'bond_basis_expansion', 'bond_embedding',
                               'bond_weights_ag', 'bond_weights_bg',
                               'angle_basis_expansion', 'angle_embedding')
            for name, param in self.model.named_parameters():
                if any(name.startswith(p) for p in frozen_prefixes):
                    param.requires_grad = False

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"CHGNet: {trainable:,} / {total:,} params trainable "
              f"({'backbone frozen' if freeze_backbone else 'full fine-tune'})")

    def fine_tune(self, graphs, energies, epochs=20, lr=1e-3,
                  batch_size=16, verbose=True):
        """Fine-tune on pre-computed graphs and energies — manual training loop."""
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=lr)
        criterion = nn.MSELoss()

        targets = torch.tensor(energies, dtype=torch.float32, device=self.device)

        n = len(graphs)
        idx = list(range(n))

        self.model.train()
        for epoch in range(epochs):
            random.shuffle(idx)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, n, batch_size):
                batch_idx = idx[i:i + batch_size]
                batch_graphs = [graphs[j] for j in batch_idx]
                batch_targets = targets[batch_idx]

                optimizer.zero_grad()
                out = self.model(batch_graphs, task='e')
                pred = out['e']
                loss = criterion(pred, batch_targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            if verbose and (epoch + 1) % 5 == 0:
                print(f"    Epoch {epoch+1}/{epochs}, loss={epoch_loss/n_batches:.4f}")

    def precompute_graphs(self, structures):
        """Convert structures to CrystalGraphs on CPU."""
        return [self.model.graph_converter(s) for s in structures]

    def predict_with_uncertainty(self, graphs, n_passes=20):
        """
        MC dropout inference over pre-computed CrystalGraphs.
        Graphs live on CPU; each batch is moved to device transiently.

        Returns:
            means (np.ndarray), stds (np.ndarray), shape (N,)
        """
        self.model.eval()
        self.model.mlp.train()  # keep dropout active in head only

        batch_size = 256
        all_pass_preds = []

        with torch.no_grad():
            for _ in range(n_passes):
                pass_preds = []
                for i in range(0, len(graphs), batch_size):
                    out = self.model(graphs[i:i + batch_size], task='e')
                    pass_preds.extend(out['e'].tolist())
                all_pass_preds.append(pass_preds)

        self.model.eval()

        all_pass_preds = np.array(all_pass_preds)
        return all_pass_preds.mean(axis=0), all_pass_preds.std(axis=0)
