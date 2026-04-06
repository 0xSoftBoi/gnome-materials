import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader


class GNNRegressor(nn.Module):
    """Graph Neural Network with dropout for uncertainty estimation."""

    def __init__(self, in_channels, hidden_dim=64, dropout_p=0.3):
        super().__init__()
        self.dropout_p = dropout_p

        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, batch):
        # Graph convolutions with dropout enabled at all times
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_p, training=True)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_p, training=True)

        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_p, training=True)

        # Global pooling
        x = global_mean_pool(x, batch)

        # MLP head
        x = self.mlp(x)
        return x

    def predict_with_uncertainty(self, loader, n_passes=20, device='cpu'):
        """MC Dropout: multiple forward passes to estimate μ and σ."""
        self.eval()
        all_preds = []

        with torch.no_grad():
            for _ in range(n_passes):
                batch_preds = []
                for batch in loader:
                    batch = batch.to(device)
                    out = self.forward(batch.x, batch.edge_index, batch.batch)
                    batch_preds.append(out.cpu().numpy())
                all_preds.append(batch_preds)

        # Stack predictions: shape (n_passes, n_batches, batch_size, 1)
        means = []
        stds = []

        n_batches = len(loader)
        for batch_idx in range(n_batches):
            batch_predictions = [all_preds[p][batch_idx] for p in range(n_passes)]
            batch_predictions = torch.tensor(batch_predictions, dtype=torch.float32)  # (n_passes, batch_size, 1)

            mu = batch_predictions.mean(dim=0)  # (batch_size, 1)
            sigma = batch_predictions.std(dim=0)  # (batch_size, 1)

            means.append(mu)
            stds.append(sigma)

        means = torch.cat(means, dim=0)  # (total_samples, 1)
        stds = torch.cat(stds, dim=0)

        return means.squeeze(-1).numpy(), stds.squeeze(-1).numpy()


def train_model(model, train_loader, val_loader, device='cpu', epochs=50, lr=1e-3):
    """Train the GNN model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y.unsqueeze(-1))

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out, batch.y.unsqueeze(-1))
                val_loss += loss.item()

        val_loss /= len(val_loader)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    return model
