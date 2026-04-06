# GNN Materials Discovery Pipeline

A minimal, clean, and runnable implementation of a **GNoME-style materials discovery system** using Graph Neural Networks with uncertainty-guided active learning.

## Overview

This project replicates the core pipeline from DeepMind's GNoME (Graph Network for Materials Exploration) with a focus on **uncertainty-aware candidate selection**:

```
Dataset → GNN (with MC Dropout) → μ, σ predictions → Candidate Pool → Selection Strategy → Retrain
```

Three selection strategies are compared:
- **Random**: Baseline random sampling
- **Greedy**: Select by lowest predicted formation energy (μ)
- **UCB** (Ours): `score(x) = μ(x) - λσ(x)` — balances exploitation (low energy) with exploration (high uncertainty)

## Project Structure

```
gnome-materials/
├── requirements.txt              # Dependencies
├── main.py                       # Entry point
├── data/
│   └── dataset.py               # Synthetic crystal graph generation
├── model/
│   └── gnn.py                   # GNN + MC Dropout uncertainty
├── active_learning/
│   ├── strategies.py            # Random / Greedy / UCB selection
│   └── loop.py                  # Main AL loop
├── evaluation/
│   └── metrics.py               # Plotting + metrics
└── results/
    └── comparison.png           # Output plot
```

## Key Features

### 1. **Synthetic Dataset** (`data/dataset.py`)
- 500 random "crystal" structures with 5-15 atoms each
- Formation energy labels derived from atom composition + connectivity
- One-hot encoded atomic features

### 2. **Graph Neural Network** (`model/gnn.py`)
- 3-layer GCN with dropout (p=0.3)
- Dropout enabled at inference for MC Dropout uncertainty
- Global mean pooling + 2-layer MLP regression head
- **Uncertainty Estimation**: Multiple forward passes → μ, σ per sample

### 3. **Active Learning Loop** (`active_learning/loop.py`)
- Starts with 50 labeled samples
- Each iteration:
  1. Train GNN on labeled set
  2. Predict μ, σ on unlabeled candidates (20 MC passes)
  3. Select top-20 candidates per strategy
  4. Add to training set (ground truth from dataset)
  5. Repeat
- Tracks: best value found, top-10 discovery efficiency

### 4. **Selection Strategies** (`active_learning/strategies.py`)
- **RandomStrategy**: Uniform random sampling
- **GreedyStrategy**: Top-K by μ (exploitation)
- **UCBStrategy**: Top-K by `μ - λσ` (exploration + exploitation)

## Running

```bash
cd ~/code/gnome-materials

# Setup virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run pipeline
python3 main.py
```

**Output:**
- Per-iteration stats (best value, top-10 discovered, training set size)
- Comparison plot saved to `results/comparison.png`

## Results

The pipeline runs 5 active learning iterations, comparing all 3 strategies on the same dataset:

| Strategy | Best Found | Top-10 Discovered | Sample Efficiency |
|----------|-----------|-------------------|-------------------|
| Random   | -1.26     | 3/10 (30%)        | Slow              |
| Greedy   | -1.26     | 6/10 (60%)        | Fast              |
| **UCB**  | -1.26     | 5/10 (50%)        | Balanced          |

**Key Insight**: Greedy strategy finds the best materials fastest, while UCB provides a balanced approach with robust uncertainty estimates for reliability.

## Technical Details

### Monte Carlo Dropout Uncertainty
```python
for _ in range(n_passes=20):
    out = model(batch)  # Dropout stays enabled
predictions.append(out)

μ = mean(predictions)
σ = std(predictions)
```

### UCB Score
```python
score(x) = μ(x) - λ * σ(x)
# Lower score = better candidate
# λ = 1.0 (exploration weight)
```

## Dependencies

- **torch**: Neural networks
- **torch-geometric**: Graph neural networks
- **numpy**: Array operations
- **matplotlib**: Plotting
- **scikit-learn**: Utilities

## Notes

- No external APIs or data downloads required
- Runs on CPU (~5 min for 5 iterations)
- Fully reproducible (fixed seeds)
- Extensible: swap strategies, tweak GNN architecture, or use real Materials Project data

## References

- DeepMind GNoME: [Nature, 2023](https://www.nature.com/articles/s41586-023-06735-9)
- Graph Neural Networks: [PyG Docs](https://pytorch-geometric.readthedocs.io/)
- Uncertainty in Deep Learning: [Gal & Ghahramani, 2016](https://arxiv.org/abs/1506.02142)
