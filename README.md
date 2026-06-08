# GNoME-style Active Learning for Materials Discovery

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![CHGNet](https://img.shields.io/badge/surrogate-CHGNet%20v0.3-orange)
![WBM](https://img.shields.io/badge/benchmark-Matbench%20Discovery-purple)

A GNoME-inspired active learning pipeline for identifying stable inorganic materials using a pre-trained [CHGNet](https://github.com/CederGroupHub/chgnet) surrogate with Monte Carlo Dropout uncertainty estimation. Scales to the full [WBM dataset](https://matbench-discovery.materialsproject.org/) (256,963 crystal structures) used in the Matbench Discovery benchmark.

## Key Results

### WBM benchmark (256K structures)

> With only **2,200 labeled structures** (0.9% of pool), UCB acquisition found **425 stable materials** — vs ~367 expected by random screening. **1.16x Discovery Acceleration Factor.**

| Strategy | Stable found | DAF | Budget |
|---|:-:|:-:|:-:|
| Random | 370 | 1.009x | 2,200 / 256,963 |
| Greedy (μ) | 412 | 1.124x | 2,200 / 256,963 |
| **UCB (λ=1.0)** | **425** | **1.159x** | 2,200 / 256,963 |

Pool: 256,963 WBM structures · 42,825 stable (16.7% prevalence) · DAF = (precision at budget) / prevalence

### Materials Project (2K structures)

> With **20% of the candidate budget labeled**, Greedy and UCB discovered **93–95% of the top-100 most stable structures** — vs 25% for random.

| Strategy | Top-100 Recall | Best Found (eV/atom) |
|---|:-:|:-:|
| Random | 25% | −4.375 |
| Greedy (μ) | **95%** | −4.403 |
| UCB (λ=1.0) | **93%** | −4.403 |

## Method

```
WBM pool (256,963 structures)
         │
         ▼
  CHGNet graph_converter → CrystalGraph objects
         │
   ┌─────┴────────────────────────────────────────┐
   │  Active Learning Loop (20 iterations)        │
   │                                              │
   │  1. Stage 1: point estimate on 25K random    │
   │     candidates (no dropout, fast)            │
   │                                              │
   │  2. Stage 2: MC Dropout (10 passes) on       │
   │     top 5K by Stage 1 → μ, σ per structure  │
   │                                              │
   │  3. Acquisition: select top-100 by           │
   │     UCB score = μ(x) − λ·σ(x)               │
   │                                              │
   │  4. Oracle: look up DFT e_above_hull         │
   │     from WBM summary (simulated DFT)         │
   └──────────────────────────────────────────────┘
         │
         ▼
  Track: DAF per iteration
         stable structures found vs. budget
```

**Surrogate**: CHGNet v0.3.0 (412,525 params). Backbone frozen; MC Dropout (p=0.3) in the MLP readout head produces per-structure uncertainty estimates.

**Key finding**: Fine-tuning the surrogate on the small initial labeled set *hurts* performance on WBM. CHGNet was pre-trained on 700K MP structures and is already well-calibrated — updating on a small biased sample causes catastrophic forgetting. The static pre-trained surrogate with MC Dropout uncertainty achieves consistent 1.1–1.3x DAF improvement.

**Acquisition strategies**:
- `RandomStrategy`: uniform random (baseline)
- `GreedyStrategy`: lowest predicted μ (pure exploitation)
- `UCBStrategy`: `μ(x) − λ·σ(x)`, λ=1.0 (exploration–exploitation)

## Setup

```bash
git clone https://github.com/0xSoftBoi/gnome-materials.git
cd gnome-materials
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

WBM data files are downloaded automatically on first run. For the Materials Project experiment, set your API key:

```bash
export MP_API_KEY="your_key_here"  # get free key at materialsproject.org/api
```

## Reproducing the Experiments

```bash
# WBM active learning campaign — full 256K (~2 hours on CPU)
python3 main.py --experiment wbm

# Quick demo on 20K subset (~15 minutes)
python3 main.py --experiment wbm --wbm-max 20000

# CHGNet surrogate on Materials Project (2K structures, ~30 min)
python3 main.py --experiment chgnet

# Synthetic scaling / lambda sensitivity
python3 main.py --experiment scaling
python3 main.py --experiment lambda
```

Output plots saved to `results/`.

## Project Structure

```
gnome-materials/
├── main.py                        # Experiment entry points
├── data/
│   ├── wbm_dataset.py             # WBM loader, DFT oracle labels
│   ├── mp_dataset_chgnet.py       # Materials Project raw-structure dataset
│   └── dataset.py                 # Synthetic dataset (scaling experiments)
├── model/
│   └── chgnet_surrogate.py        # CHGNet + MC Dropout wrapper
├── active_learning/
│   ├── strategies.py              # Random / Greedy / UCB
│   ├── loop_wbm.py                # WBM AL loop (DAF metric, two-stage shortlist)
│   └── loop_chgnet.py             # MP AL loop (top-100 recall metric)
└── evaluation/
    ├── wbm_metrics.py             # DAF plotting
    └── metrics.py                 # Formation energy / recall plotting
```

## Citation

If this work is useful, please also cite:

```bibtex
@article{riebesell2024matbench,
  title   = {Matbench Discovery: A Framework to Evaluate Machine Learning
             Crystal Stability Predictions},
  author  = {Riebesell, Janosh and others},
  journal = {Nature Machine Intelligence},
  year    = {2024},
  doi     = {10.1038/s42256-024-00954-x}
}

@article{deng2023chgnet,
  title   = {{CHGNet} as a pretrained universal neural network potential for
             charge-informed atomistic modelling},
  author  = {Deng, Bowen and Zhong, Peichen and Jun, KyuJung and Riebesell, Janosh
             and Han, Kevin and Bartel, Christopher J and Ceder, Gerbrand},
  journal = {Nature Machine Intelligence},
  volume  = {5},
  pages   = {1031--1041},
  year    = {2023},
  doi     = {10.1038/s42256-023-00716-3}
}

@inproceedings{gal2016dropout,
  title     = {Dropout as a {B}ayesian Approximation: Representing Model Uncertainty
               in Deep Learning},
  author    = {Gal, Yarin and Ghahramani, Zoubin},
  booktitle = {ICML},
  year      = {2016},
  url       = {https://arxiv.org/abs/1506.02142}
}
```
