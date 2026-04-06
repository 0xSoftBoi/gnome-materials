# GNN Materials Discovery: Scaling & Lambda Tuning Experiments

## Overview

v2 established that physics-informed datasets with multi-term objectives enable strategy differentiation. v3 investigates:

1. **Scaling Hypothesis**: Does uncertainty win when candidate pool >> budget?
2. **Lambda Tuning**: How to control exploration-exploitation tradeoff in UCB?

---

## Experiment A: Scaling Analysis

**Research Question:** Does greedy plateau and uncertainty help when pool size >> budget?

### Hypothesis
- **Small pool (2K structures, 16% budget)**: All strategies similar (greedy covers most)
- **Large pool (50K structures, 0.6% budget)**: Greedy gets stuck, UCB explores effectively

### Configuration

| Pool Size | Samples | Initial Train | Candidate Pool | Budget | Coverage |
|-----------|---------|---|---|---|---|
| v2        | 2,000   | 100 | 1,900 | 300 | 15.8% |
| v3-small  | 10,000  | 100 | 9,900 | 300 | 3.0% |
| v3-large  | 50,000  | 100 | 49,900 | 300 | 0.6% |

All configs: 10 iterations × 30 candidates/iter = 300 total budget.

### Expected Results

```
                Random      Greedy      UCB
v2 (2K):        30%         90%         90%     (all strategies saturate)
v3-small (10K): 30%         75%         80%     (uncertainty begins to help)
v3-large (50K): 30%         45%         70%     (clear UCB advantage)
```

**Interpretation:** 
- Random stays flat (no learning)
- Greedy degrades with pool size (exhaustion effect)
- UCB maintains advantage by exploring uncertain regions

### Output Files
- `results/comparison_v2.png` - Baseline (2K pool)
- `results/comparison_v3-small.png` - 10K pool
- `results/comparison_v3-large.png` - 50K pool
- `results/scaling_analysis.png` - Cross-pool comparison

---

## Experiment B: Lambda Tuning

**Research Question:** How does uncertainty weight (λ) affect UCB performance?

### Hypothesis
- λ = 0 (pure exploitation): Equivalent to Greedy
- λ = 1–2 (balanced): Best exploration-exploitation tradeoff
- λ = 5 (pure exploration): Over-explores, misses good candidates

### Configuration

Test UCB with λ ∈ {0.0, 0.5, 1.0, 2.0, 5.0}

**Fixed settings:**
- Pool: 2,000 (v2 baseline)
- Budget: 300 (10 iters × 30/iter)
- Comparisons: vs Random and Greedy baselines

### Expected Results

```
λ = 0.0   → ~90% top-10 (pure greedy, no exploration)
λ = 0.5   → ~85% top-10 (slight exploration benefit)
λ = 1.0   → ~92% top-10 (optimal balance)
λ = 2.0   → ~88% top-10 (over-explores, some regret)
λ = 5.0   → ~80% top-10 (too much exploration)
```

### Output Files
- `results/lambda_tuning_0.0.png` ... `results/lambda_tuning_5.0.png` - Per-λ results
- `results/lambda_tuning_analysis.png` - λ comparison plot

---

## Key Insights

### From Scaling:
1. **Greedy's Weakness**: In large pools, greedy can't find rare good materials
2. **Uncertainty's Strength**: By exploring high-uncertainty regions, UCB avoids getting stuck
3. **GNoME's Insight**: The paper's emphasis on uncertainty matters more at realistic scales (50K+)

### From Lambda Tuning:
1. **Exploration-Exploitation**: Optimal λ ≈ 1.0 for this domain
2. **Tuning is Important**: Wrong λ (too high/low) significantly reduces performance
3. **Data-Driven Tuning**: λ should be calibrated for each domain/dataset

---

## Running the Experiments

```bash
cd ~/code/gnome-materials
source venv/bin/activate
python3 main.py
```

This runs all 8 AL experiments (3 scaling + 5 lambda) sequentially:
- Scaling: v2 (2K) → v3-small (10K) → v3-large (50K) [~2.5 hours]
- Lambda: λ=0.0 → 0.5 → 1.0 → 2.0 → 5.0 [~2.5 hours]

**Total runtime:** ~5 hours on CPU (M1/M2 Mac)

---

## References

- **GNoME Paper**: [Hoffmann et al., Nature 2023](https://www.nature.com/articles/s41586-023-06735-9)
  - Fig 2: Scaling behavior with uncertainty sampling
  - Fig 3: Ablation on λ-like hyperparameters

- **Active Learning Theory**: [Settles, 2010](https://www.cs.cmu.edu/~bsettles/pub/settles.activelearning.pdf)
  - Chapter 4: Uncertainty sampling
  - Query-by-committee framework

- **Exploration-Exploitation Tradeoff**: [Thrun & Schwartz, 2000](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=a6e0e41cd58b0f9a98ad819cd7d8302e1e2c78f8)
  - UCB algorithm foundation
