# Master Results Table: MSENet Improvement Benchmark

This file provides a concise, copy-ready reference of all test accuracies and performance deltas across the entire repository. For complete visual plots, deep-dive discussions, and paired statistical test metrics, please refer to [`RESULTS.md`](RESULTS.md).

---

## Master Benchmark Table

- **Dataset:** MiniImageNet (84x84, 5-way 5-shot evaluation over 600 fixed test episodes)
- **Backbone:** ResNet-18 (ImageNet-pretrained)
- **Lowest Observed Baseline:** **81.90%** (81.8956%)
- **Canonical MSENet Baseline:** **82.23%** (82.2356%)

| Rank | Model / Novelty Configuration | Subfolder / Source | Distance Metric | Test Acc (5w-5s) | 95% Conf. Interval | $\Delta$ vs. Lowest Base (81.90%) | $\Delta$ vs. Canon Base (82.23%) | Best Val Acc (%) |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 | **Stacked DW: DW + LayerNorm + EMA** | [`Stacked-DW/`](Stacked-DW) | Euclidean | **85.3000%** | **[84.82%, 85.78%]** | **+3.40 pp** | **+3.07 pp** | 84.36% |
| 🥈 | **Stacked DW: DW + LN + EMA + GeM** | [`Stacked-DW/`](Stacked-DW) | Euclidean | **85.2689%** | **[84.77%, 85.76%]** | **+3.37 pp** | **+3.04 pp** | **84.45%** |
| 🥉 | **Stacked Euclidean: EMA + LN + GeM** | [`Stacked-Euclidean/`](Stacked-Euclidean) | Euclidean | **85.0778%** | **[84.59%, 85.56%]** | **+3.18 pp** | **+2.85 pp** | **84.76%** |
| 4 | **Stacked Euclidean: EMA + LayerNorm** | [`Stacked-Euclidean/`](Stacked-Euclidean) | Euclidean | **84.8133%** | [84.32%, 85.31%] | **+2.91 pp** | **+2.58 pp** | 84.32% |
| 5 | **Stacked Euclidean: EMA + GeM** | [`Stacked-Euclidean/`](Stacked-Euclidean) | Euclidean | **84.7844%** | [84.29%, 85.28%] | **+2.88 pp** | **+2.55 pp** | 84.51% |
| 6 | **Stacked Cosine: EMA + GeM** | [`Stacked-Cosine/`](Stacked-Cosine) | Cosine | **84.5311%** | [84.03%, 85.04%] | **+2.63 pp** | **+2.30 pp** | 84.19% |
| 7 | **Stacked Cosine: EMA + LayerNorm** | [`Stacked-Cosine/`](Stacked-Cosine) | Cosine | **84.5311%** | [84.04%, 85.02%] | **+2.63 pp** | **+2.30 pp** | 84.42% |
| 8 | **Stacked Cosine: EMA + LN + GeM** | [`Stacked-Cosine/`](Stacked-Cosine) | Cosine | **84.4311%** | [83.94%, 84.93%] | **+2.53 pp** | **+2.20 pp** | 83.98% |
| 9 | **EMA Regularization only (A4)** | [`5-way-ablation/`](5-way-ablation) | Euclidean | **84.1244%** | [83.63%, 84.62%] | **+2.22 pp** | **+1.89 pp** | 83.75% |
| 10 | **Task-Adaptive Dynamic Weights** | [`msenet_DynamicWeights/`](msenet_DynamicWeights) | Cosine | **83.50%** | [83.00%, 84.00%] | **+1.60 pp** | **+1.27 pp** | 83.86% |
| 11 | **Adjacent Cross-Attention** | [`Cross-Attention/`](Cross-Attention) | Euclidean | **83.25%** | [82.73%, 83.77%] | **+1.35 pp** | **+1.02 pp** | 82.26% |
| 12 | **LayerNorm + Learnable Temp** | [`LayerNorm_LearnableTemp/`](LayerNorm_LearnableTemp) | Euclidean | **83.18%** | [82.66%, 83.69%] | **+1.28 pp** | **+0.95 pp** | 82.52% |
| 13 | **Dynamic Weights - Hybrid** | [`msenet_DynamicWeights/`](msenet_DynamicWeights) | Hybrid | **83.13%** | [82.60%, 83.65%] | **+1.23 pp** | **+0.90 pp** | 83.70% |
| 14 | **GeM Pooling only** | [`GeM_ContrastiveLoss/`](GeM_ContrastiveLoss) | Euclidean | **83.06%** | [82.53%, 83.58%] | **+1.16 pp** | **+0.83 pp** | 82.96% |
| 15 | **LayerNorm only** | [`LayerNorm_LearnableTemp/`](LayerNorm_LearnableTemp) | Euclidean | **83.06%** | [82.54%, 83.58%] | **+1.16 pp** | **+0.83 pp** | 82.66% |
| 16 | **Bidirectional Cross-Attention** | [`Cross-Attention/`](Cross-Attention) | Euclidean | **83.06%** | [82.55%, 83.57%] | **+1.16 pp** | **+0.83 pp** | 82.24% |
| 17 | **Cosine Distance (Fixed $\gamma$)** | [`msenet_cosine/`](msenet_cosine) | Cosine | **82.97%** | [82.45%, 83.49%] | **+1.07 pp** | **+0.74 pp** | 83.07% |
| 18 | **Dynamic Weights (DW only, D0)** | [`Stacked-DW/`](Stacked-DW) | Euclidean | **82.88%** | [82.36%, 83.39%] | **+0.98 pp** | **+0.65 pp** | 82.72% |
| 19 | **GeM + Contrastive Loss** | [`GeM_ContrastiveLoss/`](GeM_ContrastiveLoss) | Euclidean | **82.68%** | [82.15%, 83.21%] | **+0.78 pp** | **+0.45 pp** | 82.51% |
| 20 | **Hybrid Distance (Fixed $\gamma$)** | [`msenet_cosine/`](msenet_cosine) | Hybrid | **82.66%** | [82.14%, 83.18%] | **+0.76 pp** | **+0.43 pp** | 82.42% |
| 21 | **Dynamic Weights - Euclidean** | [`msenet_DynamicWeights/`](msenet_DynamicWeights) | Euclidean | **82.57%** | [82.05%, 83.09%] | **+0.67 pp** | **+0.34 pp** | 83.38% |
| 22 | **Label Smoothing (A5)** | [`5-way-ablation/`](5-way-ablation) | Euclidean | **82.52%** | [81.98%, 83.05%] | **+0.62 pp** | **+0.29 pp** | 82.38% |
| 23 | **Stacked Baseline (E0)** | [`Stacked-Euclidean/`](Stacked-Euclidean) | Euclidean | **82.52%** | [82.01%, 83.04%] | **+0.62 pp** | **+0.29 pp** | 82.96% |
| 24 | **BatchNorm Eval Freeze (A1)** | [`5-way-ablation/`](5-way-ablation) | Euclidean | **82.33%** | [81.79%, 82.87%] | **+0.43 pp** | **+0.10 pp** | 82.00% |
| 25 | **Contrastive Loss only** | [`GeM_ContrastiveLoss/`](GeM_ContrastiveLoss) | Euclidean | **82.26%** | [81.74%, 82.78%] | **+0.36 pp** | **+0.03 pp** | 82.22% |
| 26 | **MSENet Baseline (Canonical)** | [`msenet_baseline/`](msenet_baseline) | Euclidean | **82.23%** | [81.72%, 82.75%] | **+0.33 pp** | **0.00 pp** | 81.25% |
| 27 | **Cosine Distance (Learnable $\gamma$)** | [`msenet_cosine/`](msenet_cosine) | Cosine | **82.15%** | [81.63%, 82.67%] | **+0.25 pp** | -0.08 pp | 81.67% |
| 28 | **Stacked Cosine Baseline (C0)** | [`Stacked-Cosine/`](Stacked-Cosine) | Cosine | **82.07%** | [81.55%, 82.59%] | **+0.17 pp** | -0.16 pp | 82.53% |
| 29 | **Hybrid Distance (Learnable $\gamma$)** | [`msenet_cosine/`](msenet_cosine) | Hybrid | **81.98%** | [81.46%, 82.50%] | **+0.08 pp** | -0.25 pp | 82.08% |
| 30 | **MSENet Baseline (Lowest)** | [`DistanceComparison/`](DistanceComparison) | Euclidean | **81.90%** | [81.34%, 82.45%] | **0.00 pp** | -0.33 pp | 82.30% |
| 31 | **Discriminative LR (A2)** | [`5-way-ablation/`](5-way-ablation) | Euclidean | **78.89%** | [78.33%, 79.46%] | -3.01 pp | -3.34 pp | 78.59% |

---

## Visual Summary Plots

1. [Figure 1: Master Leaderboard Bar Chart](plots/01_master_leaderboard_accuracy.png)
2. [Figure 2: Distance Metric Comparison (Static vs. Dynamic Weights)](plots/02_distance_metrics_dynamic_vs_static.png)
3. [Figure 3: Evolutionary Trajectory Exceeding 85% Accuracy](plots/03_evolutionary_roadmap_over_85.png)
4. [Figure 4: Accuracy Gains (pp) Over Baseline](plots/04_gains_over_baseline.png)
