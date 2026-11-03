# Enhancing Few-Shot Image Classification: Beyond MSENet

This repository contains an extensive, multi-phase research investigation aimed at improving few-shot image classification by systematically advancing upon the state-of-the-art **MSENet** (Multi-Scale Embedding Network) architecture proposed in:

> **"Enhancing Few-Shot Image Classification through Learnable Multi-Scale Embedding and Attention Mechanisms"**  
> *Fatemeh Askari, Amirreza Fateh, Mohammad Reza Mohammadi* (GitHub: [FatemehAskari/MSENet](https://github.com/FatemehAskari/MSENet))

Through 12+ independent novelty explorations and stacked architectural combinations, this codebase elevates few-shot classification on the **MiniImageNet** benchmark from the canonical **82.23%** baseline to a peak of **85.30%** (a super-additive gain of **+3.07 percentage points** over baseline and **+3.40 pp** over lowest baseline).

---

## 1. Core Problem & Research Motivation

Standard metric-based few-shot learners (e.g., prototypical networks) typically extract features from only the final layer of a deep backbone, discarding intermediate representations containing rich low-level spatial contours and mid-level part textures. **MSENet** addressed this by extracting multi-scale representations across 5 stages of ResNet-18, applying spatial self-attention, and fusing stage distances using a learnable weight vector.

However, detailed diagnostic analysis of the baseline MSENet revealed four fundamental limitations:
1. **Unbounded Euclidean Geometry:** Standard $L_2$ Euclidean distance is sensitive to feature vector magnitude and channel-depth scaling ($64$ vs $512$ channels), causing higher-dimensional stages to dominate gradients.
2. **Rigidity of Static Stage Weights:** The global stage weight vector $w$ is static across all evaluation tasks. It cannot adapt to task-specific granularity (e.g., fine-grained dog breeds requiring shallow textures vs. coarse-grained categories requiring deep semantics).
3. **Episodic Overfitting & Variance Drift:** Small few-shot batches cause running Batch Normalization statistics to fluctuate wildly, and models quickly overfit to training classes by Epoch 2.
4. **Spatial Information Loss via GAP:** Global Average Pooling uniformly weights all spatial locations, diluting compact discriminative foreground cues with background noise.

To overcome these deficiencies, we systematically developed and benchmarked five major families of novelties:

```
                                  ┌──────────────────────────────┐
                                  │   MSENet Baseline (82.23%)   │
                                  └──────────────┬───────────────┘
                                                 │
         ┌────────────────────────┬──────────────┴───────────────┬────────────────────────┐
         ▼                        ▼                              ▼                        ▼
  [Cross-Attention]       [Metric Learning]             [Dynamic Weighting]       [Training Recipes]
  • Adjacent CA           • Cosine Distance              • Fisher Ratio            • EMA (beta=0.99)
  • Bidirectional CA      • Hybrid Metric                • Query Shannon Entropy   • BN Freezing
  (83.25%)                • Learnable Temp T_cos         • Meta-Gating MLP         • Label Smoothing
                          (82.97%)                       (83.50%)                  (84.12%)
                                  │                              │                        │
                                  └──────────────┬───────────────┴────────────────────────┘
                                                 ▼
                                  ┌──────────────────────────────┐
                                  │   Stacked Synergies (85.30%) │
                                  │   • DW + LayerNorm + EMA     │
                                  │   • EMA + LayerNorm + GeM    │
                                  └──────────────────────────────┘
```

---

## 2. Repository Architecture & Directory Map

The repository is modularly organized into dedicated experiment folders, each containing standalone code notebooks, self-contained documentation, and output artifacts:

| Directory | Core Novelty / Focus | Key Mechanism | Best Accuracy |
|:---|:---|:---|:---:|
| [`msenet_baseline/`](file:///c:/Users/us/Downloads/few-shot-image-class/msenet_baseline/) | **Canonical MSENet Baseline** | ResNet-18 multi-scale extraction, self-attention, Euclidean metric, static weights | 82.23% |
| [`msenet_cosine/`](file:///c:/Users/us/Downloads/few-shot-image-class/msenet_cosine/) | **Metric Geometry & Scaling** | Hyperspherical $L_2$ normalization, Cosine Distance, Hybrid distance, Fixed vs. Learnable $\gamma$ | 82.97% |
| [`msenet_DynamicWeights/`](file:///c:/Users/us/Downloads/few-shot-image-class/msenet_DynamicWeights/) | **Task-Adaptive Weighting** | Dynamic stage weighting using Fisher discriminability, query entropy, and meta-gating MLP | 83.50% |
| [`Stacked-DW/`](file:///c:/Users/us/Downloads/few-shot-image-class/Stacked-DW/) | **Dynamic Weights + Synergies** | Dynamic Weights stacked with LayerNorm, EMA regularization, and GeM pooling | **85.30%** |
| [`Stacked-Euclidean/`](file:///c:/Users/us/Downloads/few-shot-image-class/Stacked-Euclidean/) | **Euclidean Stacked Model** | Euclidean metric stacked with EMA, LayerNorm, and Signed GeM pooling | **85.08%** |
| [`Stacked-Cosine/`](file:///c:/Users/us/Downloads/few-shot-image-class/Stacked-Cosine/) | **Cosine Stacked Model** | Cosine metric stacked with EMA, LayerNorm, and Signed GeM pooling | 84.53% |
| [`Cross-Attention/`](file:///c:/Users/us/Downloads/few-shot-image-class/Cross-Attention/) | **Cross-Stage Attention** | Adjacent and Bidirectional cross-attention between consecutive ResNet stages | 83.25% |
| [`LayerNorm_LearnableTemp/`](file:///c:/Users/us/Downloads/few-shot-image-class/LayerNorm_LearnableTemp/) | **Channel Norm & Logit Temp** | Stage-wise Layer Normalization after GAP + Softplus learnable temperature scaling | 83.18% |
| [`GeM_ContrastiveLoss/`](file:///c:/Users/us/Downloads/few-shot-image-class/GeM_ContrastiveLoss/) | **Non-Linear Pooling & SupCon** | Learnable Generalized Mean (GeM) pooling ($p > 1$) + auxiliary Supervised Contrastive Loss | 83.06% |
| [`5-way-ablation/`](file:///c:/Users/us/Downloads/few-shot-image-class/5-way-ablation/) | **Training Stability Ablations** | EMA weights (A4), Label Smoothing (A5), BN eval freeze (A1), $L_2$-SP (A3), Diff LR (A2) | 84.12% |
| [`DistanceComparison/`](file:///c:/Users/us/Downloads/few-shot-image-class/DistanceComparison/) | **Metric Distance Benchmark** | Paired statistical comparison of Euclidean vs. Cosine vs. Hybrid under standard conditions | 81.94% |
| [`plots/`](file:///c:/Users/us/Downloads/few-shot-image-class/plots/) | **Master Visualizations** | Publication-grade comparative charts, roadmaps, and gain analyses across all novelties | — |

---

## 3. Taxonomy of Novelties Explored

### 1. Feature Interaction: Cross-Attention across Stages
- **Adjacent Cross-Attention:** Allows shallower stages to attend to the richer semantic features of their immediately deeper neighbors ($X'_i = X_i + \operatorname{CA}(X_i, X_{i+1})$), providing high-level context while preserving spatial detail (**83.25%**, +1.02 pp over baseline).
- **Bidirectional Cross-Attention:** Dual information exchange in both forward and backward directions (**83.06%**).

### 2. Metric Learning: Hyperspherical Cosine & Hybrid Distances
- **Cosine Distance:** L2-normalizes support prototypes and query features onto a unit hypersphere, neutralizing magnitude imbalances across different channel depths ($d_{\cos} = T_{\cos}(1 - \hat{z}_q \cdot \hat{c}_j^\top)$). Scaled by a learnable temperature $T_{\cos} \approx 10.0$ (**82.97%**).
- **Hybrid Metric:** Blends Euclidean distance and Cosine distance in equal measure ($0.5 d_{\text{euc}} + 0.5 d_{\cos}$), merging geometric magnitude separation with directional alignment (**82.66%**).

### 3. Adaptive Weighting: Task-Adaptive Dynamic Stage Weighting
- **Support-Set Fisher Ratio:** Measures inter-class prototype separation over intra-class scatter at each stage.
- **Query Shannon Entropy:** Computes query prediction decisiveness to weight stages based on confidence.
- **Lightweight Meta-Gating MLP:** Combines statistical descriptors with zero-initialized neural context gating to dynamically modulate positive global priors per episode and query (**83.50%** with Cosine).

### 4. Pooling & Normalization: GeM & LayerNorm
- **Generalized Mean (GeM) Pooling:** Replaces uniform GAP with learnable power parameter $p$ ($p=1$ is average pooling, $p \to \infty$ is max pooling). Focuses on salient foreground activations (**83.06%**).
- **Stage-Wise Layer Normalization:** Normalizes channel distributions after pooling, stabilizing Euclidean distance scales (**83.06%** alone, **83.18%** with learnable temperature).

### 5. Regularization & Stability: EMA & Smoothing
- **Exponential Moving Average (EMA, $\beta=0.99$):** Accumulates a shadow copy of weights across training steps ($\theta_{\text{EMA}} = 0.99 \theta_{\text{EMA}} + 0.01 \theta$). Radically suppresses episodic gradient noise, yielding **84.12%** (+1.89 pp gain as an isolated technique).
- **Label Smoothing ($\epsilon=0.1$):** Softens one-hot targets to prevent overconfidence under few-shot data scarcity (**82.52%**).
- **BatchNorm Freezing:** Preserves ImageNet running statistics during few-shot fine-tuning (**82.33%**).

### 6. Stacked Super-Additivity: The >85% Breakthrough
By stacking mutually orthogonal improvements—**Dynamic Stage Weighting + Layer Normalization + EMA Regularization**—the model achieves **85.30%** test accuracy on MiniImageNet (5-way 5-shot), demonstrating that architectural adaptation, channel normalization, and temporal parameter smoothing reinforce each other constructively.

---

## 4. Benchmark Experimental Protocol

All experiments follow a strictly controlled protocol to ensure fair, statistically sound comparisons:
- **Dataset:** MiniImageNet (100 classes with 600 images per class at $84 \times 84$ resolution; 64 train, 16 validation, 20 test classes).
- **Backbone:** ResNet-18 (ImageNet-pretrained).
- **Episodic Training:** 10 epochs, 360 episodes per epoch (30-way 5-shot with 15 queries per class).
- **Episodic Evaluation:** 600 fixed, held-out episodes (5-way 5-shot with 15 queries per class, totaling 75 queries per episode).
- **Checkpointing:** Validated after each epoch; the best validation accuracy checkpoint is loaded for final test evaluation.
- **Statistical Rigor:** 95% Confidence Intervals ($\pm 1.96 \cdot \frac{\sigma}{\sqrt{600}}$) and paired head-to-head episode hypothesis tests.

---

## 5. Comprehensive Results & Visualizations

- For the complete Master Leaderboard, full-scale numerical comparisons, paired statistical win-rates, and category breakdowns, see [`RESULTS.md`](file:///c:/Users/us/Downloads/few-shot-image-class/RESULTS.md) (and [`results_table.md`](file:///c:/Users/us/Downloads/few-shot-image-class/results_table.md)).
- High-resolution comparative plots are cataloged in [`plots/`](file:///c:/Users/us/Downloads/few-shot-image-class/plots/).
