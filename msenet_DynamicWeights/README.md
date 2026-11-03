# Task-Adaptive Dynamic Stage Weighting for MSENet

## 1. Motivation: Beyond Static Stage Weights

In the original MSENet paper:
> **"Enhancing Few-Shot Image Classification through Learnable Multi-Scale Embedding and Attention Mechanisms"**  
> *Fatemeh Askari, Amirreza Fateh, Mohammad Reza Mohammadi* (GitHub: [FatemehAskari/MSENet](https://github.com/FatemehAskari/MSENet))

The multi-scale architecture extracts features from five ResNet-18 stages (`maxpool`, `layer1`, `layer2`, `layer3`, `layer4`) and combines their stage distances using a **static vector of learnable stage weights** $w = [w_0, w_1, w_2, w_3, w_4]$:

$$\text{Score}(x_i^q, j) = -\sum_{p=0}^4 w_p \cdot d_{i,j}^p$$

### The Limitation of Static Weights
In realistic few-shot classification, **tasks vary dramatically in visual granularity and difficulty**:
- **Fine-grained tasks** (e.g., distinguishing between dog breeds or bird species) require shallow-to-intermediate spatial representations (`layer1`, `layer2`) that capture localized textures, fur patterns, and low-level edge structures.
- **Coarse-grained tasks** (e.g., distinguishing vehicles from animals) rely almost exclusively on high-level semantic abstractions (`layer3`, `layer4`).
- **Query ambiguity:** Some query instances lie close to decision boundaries and benefit from different multi-scale feature combinations than highly prototypical queries.

A single static weight vector forces the model into a compromise across all tasks, preventing it from dynamically emphasizing the most discriminative stages for a specific support set and query instance.

---

## 2. The Novelty: Task-Adaptive Dynamic Stage Weighting

This folder introduces **Task-Adaptive Dynamic Stage Weighting**, an adaptive gating framework that calculates stage weights **on-the-fly for every individual few-shot episode and query sample**.

```
                      Support Images                           Query Images
                            │                                       │
                    ┌───────┴───────┐                       ┌───────┴───────┐
                    ▼               ▼                       ▼               ▼
             [Stage 0 (maxpool)]  [Stage 4 (layer4)]  [Stage 0 (maxpool)]  [Stage 4 (layer4)]
                    │               │                       │               │
             Self-Attention   Self-Attention          Self-Attention   Self-Attention
                    │               │                       │               │
                  [GAP]           [GAP]                   [GAP]           [GAP]
                    │               │                       │               │
             Prototypes c^0   Prototypes c^4          Query z_q^0     Query z_q^4
                    │               │                       │               │
                    └───────┬───────┘                       └───────┬───────┘
                            │                                       │
                            ├───────────────────┬───────────────────┤
                            ▼                   ▼                   ▼
                     Support Fisher       Query Entropy       Stage Distances
                     Ratio (F_s)          Confidence (C_s)    (d_s)
                            │                   │                   │
                            └───────────────────┼───────────────────┘
                                                ▼
                                    TaskAdaptiveWeightGenerator
                                   (Statistical Modulator + MLP)
                                                ▼
                                   Dynamic Stage Weights w_dyn(s)
                                                ▼
                                   Final Score = - Σ w_dyn(s) * d_s
```

### Mathematical Formulation

The dynamic weighting module combines three complementary sources of task context:

#### 1. Support-Set Fisher Discriminability Ratio ($F_s$)
For each stage $s \in \{0, \dots, 4\}$, we measure how well the support set separates classes relative to its internal cluster spread:
$$F_s = \frac{\text{Inter-Class Prototype Separation}}{\text{Intra-Class Scatter} + \epsilon} = \frac{\frac{1}{N(N-1)} \sum_{j \neq k} \left(1 - \hat{c}_j^s \cdot \hat{c}_k^s\right)}{\frac{1}{|S|} \sum_{i \in S} \left(1 - \hat{z}_i^s \cdot \hat{c}_{y_i}^s\right) + 0.1}$$

Where $\hat{z}$ and $\hat{c}$ are $L_2$-normalized support features and prototypes. Normalized across stages via temperature-scaled Softmax:
$$w_{\text{fisher}}^s = \operatorname{Softmax}\left(\tau_{\text{fisher}} \cdot \tilde{F}_s\right) \times S$$

#### 2. Query-to-Prototype Confidence Margin ($C_{s,q}$)
Measures how decisive stage $s$ is for query $q$ by evaluating the normalized negative Shannon entropy of its predicted class distribution:
$$P(y=j \mid x_q, s) = \operatorname{Softmax}\left(-d^s(x_q, c_j)\right)$$
$$H_s(q) = -\frac{1}{\ln N} \sum_{j=1}^N P(y=j \mid x_q, s) \ln P(y=j \mid x_q, s)$$
$$C_{s,q} = 1.0 - H_s(q) \in [0, 1]$$
Normalized across stages:
$$w_{\text{conf}}^{s,q} = \operatorname{Softmax}\left(\tau_{\text{fisher}} \cdot \tilde{C}_{s,q}\right) \times S$$

#### 3. Lightweight Meta-Gating Network ($\delta_{\text{neural}}$)
A 2-layer neural network with LayerNorm and GELU non-linearities ingests a $3 \times S$ (15-dimensional) task descriptor vector:
$$\mathbf{v}_{\text{task}} = \left[\tilde{\mathbf{F}}, \bar{\mathbf{C}}, \tilde{\mathbf{D}}\right] \in \mathbb{R}^{15}$$
$$\delta_{\text{neural}} = \tanh\left(\mathbf{W}_2 \cdot \operatorname{GELU}\left(\operatorname{LN}\left(\mathbf{W}_1 \mathbf{v}_{\text{task}} + \mathbf{b}_1\right)\right) + \mathbf{b}_2\right) \in [-1, 1]^S$$

> **Zero-Initialization Trick:** $\mathbf{W}_2$ and $\mathbf{b}_2$ are strictly initialized to zero. At the beginning of training, $\delta_{\text{neural}} = \mathbf{0}$, meaning dynamic modulation starts cleanly at the global prior without destabilizing early backbone gradients.

#### 4. Fused Dynamic Stage Weights ($w_{\text{dyn}}$)
Analytical statistics ($w_{\text{fisher}}, w_{\text{conf}}$) and the neural gating delta ($\delta_{\text{neural}}$) are modulated and anchored around the learnable global prior $w_{\text{prior}} = \operatorname{Softplus}(w_{\text{base}})$:

$$\alpha = \sigma(\text{mix\_logit})$$
$$\text{stat\_multiplier} = \alpha \cdot w_{\text{fisher}} + (1 - \alpha) \cdot w_{\text{conf}}$$
$$w_{\text{dyn}}^{s,q} = w_{\text{prior}}^s \times \left[(1 - g) + g \cdot \text{stat\_multiplier}^{s,q}\right] \times \left(1 + g \cdot \delta_{\text{neural}}^s\right)$$

Where $g = 0.5$ is the gating scale bounding the modulation range.

---

## 3. Systematic Prototype Distance Comparison

We evaluate Task-Adaptive Dynamic Stage Weighting across **all three prototypical distance functions**:

1. **Euclidean Distance ($d_{\text{euc}}$):** Standard $L_2$ norm between query feature $z_q$ and class prototype $c_j$:
   $$d_{\text{euc}}(z_q, c_j) = \|z_q - c_j\|_2$$
2. **Cosine Distance ($d_{\cos}$):** Hyperspherical angular distance with learnable temperature scaling $T_{\cos}$:
   $$d_{\cos}(z_q, c_j) = T_{\cos} \cdot \left(1 - \frac{z_q}{\|z_q\|_2} \cdot \frac{c_j^\top}{\|c_j\|_2}\right)$$
3. **Hybrid Distance ($d_{\text{hybrid}}$):** Equal-weight blend combining geometric separation and directional alignment:
   $$d_{\text{hybrid}}(z_q, c_j) = 0.5 \cdot d_{\text{euc}}(z_q, c_j) + 0.5 \cdot d_{\cos}(z_q, c_j)$$

All three variants are trained independently from identical backbone initializations using fixed attention scaling ($\gamma_1 = \gamma_2 = 0.2$).

---

## 4. Folder Structure & Contents

```
msenet_DynamicWeights/
├── msenet_dynamicweights_withDistance_comparison.ipynb   # Main 3-way benchmark notebook
├── msenet-dynamicweights.ipynb                           # Standalone single-metric notebook
├── README.md                                             # This architecture & methodology guide
├── RESULTS.md                                            # Comprehensive results, tables, and analysis
└── results/
    ├── dynamic_distance_summary.csv                      # Main performance and resource summary
    ├── dynamic_distance_paper_table.csv                  # Paper-ready formatted summary table
    ├── dynamic_distance_paired_comparisons.csv           # Episode-level paired win/tie/loss stats
    ├── dynamic_stage_weight_summary.csv                  # Stage weight mean and std per metric
    └── plots/
        ├── dynamic_weighting_acc_comparison.png          # Accuracy comparison with 95% CIs
        ├── episode_wise_acc.png                          # Episode-level accuracy distribution
        ├── inference_time_per_episode.png                # Per-episode latency comparison
        ├── mean_dyn_stage_weights_by_proto_distance.png  # Stage weight allocations per metric
        ├── paired_episode_level_acc_differences.png      # Paired episode difference histograms
        └── peak_gpu_memory_comp.png                      # Peak GPU VRAM footprint
```

---

## 5. Summary of Key Results

| Prototype Distance Metric | Test Accuracy (%) | 95% CI (%) | Best Val Acc (%) | Test Loss | Inf Time / ep (s) | Peak GPU VRAM (GB) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Cosine + Dynamic Weights** | **83.50%** | **[83.00%, 84.00%]** | **83.86%** | **0.4519** | 0.0896s | 0.368 GB |
| **Hybrid + Dynamic Weights** | **83.13%** | **[82.60%, 83.65%]** | **83.70%** | 0.4624 | 0.0938s | 0.458 GB |
| **Euclidean + Dynamic Weights** | **82.57%** | **[82.05%, 83.09%]** | **83.38%** | 0.4754 | 0.0893s | 0.280 GB |

- **Top Performer:** Cosine Distance + Dynamic Stage Weighting achieves **83.50%** test accuracy on MiniImageNet (5-way 5-shot), outperforming original MSENet baseline (~82.24%) by **+1.26 pp** and static Cosine MSENet (82.97%) by **+0.53 pp**.
- **Paired Advantage:** In head-to-head episode comparison, Cosine beats Euclidean in **52.5%** of test episodes (+0.93 pp mean margin).
- **Negligible Overhead:** Dynamic weighting adds only 4,772 parameters (+0.04% parameter overhead) and runs in **<90 ms per episode**.

For detailed per-epoch progression, paired win-rates, stage weight distributions, and deep-dive discussions, see [`RESULTS.md`](RESULTS.md).
