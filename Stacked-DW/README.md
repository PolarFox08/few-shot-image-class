# Stacked MSENet with Dynamic Stage Weighting: Component Ablations (EMA, LayerNorm, GeM)

## 1. Motivation: Beyond Baseline Multi-Scale Embeddings and Static Gating

In the original MSENet architecture:
> **"Enhancing Few-Shot Image Classification through Learnable Multi-Scale Embedding and Attention Mechanisms"**  
> *Fatemeh Askari, Amirreza Fateh, Mohammad Reza Mohammadi* (GitHub: [FatemehAskari/MSENet](https://github.com/FatemehAskari/MSENet))

The multi-scale architecture extracts feature representations across five intermediate stages of a ResNet-18 backbone (`maxpool`, `layer1`, `layer2`, `layer3`, and `layer4`), enriches them with stage-wise spatial self-attention, pools them with Global Average Pooling (GAP), and computes class prototypes by averaging support embeddings. Query classifications are derived from weighted distances across all five stages using learnable stage weights $w = [w_1, w_2, w_3, w_4, w_5]$:

$$\text{Score}(x_i^q, j) = -\sum_{s=1}^5 w_s \cdot d_{\text{euc}}^s(z_q^s, c_j^s)$$

### Limitations of the Original MSENet Pipeline
While the original multi-scale embedding mechanism effectively incorporates coarse-to-fine visual information, several architectural and optimization bottlenecks limit its discriminative power under standard Euclidean distance:
1. **Feature Scale Discrepancies Across Stages:** Intermediate stages of ResNet-18 exhibit vastly different channel dimensions (64 to 512) and unbounded activation ranges. Without normalization post-pooling, Euclidean distances are easily dominated by unscaled channel variances rather than semantic dissimilarities.
2. **Loss of Prominent Spatial Activations in GAP:** Standard Global Average Pooling treats every spatial location uniformly ($H_s \times W_s$). In few-shot classification, salient object parts (e.g., discriminative textures, bird beaks, animal eyes) become washed out by background averaging prior to prototype construction.
3. **Episodic Optimization Instability & Meta-Gradient Variance:** Meta-learning few-shot models with episodic mini-batches (e.g., 360 episodes per epoch at 30-way 5-shot) introduces severe stochastic gradient variance. In deep feature extractors, noisy parameter updates cause representation collapse and erratic validation trajectory oscillations.
4. **Compromise of Static Stage Weights:** In realistic few-shot classification, tasks vary dramatically in visual granularity and difficulty. A static global weight vector cannot dynamically modulate the importance of low-level vs. high-level features for a specific support set and query instance.

---

## 2. The Novelty: Stacked Architectural Enhancements with Dynamic Stage Weighting

To overcome these structural limitations, this experiment stacks three complementary architectural and optimization enhancements—**Layer Normalization (LayerNorm)**, **Exponential Moving Average (EMA)**, and **Signed Generalized Mean (GeM) Pooling**—directly on top of **Task-Adaptive Dynamic Stage Weighting** within the standard **Euclidean Distance** regime:

```
                          Support & Query Images
                                    │
                                    ▼
                         ResNet-18 Backbone Stages
                 [maxpool]   [layer1]   [layer2]   [layer3]   [layer4]
                     │           │          │          │          │
                     ▼           ▼          ▼          ▼          ▼
              Stage-Wise Self-Attention Modules (learnable gamma1, gamma2)
                     │           │          │          │          │
                     ▼           ▼          ▼          ▼          ▼
             ┌────────────────────────────────────────────────────────┐
             │       Signed Generalized Mean (GeM) Pooling            │
             │           p_s = Softplus(raw_s) + 0.01                 │
             │     (Or Standard Global Average Pooling for D0 & D1)   │
             └────────────────────────────────────────────────────────┘
                                    │
                                    ▼
             ┌────────────────────────────────────────────────────────┐
             │         Stage-Specific Layer Normalization             │
             │         y = LayerNorm(z_s) across channels             │
             │         (Enabled for D1 & D2; Disabled for D0)         │
             └────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    Class Prototype & Euclidean Distance
                    d_s(z_q, c_j) = ||z_q^s - c_j^s||_2
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         │                          │                          │
         ▼                          ▼                          ▼
  Support Fisher Ratio       Query Entropy             Stage Distances
       (F_s)                Confidence (C_s)                (d_s)
         │                          │                          │
         └──────────────────────────┼──────────────────────────┘
                                    ▼
                       TaskAdaptiveWeightGenerator
                      (Statistical Modulator + MLP)
                                    ▼
                      Dynamic Stage Weights w_dyn(s)
                                    ▼
                      Score(x_q, j) = - Σ w_dyn(s) * d_s
                                    │
                                    ▼
             ┌────────────────────────────────────────────────────────┐
             │     Exponential Moving Average (EMA) Evaluation        │
             │       θ_EMA^(t) = β * θ_EMA^(t-1) + (1-β) * θ^(t)      │
             │         (Enabled for D1 & D2; Disabled for D0)         │
             └────────────────────────────────────────────────────────┘
```

### Mathematical Formulation of Stacked Components

#### 1. Task-Adaptive Dynamic Stage Weighting Module
Dynamic weights are generated on-the-fly per task and query instance by combining three information channels:
- **Support-Set Fisher Discriminability Ratio ($F_s$):** Evaluates class separability relative to intra-cluster spread:
  $$F_s = \frac{\frac{1}{N(N-1)} \sum_{j \neq k} (1 - \hat{c}_j^s \cdot \hat{c}_k^s)}{\frac{1}{|S|} \sum_{i \in S} (1 - \hat{z}_i^s \cdot \hat{c}_{y_i}^s) + 0.1}$$
  Normalized across stages: $w_{\text{fisher}}^s = \operatorname{Softmax}(\tau_{\text{fisher}} \cdot \tilde{F}_s) \times S$, with learnable temperature $\tau_{\text{fisher}}$ (initialized to 2.0).
- **Query-to-Prototype Confidence Margin ($C_{s,q}$):** Evaluates negative Shannon entropy of class predictions:
  $$P(y=j \mid x_q, s) = \operatorname{Softmax}(-d^s(x_q, c_j))$$
  $$H_s(q) = -\frac{1}{\ln N} \sum_{j=1}^N P(y=j \mid x_q, s) \ln P(y=j \mid x_q, s)$$
  $$C_{s,q} = 1.0 - H_s(q), \quad w_{\text{conf}}^{s,q} = \operatorname{Softmax}(\tau_{\text{fisher}} \cdot \tilde{C}_{s,q}) \times S$$
- **Lightweight Meta-Gating MLP ($\delta_{\text{neural}}$):** Ingests the 15-dimensional task descriptor $[\tilde{\mathbf{F}}, \bar{\mathbf{C}}, \tilde{\mathbf{D}}] \in \mathbb{R}^{15}$ and outputs modulation deltas $\delta_{\text{neural}} \in [-1, 1]^S$ with zero-initialized final layer weights $\mathbf{W}_2$ and biases $\mathbf{b}_2$.
- **Dynamic Weight Fusion:** Blended with learnable gating scale $g = 0.5$ and global softplus prior $w_{\text{prior}} = \operatorname{Softplus}(w_{\text{base}})$:
  $$\alpha = \sigma(\text{mix\_logit})$$
  $$\text{stat\_multiplier} = \alpha \cdot w_{\text{fisher}} + (1 - \alpha) \cdot w_{\text{conf}}$$
  $$w_{\text{dyn}}^{s,q} = w_{\text{prior}}^s \times \left[(1 - g) + g \cdot \text{stat\_multiplier}^{s,q}\right] \times \left(1 + g \cdot \delta_{\text{neural}}^s\right)$$

#### 2. Stage-Specific Layer Normalization (LayerNorm)
LayerNorm is applied immediately following spatial pooling across all five ResNet-18 feature stages:

$$\hat{z}_s = \frac{z_s - \mu_s}{\sqrt{\sigma_s^2 + \epsilon}}, \quad y_s = \gamma_s \odot \hat{z}_s + \beta_s$$

Where:
- $\mu_s$ and $\sigma_s^2$ denote channel-wise mean and variance across channel dimension $C_s \in \{64, 64, 128, 256, 512\}$.
- $\gamma_s, \beta_s \in \mathbb{R}^{C_s}$ represent learnable affine scale and shift parameters.
- **Impact:** LayerNorm prevents deeper stages with higher unscaled activation magnitudes from drowning out shallower representations, ensuring Euclidean distances correspond to balanced semantic divergences.

#### 3. Exponential Moving Average (EMA)
To stabilize parameters against noisy episodic meta-gradients, an exponential moving average is maintained after each optimizer update step:

$$\theta_{\text{EMA}}^{(t)} = \beta \cdot \theta_{\text{EMA}}^{(t-1)} + (1 - \beta) \cdot \theta^{(t)}$$

Where:
- Decay rate $\beta = 0.99$.
- Model evaluation (both validation and final testing) evaluates the smoothed parameter checkpoint $\theta_{\text{EMA}}$.

#### 4. Signed Generalized Mean (GeM) Pooling
Signed Generalized Mean Pooling replaces standard Global Average Pooling (GAP) in the post-attention stage to retain localized salient features while handling potential negative residual activations:

$$x = x^+ - x^- \quad \text{where } x^+ = \max(x, 0), \; x^- = \max(-x, 0)$$
$$\operatorname{GeM}(x, p_s) = \operatorname{GeM}_{p_s}(x^+) - \operatorname{GeM}_{p_s}(x^-)$$
$$\operatorname{GeM}_{p_s}(t) = \exp\left(\frac{1}{p_s}\left(\operatorname{logsumexp}(p_s \cdot \ln(\operatorname{clamp}(t, \epsilon))) - \ln(|\Omega|)\right)\right)$$

Where:
- $|\Omega| = H_s \times W_s$ denotes spatial resolution.
- Power parameter $p_s$ is strictly positive and learnable: $p_s = \operatorname{Softplus}(\text{gem\_raw}_s) + p_{\text{floor}}$ with $p_{\text{floor}} = 0.01$ and $p_{\text{init}} = 3.0$.

---

## 3. Evaluated Configurations

The study systematically ablates and stacks components on top of the Dynamic Stage Weighting framework under standard Euclidean distance:

| Configuration ID | Description | Dynamic Weights | LayerNorm | EMA ($\beta = 0.99$) | Signed GeM Pooling | Learnable Attention ($\gamma_1, \gamma_2$) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **D0** | DW only (Baseline) | Yes | No | No | No (GAP) | Yes |
| **D1** | DW + LayerNorm + EMA | Yes | Yes | Yes | No (GAP) | Yes |
| **D2** | DW + LayerNorm + EMA + GeM | Yes | Yes | Yes | Yes | Yes |

All three models share identical initializations, learning rates, seeds, and episodic data schedules.

---

## 4. Implementation Details and Hyperparameters

- **Backbone Architecture:** ResNet-18 pretrained on ImageNet (`use_imagenet_pretrained = True`).
- **Input Resolution:** $84 \times 84 \times 3$.
- **Distance Metric:** Euclidean distance ($L_2$ norm), fixed across all stages.
- **Stage Weight Initialization:** $w_0 = [1.0, 1.1, 1.2, 1.3, 1.4]$ with initial softplus transformation.
- **Spatial Self-Attention Scaling:** Learnable $\gamma_1, \gamma_2$ initialized at $0.20$.
- **Dynamic Weight Generator:**
  - Support Fisher Ratio: $\tau_{\text{fisher}}$ init $= 2.0$.
  - Neural Gating Scale $g = 0.5$.
  - Neural Meta-Gating MLP: Linear(15, 32) $\to$ LayerNorm(32) $\to$ GELU $\to$ Linear(32, 5) with zero weights and bias initialization on final layer.
  - Learning rate `dynamic_gate_lr = 0.001`.
- **Optimization Strategy:**
  - Optimizer: Adam ($\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$).
  - Backbone learning rate: $\eta_{\text{backbone}} = 10^{-4}$.
  - Stage weights learning rate: $\eta_{\text{weights}} = 5 \times 10^{-3}$.
  - GeM learning rate: $\eta_{\text{gem}} = 10^{-3}$.
  - Learning rate scheduler: Cosine Annealing over 10 epochs.
  - Automatic Mixed Precision (AMP): Enabled.
- **Episodic Setup:**
  - Training schedule: 10 epochs $\times$ 360 episodes/epoch = 3,600 training tasks.
  - Training task shape: 30-way 5-shot (5 support, 15 query images per class).
  - Validation schedule: 600 fixed episodes/epoch at 5-way 5-shot (5 support, 15 query images per class).
  - Test schedule: 600 fixed test episodes at 5-way 5-shot.
- **Fixed Seeds for Reproducibility:**
  - Global seed: `42`
  - Training episode seed: `20263`
  - Validation episode seed: `20261`
  - Test episode seed: `20262`

---

## 5. Folder Structure & Contents

```
Stacked-DW/
├── Source Code.ipynb                      # Complete training, parameter tracking, and evaluation notebook
├── README.md                              # This architecture & methodology guide
├── RESULTS.md                             # Comprehensive empirical results, comparison tables, and analysis
└── results/
    ├── experiment_summary.txt             # Plaintext run execution summary and environment details
    ├── comparison_tables/
    │   ├── computational_efficiency.csv   # Training/inference runtimes, memory footprint, parameters
    │   ├── dynamic_weights_test_stats.csv # Dynamic stage weight statistics across test episodes
    │   ├── experiment_config.csv          # Complete hyperparameters and environment metadata
    │   ├── improvement_over_D0.csv        # Absolute and relative test accuracy gains over D0
    │   ├── main_comparison.csv            # Primary comparative metrics and test/val statistics
    │   ├── per_epoch_metrics.csv          # Per-epoch loss, accuracy, timings, and dynamic weight traces
    │   └── three_way_comparison.csv       # Summary table covering key metrics across D0, D1, and D2
    ├── parameter_tracking/
    │   ├── final_parameters.csv           # Checkpoint and end-of-training parameter states
    │   └── parameter_trajectories.csv     # Trajectories of weights, prior, gamma, gate norms, GeM p
    └── plots/
        ├── 01_train_accuracy_vs_epoch.png
        ├── 02_train_loss_vs_epoch.png
        ├── 03_val_accuracy_vs_epoch.png
        ├── 04_val_loss_vs_epoch.png
        ├── 05_test_accuracy_comparison.png     # Embedded in RESULTS.md
        ├── 06_training_time_comparison.png
        ├── 07_inference_time_comparison.png
        ├── 08_gpu_memory_comparison.png
        ├── 09_stage_weight_trajectories.png
        ├── 10_gamma_trajectories.png
        ├── 11_gem_p_trajectories.png
        ├── 12_per_episode_accuracy_boxplot.png
        ├── 13_dw_parameter_trajectories.png
        └── 14_dw_dynamic_weights_per_epoch.png
```

---

## 6. Summary of Key Results

| Experiment | Configuration | Test Accuracy (5w-5s) | 95% Confidence Interval | Delta vs. D0 | Best Val Accuracy | Inference Latency / ep | Peak Train GPU VRAM | Total Parameters |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **D0** | DW only (Baseline) | 82.88% | [82.36%, 83.39%] | — | 82.72% (Ep 1) | **37.11 ms** | **3.89 GB** | 11,618,586 |
| **D1** | **DW + LayerNorm + EMA** | **85.30%** | **[84.82%, 85.78%]** | **+2.42 pp** | 84.36% (Ep 1) | 37.16 ms | 3.93 GB | 11,620,634 |
| **D2** | DW + LayerNorm + EMA + GeM | 85.27% | [84.77%, 85.76%] | +2.39 pp | **84.45%** (Ep 1) | 42.49 ms | 4.67 GB | 11,620,639 |

- **Top Test Performer:** **D1 (DW + LayerNorm + EMA)** achieves **85.30%** 5-way 5-shot test accuracy on MiniImageNet, delivering a large **+2.42 pp** absolute gain over baseline D0 (**82.88%**) and outperforming the original static MSENet baseline (~82.24% – 82.48%) by **+2.82 to +3.06 pp**.
- **Top Validation Performer:** **D2 (DW + LayerNorm + EMA + GeM)** records the highest validation accuracy at **84.45%** and achieves a near-identical test accuracy of **85.27%** (+2.39 pp over D0).
- **Statistically Significant Advantage:** Both D1 and D2 demonstrate paired 95% confidence intervals strictly excluding zero ($[+2.12\text{ pp}, +2.73\text{ pp}]$ for D1 vs. D0; $[+2.08\text{ pp}, +2.70\text{ pp}]$ for D2 vs. D0).
- **Minimal Hardware Overhead:** LayerNorm and EMA add zero noticeable latency (+0.05 ms) and only +2,048 parameters. GeM pooling adds only 5 scalar parameters with modest compute overhead (+5.33 ms/ep latency), operating at high throughput (>23 tasks/s).

For comprehensive per-epoch traces, dynamic weight distributions, parameter trajectories, and detailed hardware benchmarks, refer to [`RESULTS.md`](RESULTS.md).
