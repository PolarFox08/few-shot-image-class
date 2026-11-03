# Training-Stability Ablation Study on MSENet (5-Way Ablations)

## 1. Motivation: Mitigating Overfitting & Training Instability in Few-Shot Learning

In the original MSENet framework:
> **"Enhancing Few-Shot Image Classification through Learnable Multi-Scale Embedding and Attention Mechanisms"**  
> *Fatemeh Askari, Amirreza Fateh, Mohammad Reza Mohammadi* (GitHub: [FatemehAskari/MSENet](https://github.com/FatemehAskari/MSENet))

The architecture uses a pretrained ResNet-18 backbone augmented with multi-scale feature extraction across five stages (`maxpool`, `layer1`, `layer2`, `layer3`, `layer4`), followed by self-attention mechanisms, global average pooling (GAP), and learnable stage distance aggregation:

$$\text{Score}(x_i^q, j) = -\sum_{p=0}^4 w_p \cdot d_{i,j}^p$$

### The Core Problem: Overfitting and Feature Drift under Episodic Training
While MSENet achieves strong few-shot performance on MiniImageNet, training such highly expressive multi-scale architectures with end-to-end backpropagation introduces distinct optimization failure modes:
1. **Severe Episodic Overfitting:** In standard MSENet training (30-way 5-shot training episodes, 15 query images per class), training accuracy rapidly reaches $\sim 99.96\%$ by epoch 6–10, while validation accuracy peaks very early (often at Epoch 1 at $\sim 82.34\%$) and subsequently deteriorates towards $\sim 78.8\%$.
2. **Backbone Drift and Feature Distortion:** Pretrained on ImageNet-1K, the lower and intermediate stages (`conv1`, `layer1`, `layer2`) contain generalizable visual primitives (edges, textures, basic geometries). Unconstrained episodic gradient updates can cause the backbone to drift significantly away from its pretrained initialization, degrading transferability to unseen test classes.
3. **Batch Normalization Instability:** Few-shot episodic batches have distinct distributions from standard MiniImageNet batches. Updating running statistics ($\mu, \sigma^2$) on mini-batches during meta-training can introduce high variance and mismatch between support and query evaluations.
4. **Parameter Weight Divergence:** As training progresses, learnable stage weights and scaling factors ($\gamma_1, \gamma_2$) can oscillate or turn negative if unconstrained, skewing distance metric combinations.

To systematically isolate and address these stability bottlenecks, this experiment investigates **five independent training-stability ablations (A1 through A5)** against the exact unperturbed **Baseline MSENet**.

---

## 2. Experimental Ablations (A1–A5): Conceptual & Mathematical Formulations

Every ablation configuration modifies **strictly one architectural or optimization component** relative to the original MSENet baseline, keeping all seeds, data splits, episode sequences, and evaluation protocols strictly identical.

```
                                  Input Episode
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                             ▼
                   Support Set                     Query Set
                   (5-way 5-shot)               (5-way 15-query)
                         │                             │
                         └──────────────┬──────────────┘
                                        │
                       ┌────────────────┴────────────────┐
                       │     ResNet-18 Backbone Stages   │
                       │   [A1: BN Frozen in Eval Mode]  │
                       │   [A2: Discriminative LR + Stem]│
                       │   [A3: L2-SP Drift Penalty]     │
                       └────────────────┬────────────────┘
                                        │
                         Multi-Scale Self-Attention (x5)
                                        │
                               Global Average Pool
                                        │
                         Euclidean Prototypical Distances
                                        │
                           Stage Distance Aggregation
                                        │
                        ┌───────────────┴───────────────┐
                        │       Loss & Optimization     │
                        │   [A4: EMA Model Weight Avg]  │
                        │   [A5: Label Smoothing (0.1)] │
                        └───────────────────────────────┘
```

### Baseline: Original MSENet
- Standard ResNet-18 backbone fine-tuned end-to-end with Adam optimizer ($\text{lr} = 10^{-4}$ for backbone and heads).
- Standard Cross-Entropy loss over episodic class logits.
- Standard dynamic Batch Normalization tracking in `model.train()` mode.

---

### A1: Freeze Batch Normalization Statistics
- **Mechanism:** Forces all `nn.BatchNorm2d` layers into evaluation mode (`model.eval()`) immediately after every `model.train()` call during episodic training. Running mean ($\mu_{\text{pretrained}}$) and running variance ($\sigma^2_{\text{pretrained}}$) are strictly frozen to their ImageNet pretrained statistics. Learnable affine parameters ($\gamma_{\text{bn}}, \beta_{\text{bn}}$) remain trainable.
- **Hypothesis/Purpose:** Few-shot episodic batches exhibit high variance due to small sample sizes. Freezing running statistics prevents noisy batch-level shifts from destabilizing pretrained feature representations.
- **Formulation:**
  $$\hat{x} = \frac{x - \mu_{\text{pretrained}}}{\sqrt{\sigma^2_{\text{pretrained}} + \epsilon}} \cdot \gamma_{\text{bn}} + \beta_{\text{bn}}$$

---

### A2: Differential (Discriminative) Learning Rates & Frozen Stem/Layer1
- **Mechanism:** Assigns depth-dependent learning rates across network tiers and completely freezes early feature extractors:
  - **Stem (`conv1`, `bn1`) and `layer1`:** Completely frozen ($\text{lr} = 0$, `requires_grad = False`).
  - **`layer2`:** $\text{lr} = 1 \times 10^{-6}$.
  - **`layer3`:** $\text{lr} = 3 \times 10^{-6}$.
  - **`layer4`:** $\text{lr} = 1 \times 10^{-5}$.
  - **Attention Heads & Special Parameters** ($\gamma_1, \gamma_2$, stage weights): $\text{lr} = 5 \times 10^{-3}$.
- **Hypothesis/Purpose:** Allows newly initialized attention modules and metric parameters to learn rapidly without distorting general-purpose low-level pretrained visual features.
- **Formulation:**
  $$\theta_i^{(t+1)} = \theta_i^{(t)} - \eta_i \nabla_{\theta_i} \mathcal{L}$$
  where $\eta_i \in \{0, 10^{-6}, 3 \times 10^{-6}, 10^{-5}, 5 \times 10^{-3}\}$.

---

### A3: $L_2$-SP (Starting Point) Regularization
- **Mechanism:** Imposes an explicit $L_2$ penalty penalizing deviations of the current backbone parameters from their original ImageNet pretrained weights.
- **Hypothesis/Purpose:** Instead of decaying weights toward the origin $\mathbf{0}$ (standard weight decay), $L_2\text{-SP}$ acts as an anchor to preserve pretrained knowledge and prevent catastrophic drift during episodic training.
- **Formulation:**
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \frac{\alpha}{2} \sum_{p \in \Theta_{\text{backbone}}} \left\| \theta_p - \theta_{p, \text{pretrained}} \right\|_2^2$$
  where $\alpha = 10^{-3}$ ($0.001$).

---

### A4: Exponential Moving Average (EMA) of Weights
- **Mechanism:** Maintains an exponential moving average shadow copy of all model state parameters (including floating-point affine parameters and BN statistics) updated after every optimizer step with decay factor $\beta = 0.99$. **The EMA shadow model is strictly used for validation and testing.**
- **Hypothesis/Purpose:** Alleviates episodic parameter oscillation and dampens stochastic noise across gradient updates, generating a smooth, flat-minima ensemble representation.
- **Formulation:**
  $$\theta_{\text{EMA}}^{(t)} = \beta \theta_{\text{EMA}}^{(t-1)} + (1 - \beta) \theta^{(t)}$$
  where $\beta = 0.99$.

---

### A5: Label Smoothing Regularization
- **Mechanism:** Replaces hard one-hot target distributions with smoothed probabilities using a smoothing coefficient $\epsilon = 0.1$.
- **Hypothesis/Purpose:** Few-shot training quickly forces logits to extreme values, causing severe overconfidence and sample memorization. Label smoothing bounds gradient magnitudes and encourages representation clustering without extreme logit margins.
- **Formulation:**
  $$q_i = \begin{cases} 1 - \epsilon + \frac{\epsilon}{K}, & \text{if } i = y \\ \frac{\epsilon}{K}, & \text{otherwise} \end{cases}$$
  $$\mathcal{L}_{\text{CE-smooth}} = - \sum_{i=1}^K q_i \ln p_i$$
  where $K = 30$ during training episodes, $y$ is the ground-truth class, and $\epsilon = 0.1$.

---

## 3. Implementation Details & Hyperparameter Setup

The experimental parameters and runtime environment are documented below:

| Hyperparameter / Setting | Value / Specification |
|:---|:---|
| **Dataset** | MiniImageNet (84 $\times$ 84 resolution, RGB) |
| **Backbone Architecture** | ResNet-18 (Pretrained on ImageNet-1K via TorchVision) |
| **Distance Metric** | Euclidean Distance ($L_2$ Prototypical Metric) |
| **Episodic Training Setup** | 10 Epochs, 360 Episodes/Epoch (30-way, 5-shot, 15 query images/class) |
| **Validation Setup** | 600 Fixed Episodes/Epoch (5-way, 5-shot, 15 query images/class) |
| **Test Evaluation Setup** | 600 Fixed Episodes (5-way, 5-shot, 15 query images/class) |
| **Random Seeds** | Master Seed: `42`; Episode Seeds (Train/Val/Test): `(20263, 20261, 20262)` |
| **Optimization** | Adam Optimizer ($\beta_1 = 0.9, \beta_2 = 0.999$, Cosine Annealing LR Schedule) |
| **Base Learning Rate** | $\text{lr} = 10^{-4}$ (Baseline, A1, A3, A4, A5); Discriminative tiers (A2) |
| **Precision** | Mixed Precision (PyTorch AMP `torch.cuda.amp.autocast`) |
| **Hardware Environment** | Tesla T4 GPU, Python 3.13.15, PyTorch 2.11.0+cu128, EasyFSL 1.5.0 |

---

## 4. Folder Structure & Contents

```
5-way-ablation/
├── Source Code.ipynb                             # Complete executable notebook with full training pipeline
├── README.md                                     # Theoretical foundations, ablation mechanics & guide
├── RESULTS.md                                    # Comprehensive analysis, tables, and metric breakdowns
└── results/
    ├── comparison_tables/
    │   ├── main_comparison.csv                   # Core test/val accuracy, paired deltas, and compute costs
    │   ├── efficiency_metrics.csv                # Latency, throughput, parameters, and memory allocations
    │   ├── final_stage_weights.csv               # Learned multi-scale stage weights across configurations
    │   ├── final_gamma_temperature.csv           # Attention scaling parameters (gamma1, gamma2)
    │   ├── learned_param_stats.csv               # Backbone parameter drift, attention weight norms
    │   ├── per_epoch_metrics.csv                 # Complete 10-epoch training and validation trajectory
    │   ├── per_episode_test_accuracy.csv         # 600-episode raw test accuracy array per model
    │   └── run_environment.csv                   # Environment specs, library versions, seed signatures
    └── plots/
        ├── 01_train_accuracy_vs_epoch.png        # 30-way training accuracy trajectories
        ├── 02_train_loss_vs_epoch.png            # Training loss trajectories
        ├── 03_val_accuracy_vs_epoch.png          # 5-way validation accuracy curves
        ├── 04_val_loss_vs_epoch.png              # 5-way validation loss curves
        ├── 05_test_accuracy_comparison.png       # Primary test accuracy bar chart with 95% CIs
        ├── 06_training_time_comparison.png       # Total training duration comparison
        ├── 07_inference_time_comparison.png      # Latency per episode benchmark
        ├── 08_gpu_memory_comparison.png          # Peak GPU VRAM allocated/reserved
        ├── 09a_stage_weight_trajectories.png     # Evolution of stage weights across epochs
        ├── 09b_gamma_trajectories.png            # Evolution of attention gammas across epochs
        ├── 10_generalisation_gap.png             # Generalization gap (Train vs Val accuracy)
        ├── 11_per_episode_test_accuracy_boxplot.png # Per-episode test accuracy distributions
        ├── 12_paired_difference_vs_baseline.png  # Paired episode difference distributions
        └── 13_backbone_drift_from_pretrained.png # Relative drift of backbone weights from ImageNet
```

---

## 5. Summary of Key Findings

| Configuration | Test Accuracy (%) | 95% CI (%) | $\Delta$ vs. Baseline (pp) | Stat. Significant (95%) | Best Val Acc (%) | Train Time |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **A4 (EMA, $\beta=0.99$)** | **84.12%** | **[83.63%, 84.62%]** | **+1.89 pp** | **Yes (Positive)** | **83.75%** | 28.06 min |
| **A5 (Label Smoothing 0.1)** | **82.52%** | **[81.98%, 83.05%]** | **+0.28 pp** | **Yes (Positive)** | 82.38% | 28.38 min |
| **A1 (Frozen BatchNorm)** | **82.33%** | **[81.79%, 82.87%]** | +0.10 pp | No | 82.00% | 27.93 min |
| **Baseline (Original)** | **82.24%** | **[81.72%, 82.75%]** | 0.00 pp | — | 82.34% | 28.11 min |
| **A3 ($L_2$-SP, $\alpha=10^{-3}$)** | **82.24%** | **[81.72%, 82.75%]** | 0.00 pp | No | 82.22% | 28.63 min |
| **A2 (Discriminative LR)** | **78.89%** | **[78.33%, 79.46%]** | **-3.34 pp** | **Yes (Negative)** | 78.59% | 23.41 min |

- **Clear Winner — Exponential Moving Average (A4):** Achieving **84.12%** test accuracy (+1.89 pp statistically significant improvement, paired $p < 0.05$), EMA effectively resolves parameter oscillation and regularizes the high-capacity multi-scale network without extra inference latency.
- **Mild Benefit — Label Smoothing (A5):** Delivers a statistically significant improvement of **+0.28 pp** (82.52%), mitigating overconfident episodic predictions.
- **Catastrophic Failure — Discriminative Learning Rates (A2):** Restricting backbone learning rates severely starves representation adaptation, resulting in a **-3.34 pp** drop (78.89%) and late-epoch training collapse.
- **Invariance — Frozen BatchNorm (A1) and $L_2$-SP (A3):** Neither A1 nor A3 meaningfully alters performance under early checkpointing (Epoch 1), as parameter drift has not yet accumulated significantly at Epoch 1.

For comprehensive tables, epoch-by-epoch progressions, paired statistical evaluations, and parameter drift analysis, see [`RESULTS.md`](file:///c:/Ashwin/College/Semester-5/DL Lab/few-shot-image-class/5-way-ablation/RESULTS.md).
