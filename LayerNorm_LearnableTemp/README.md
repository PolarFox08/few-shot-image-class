# Stage-Wise Layer Normalization and Learnable Temperature Scaling for MSENet

## 1. Motivation: Feature Scale Imbalances and Probability Calibration

In the original MSENet paper:
> **"Enhancing Few-Shot Image Classification through Learnable Multi-Scale Embedding and Attention Mechanisms"**  
> *Fatemeh Askari, Amirreza Fateh, Mohammad Reza Mohammadi* (GitHub: [FatemehAskari/MSENet](https://github.com/FatemehAskari/MSENet))

The multi-scale architecture extracts feature representations from five stages of a ResNet-18 backbone (`maxpool`, `layer1`, `layer2`, `layer3`, `layer4`). After stage-wise self-attention mechanisms and Global Average Pooling (GAP), the model forms class prototypes and computes unnormalized Euclidean distances across all five stages, combining them via a static vector of learnable stage weights $w = [w_1, w_2, w_3, w_4, w_5]$:

$$\text{Distance}(x^q, c_j) = \sum_{s=1}^5 w_s \cdot \|z_s(x^q) - c_j^s\|_2^2$$

The class probability distribution is then obtained by passing the negative weighted distances directly into a standard Softmax function:

$$P(y = j \mid x^q) = \frac{\exp(-\text{Distance}(x^q, c_j))}{\sum_{k=1}^N \exp(-\text{Distance}(x^q, c_k))}$$

### Limitations of the Original Architecture

1. **Inter-Stage Scale and Variance Discrepancies:**  
   Across the five ResNet-18 stages, channel dimensions increase progressively ($64 \to 64 \to 128 \to 256 \to 512$). Deeper stages with 512 channels naturally accumulate higher Euclidean $L_2$ vector norms simply due to dimensional accumulation ($\sum_{c=1}^D (z_c - c_c)^2$), compared to shallow 64-channel stages. Furthermore, non-linear activations and attention scaling introduce shifting variance distributions across feature dimensions. Without channel-wise normalization, Euclidean distance comparisons are disproportionately dominated by dimensional scaling rather than discriminative class boundaries.

2. **Uncalibrated Metric Softmax Sharpness:**  
   In metric-based few-shot learning, raw negative Euclidean distances typically do not match the natural logit scale expected by standard cross-entropy loss. In traditional classification, final dense layer logits adapt their dynamic range through weight magnitudes. In prototypical metric networks, raw squared or $L_2$ Euclidean distances can be arbitrarily sharp or diffuse depending on feature magnitudes. Without a dedicated temperature hyperparameter or learnable scaling factor, the categorical Softmax distribution is susceptible to overconfidence or underconfidence during episodic training.

---

## 2. Architectural Modifications & Novel Contributions

To address both feature normalization across stages and logit calibration during metric classification, this study implements and evaluates two architectural enhancements both in isolation and in joint combination:

```
                          Support & Query Images
                                    │
                                    ▼
                   ResNet-18 Multi-Scale Feature Maps
           ┌───────────┬───────────┬───────────┬───────────┐
           ▼           ▼           ▼           ▼           ▼
       [maxpool]   [layer1]    [layer2]    [layer3]    [layer4]
        (64 ch)     (64 ch)    (128 ch)    (256 ch)    (512 ch)
           │           │           │           │           │
     Self-Attention Self-Attention Self-Attention Self-Attention Self-Attention
           │           │           │           │           │
          GAP         GAP         GAP         GAP         GAP
           │           │           │           │           │
           ▼           ▼           ▼           ▼           ▼
     ┌───────────────────────────────────────────────────────────┐
     │  Novelty 1: Stage-Wise Layer Normalization (LayerNorm)     │
     │  (Shared across Support & Query, separate LN per stage)   │
     │  LN(64)       LN(64)      LN(128)     LN(256)     LN(512) │
     └─────────────────────────────┬─────────────────────────────┘
                                   │
                                   ▼
             Support Prototypes c_j^s  &  Query Embeddings z_s(x^q)
                                   │
                                   ▼
                   Stage-Wise Euclidean Distances:
                       d_s = ||z_s(x^q) - c_j^s||_2
                                   │
                                   ▼
                   Weighted Distance Aggregation:
                       D(x^q, c_j) = Σ w_s * d_s
                                   │
                                   ▼
     ┌───────────────────────────────────────────────────────────┐
     │  Novelty 2: Learnable Temperature Scaling (Softplus)      │
     │  T = Softplus(t_raw),  Logits = -D(x^q, c_j) / T          │
     └─────────────────────────────┬─────────────────────────────┘
                                   │
                                   ▼
                     Categorical Probability:
                       P(y = j | x^q) = Softmax(-D / T)
```

### 1. Stage-Wise Layer Normalization (LayerNorm)
Layer Normalization is introduced immediately after Global Average Pooling (GAP) at each of the five ResNet-18 feature extraction stages, applied identically to support representations and query representations.

For an embedding vector $x \in \mathbb{R}^D$ at a given stage $s \in \{1, 2, 3, 4, 5\}$:

$$\mu = \frac{1}{D} \sum_{k=1}^D x_k, \qquad \sigma^2 = \frac{1}{D} \sum_{k=1}^D (x_k - \mu)^2$$

$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

$$y = \gamma \odot \hat{x} + \beta$$

Where:
- $\epsilon = 1 \times 10^{-5}$ prevents division by zero.
- $\gamma \in \mathbb{R}^D$ and $\beta \in \mathbb{R}^D$ are learnable affine gain and bias parameters initialized to $\mathbf{1}$ and $\mathbf{0}$ respectively.
- Separate LayerNorm instances are constructed for each stage to accommodate channel dimensions ($D_1=64, D_2=64, D_3=128, D_4=256, D_5=512$).
- **Total LayerNorm parameter overhead:** $2 \times (64 + 64 + 128 + 256 + 512) = 2{,}048$ parameters.

**Theoretical Justification:**  
By normalizing the channel embedding across each feature stage to zero mean and unit variance before computing prototypes and distances, LayerNorm standardizes the scale of each stage vector. This stabilizes feature variances across differing channel capacities, ensuring that multi-stage Euclidean distances measure structural and semantic divergence rather than unstandardized activation scales.

### 2. Learnable Temperature Scaling
In standard prototypical metric networks, categorical probabilities are formed directly from negative distances:

$$P(y = j \mid x^q) = \operatorname{Softmax}(-d(x^q, c_j))$$

We introduce a scalar learnable temperature parameter $T > 0$ that dynamically rescales the logit magnitude:

$$P(y = j \mid x^q) = \operatorname{Softmax}\left(\frac{-d(x^q, c_j)}{T}\right)$$

To guarantee strict positivity throughout optimization without numerical collapse, the temperature parameter is parameterized via the Softplus transformation:

$$T = \operatorname{Softplus}(t_{\text{raw}}) = \ln(1 + \exp(t_{\text{raw}}))$$

- **Initialization:** $T_{\text{init}} = 1.0$, which corresponds to an initial unconstrained parameter value $t_{\text{raw}} = \ln(\exp(1.0) - 1) \approx 0.5413$.
- **Parameter Overhead:** 1 scalar parameter.

**Theoretical Justification:**  
The temperature parameter adjusts the entropy of the predicted categorical distribution. If $T > 1$, probability peaks are softened, smoothing gradient updates across hard negatives. If $T < 1$, confident predictions are sharpened, penalizing borderline confusions. By making $T$ learnable, gradient backpropagation optimizes metric logit calibration alongside backbone representations.

---

## 3. Systematic 4-Variant Factorial Design

To understand the individual contributions and mutual interactions of these modifications, four architectural variants were trained and benchmarked under identical conditions:

1. **Variant 1: Baseline (Unchanged):**  
   Standard MSENet architecture with ResNet-18 backbone, multi-scale feature extraction (`maxpool`, `layer1`, `layer2`, `layer3`, `layer4`), stage-wise self-attention, learnable stage weights, Euclidean prototypical distance, and fixed unit temperature ($T = 1$).
2. **Variant 2: LayerNorm Only:**  
   Standard MSENet + five stage-wise LayerNorm modules after GAP. Fixed unit temperature ($T = 1$).
3. **Variant 3: Learnable Temperature Only:**  
   Standard MSENet + learnable Softplus temperature scaling parameter $T$. No LayerNorm.
4. **Variant 4: LayerNorm + Temperature (Combined):**  
   Standard MSENet + five stage-wise LayerNorm modules + learnable Softplus temperature scaling parameter $T$.

---

## 4. Implementation Details & Hyperparameters

All four variants were implemented, trained, and evaluated in the unified notebook [`Source Code.ipynb`](Source Code.ipynb) on a single **NVIDIA Tesla T4 GPU (16 GB)** using PyTorch with Automatic Mixed Precision (`torch.cuda.amp.autocast`).

### Hyperparameter Configuration

| Parameter | Configuration Value | Description |
|:---|:---:|:---|
| **Backbone Architecture** | ResNet-18 | Pretrained on ImageNet (weights loaded successfully) |
| **Input Resolution** | $84 \times 84 \times 3$ | Standard MiniImageNet benchmark resolution |
| **Benchmark Dataset** | MiniImageNet | 100 classes total: 64 train, 16 validation, 20 test |
| **Training Episode Setup** | 30-way 5-shot | 30 classes per episode, 5 support + 15 query images |
| **Evaluation Episode Setup** | 5-way 5-shot | 5 classes per episode, 5 support + 15 query images |
| **Query Set Size** | 15 images / class | 75 query images evaluated per 5-way test episode |
| **Total Epochs** | 10 epochs | 360 episodes per epoch (3,600 training episodes total) |
| **Validation Tasks** | 600 episodes / epoch | Evaluated at the conclusion of every training epoch |
| **Final Test Tasks** | 600 episodes | Paired across all 4 models using deterministic seed 2026 |
| **Optimizer** | Adam | Cosine Annealing learning rate schedule |
| **Backbone Learning Rate** | $1 \times 10^{-4}$ | Learning rate for ResNet-18 layers |
| **Stage Weights LR** | $5 \times 10^{-3}$ | Learning rate for learnable stage weights $w_1 - w_5$ |
| **LayerNorm Learning Rate** | $1 \times 10^{-4}$ | Learning rate for LayerNorm $\gamma$ and $\beta$ |
| **Temperature LR** | $5 \times 10^{-3}$ | Learning rate for $t_{\text{raw}}$ |
| **Initial Stage Weights** | `[1.0, 1.1, 1.2, 1.3, 1.4]` | Initial weights prior to episodic gradient updates |
| **Initial Attention Scaling** | $\gamma_1 = 0.2, \gamma_2 = 0.2$ | Support ($\gamma_1$) and query ($\gamma_2$) attention scale factors |
| **Initial Temperature** | $T = 1.0$ | Corresponding to $t_{\text{raw}} \approx 0.5413$ |
| **Distance Metric** | Euclidean ($L_2$) | Stage-wise Euclidean prototype distances |
| **Random Seeds** | Global: 42, Eval: 2026 | Deterministic episode hash validation & test sets |
| **Val Episode Hash** | `8bff73622cef` | Exact identical validation sequence across all variants |
| **Test Episode Hash** | `c243ad17abe6` | Exact identical test sequence across all variants |

---

## 5. Directory Structure & File Manifest

```
LayerNorm_LearnableTemp/
├── README.md                                    # Architectural guide, mathematical formulations, setup
├── RESULTS.md                                   # Comprehensive experimental results, tables, and analysis
├── Source Code.ipynb                            # Complete experimental notebook (training & evaluation)
└── results/
    ├── learned_parameters.json                  # Final learned parameter tensors & LayerNorm statistics
    ├── comparison_tables/                       # Exported benchmark CSV comparison tables
    │   ├── config_and_environment.csv           # System hardware, environment, and hyperparameter log
    │   ├── layernorm_summary.csv                # LayerNorm weight/bias stats across stages for Variants 2 & 4
    │   ├── learned_parameters.csv               # Learned stage weights, gammas, and temperatures
    │   ├── paired_vs_baseline.csv               # Paired t-tests, p-values, and win/tie/loss tallies
    │   ├── paper_style_table.csv                # Compact benchmark table formatted for publication
    │   ├── per_episode_test_accuracy.csv        # Raw accuracy per episode (600 test episodes x 4 variants)
    │   ├── per_epoch_history.csv                # Per-epoch training/val loss, accuracy, and parameter history
    │   └── summary_comparison.csv               # Comprehensive multi-metric summary table
    └── plots/                                   # High-resolution visualization figures
        ├── 1_test_accuracy_ci.png               # Main test accuracy comparison plot with 95% CIs
        ├── 2_training_accuracy_per_epoch.png    # Training accuracy trajectories across 10 epochs
        ├── 3_training_loss_per_epoch.png        # Training loss descent across 10 epochs
        ├── 4_validation_accuracy_per_epoch.png  # Validation accuracy trajectories across 10 epochs
        ├── 5_validation_loss_per_epoch.png      # Validation loss trajectories across 10 epochs
        ├── 6_training_time.png                  # Total and per-epoch training time comparison
        ├── 7_inference_time.png                 # Inference latency per episode and query image
        ├── 8_gpu_memory.png                     # Peak GPU memory allocations (training vs. inference)
        ├── 9_learned_weights_and_temperature.png# Learned stage weights and calibrated temperatures
        ├── 10_four_variant_accuracy_comparison.png # Multi-metric validation and test performance
        └── 11_learning_rate_trajectory.png      # Cosine annealing learning rate schedules
```

---

## 6. Summary of Key Findings

| Model Variant | Test Accuracy (%) | 95% Confidence Interval (%) | $\Delta$ vs. Baseline (pp) | Best Val Accuracy (%) | Paired Win / Tie / Loss | Extra Params |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. Baseline (unchanged)** | 82.48% | [81.94%, 83.01%] | — | 82.73% | — | 0 |
| **2. LayerNorm only** | 83.06% | [82.54%, 83.58%] | **+0.58 pp** | 82.66% | **285 / 99 / 216** | +2,048 (+0.018%) |
| **3. Learnable temp only** | 82.47% | [81.94%, 83.00%] | -0.00 pp | 82.55% | 200 / 181 / 219 | +1 (+0.000%) |
| **4. LayerNorm + temp** | **83.18%** | **[82.66%, 83.71%]** | **+0.71 pp** | **82.79%** | **297 / 88 / 215** | +2,049 (+0.018%) |

- **LayerNorm is the Decisive Factor:** Stage-wise Layer Normalization delivers a statistically significant performance boost of **+0.58 pp** ($p = 0.00012$). Standardizing feature scales before Euclidean distance calculation allows multi-scale stages to contribute based on feature quality rather than raw dimensional channel magnitude.
- **Synergy with Temperature Scaling:** While learnable temperature scaling in isolation has virtually zero effect (-0.002 pp), it exhibits strong positive synergy when combined with LayerNorm, achieving the top test accuracy of **83.18%** (**+0.71 pp** over baseline, $p = 5.0 \times 10^{-6}$).
- **Negligible Resource Overhead:** The combined model adds only 2,049 parameters (+0.018% total parameter increase) and maintains identical inference latency (**42.46 ms/episode** vs. **42.43 ms/episode** for baseline).

For full empirical evaluation, statistical hypothesis tests, per-epoch trajectories, and parameter analyses, refer to [`RESULTS.md`](RESULTS.md).
