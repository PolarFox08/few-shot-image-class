# Stacked MSENet with Cosine Similarity: Component Ablations (EMA, LayerNorm, GeM)

## 1. Motivation: Beyond Baseline Multi-Scale Embeddings in Cosine Space

In the original MSENet architecture:
> **"Enhancing Few-Shot Image Classification through Learnable Multi-Scale Embedding and Attention Mechanisms"**  
> *Fatemeh Askari, Amirreza Fateh, Mohammad Reza Mohammadi* (GitHub: [FatemehAskari/MSENet](https://github.com/FatemehAskari/MSENet))

The multi-scale architecture extracts feature representations across five intermediate stages of a ResNet-18 backbone (`maxpool`, `layer1`, `layer2`, `layer3`, and `layer4`), enriches them with stage-wise spatial self-attention, pools them with Global Average Pooling (GAP), and computes class prototypes by averaging support embeddings. 

When adapted to a **Cosine Similarity** metric, the query classification score is computed via temperature-scaled cosine similarity across all five stages using learnable stage weights $w = [w_1, w_2, w_3, w_4, w_5]$:

$$\text{Score}(x_i^q, j) = \tau \sum_{s=1}^5 w_s \cdot \cos(z_q^s, c_j^s) = \tau \sum_{s=1}^5 w_s \cdot \frac{\langle z_q^s, c_j^s \rangle}{\|z_q^s\|_2 \|c_j^s\|_2}$$

Where $\tau$ is the cosine temperature scaling parameter (initialized to $10.0$).

### Limitations of the Baseline MSENet Pipeline Under Cosine Distance
While directional cosine distance naturally bounds pairwise metrics to $[-1, 1]$ and removes raw vector magnitude disparities, several architectural and optimization challenges remain:
1. **Loss of Salient Local Spatial Cues via GAP:** Standard Global Average Pooling averages all spatial locations uniformly ($H_s \times W_s$). In few-shot classification, informative foreground features (e.g., discriminative textures or key object parts) are diluted by uninformative background regions prior to prototype construction.
2. **Channel Distribution Shift and Intra-Stage Imbalance:** While L2 normalization normalizes vector length, intermediate ResNet stages exhibit extreme channel activation variances. Without channel-wise normalization post-pooling, certain dominant feature channels skew directional prototype alignment.
3. **Episodic Optimization Instability & Meta-Gradient Variance:** Training on episodic mini-batches (360 tasks/epoch at 30-way 5-shot) induces high stochastic gradient noise. Deep feature representations can undergo abrupt trajectory fluctuations, leading to suboptimal convergence and generalization gaps between training and evaluation.

---

## 2. The Novelty: Stacked Architectural Enhancements

To tackle these challenges, this experiment systematically evaluates and stacks three complementary architectural and optimization enhancements within the **Cosine Distance** regime:

```
                          Support & Query Images
                                    │
                                    ▼
                         ResNet-18 Backbone Stages
                 [maxpool]   [layer1]   [layer2]   [layer3]   [layer4]
                     │           │          │          │          │
                     ▼           ▼          ▼          ▼          ▼
              Stage-Wise Self-Attention Modules (with learnable gamma1, gamma2)
                     │           │          │          │          │
                     ▼           ▼          ▼          ▼          ▼
             ┌────────────────────────────────────────────────────────┐
             │       Signed Generalized Mean (GeM) Pooling            │
             │           p_s = Softplus(raw_s) + 0.01                 │
             └────────────────────────────────────────────────────────┘
                                    │
                                    ▼
             ┌────────────────────────────────────────────────────────┐
             │         Stage-Specific Layer Normalization             │
             │         y = LayerNorm(z_s) across channels             │
             └────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                  Class Prototype & Cosine Similarity
                  cos(z_q, c_j) = <z_q, c_j> / (||z_q||_2 * ||c_j||_2)
                                    │
                                    ▼
                       Weighted Multi-Scale Fusion
                   Score(x_q, j) = tau * Σ w_s * cos_s
                                    │
                                    ▼
             ┌────────────────────────────────────────────────────────┐
             │     Exponential Moving Average (EMA) Evaluation        │
             │       θ_EMA^(t) = β * θ_EMA^(t-1) + (1-β) * θ^(t)      │
             └────────────────────────────────────────────────────────┘
```

### Mathematical Formulation of Stacked Components

#### 1. Signed Generalized Mean (GeM) Pooling
Generalized Mean (GeM) pooling replaces standard Global Average Pooling (GAP) in the post-attention stage. Because the self-attention residual formulation ($y = \gamma \cdot \text{Attn}(x) + x$) can produce negative activations, a **Signed GeM** formulation is implemented via positive/negative decomposition:

$$x = x^+ - x^- \quad \text{where } x^+ = \max(x, 0), \; x^- = \max(-x, 0)$$

$$\operatorname{GeM}(x, p_s) = \operatorname{GeM}_{p_s}(x^+) - \operatorname{GeM}_{p_s}(x^-)$$

For non-negative tensor components, GeM is computed in a numerically stable log-sum-exp form:

$$\operatorname{GeM}_{p_s}(t) = \exp\left(\frac{1}{p_s}\left(\operatorname{logsumexp}(p_s \cdot \ln(\operatorname{clamp}(t, \epsilon))) - \ln(|\Omega|)\right)\right)$$

Where:
- $|\Omega| = H_s \times W_s$ denotes the spatial resolution of stage $s$.
- $p_s$ is a strictly positive, learnable power parameter parameterized via Softplus:
  $$p_s = \operatorname{Softplus}(\text{gem\_raw}_s) + p_{\text{floor}}, \quad p_{\text{floor}} = 0.01$$
- Initialized at $p_s = 3.0$ with $\epsilon = 10^{-6}$. When $p_s = 1$, GeM mathematically reduces to GAP; as $p_s \to \infty$, it approaches max pooling, allowing the model to adaptively focus on salient spatial features rather than uniform spatial averages.

#### 2. Layer Normalization (LayerNorm)
LayerNorm is applied immediately after spatial pooling across each of the five ResNet-18 feature stages, prior to class prototype calculation and cosine similarity computation:

$$\hat{z}_s = \frac{z_s - \mu_s}{\sqrt{\sigma_s^2 + \epsilon_{\text{LN}}}}$$

$$y_s = \gamma_s \odot \hat{z}_s + \beta_s$$

Where:
- $\mu_s = \frac{1}{C_s}\sum_{c=1}^{C_s} z_{s,c}$ and $\sigma_s^2 = \frac{1}{C_s}\sum_{c=1}^{C_s}(z_{s,c} - \mu_s)^2$ compute the mean and variance across channel dimension $C_s \in \{64, 64, 128, 256, 512\}$.
- $\gamma_s, \beta_s \in \mathbb{R}^{C_s}$ represent learnable affine scale and shift parameters.

**Effect in Cosine Space:** Normalizing across channel dimensions centers and standardizes feature distributions before projection onto the unit hypersphere. This prevents high-variance feature channels from biasing directional angle comparisons.

#### 3. Exponential Moving Average (EMA)
To stabilize optimization across episodic meta-training, an exponential moving average of all model parameters (and floating-point Batch Normalization running statistics) is maintained after every optimizer step:

$$\theta_{\text{EMA}}^{(t)} = \beta \cdot \theta_{\text{EMA}}^{(t-1)} + (1 - \beta) \cdot \theta^{(t)}$$

Where:
- Decay rate $\beta = 0.99$.
- The shadow weights $\theta_{\text{EMA}}$ are detached from backpropagation and exclusively evaluated during validation and test benchmarking. This smooths out high-frequency gradient noise and stabilizes prototype geometry.

---

## 3. Systematic Stacked Experiment Configurations

The framework evaluates four distinct configurations under identical experimental conditions and seeds to isolate the individual and combined contributions of each component:

| Experiment Key | Configuration Name | Exponential Moving Average (EMA) | Generalized Mean (GeM) | Layer Normalization (LayerNorm) |
|:---:|:---|:---:|:---:|:---:|
| **C0** | **Original Baseline** | Disabled (`False`) | Disabled (`False` — GAP) | Disabled (`False`) |
| **C1** | **EMA + LayerNorm + GeM** | Enabled (`True`, $\beta=0.99$) | Enabled (`True`, $p_{\text{init}}=3.0$) | Enabled (`True`) |
| **C2** | **EMA + GeM** | Enabled (`True`, $\beta=0.99$) | Enabled (`True`, $p_{\text{init}}=3.0$) | Disabled (`False`) |
| **C3** | **EMA + LayerNorm** | Enabled (`True`, $\beta=0.99$) | Disabled (`False` — GAP) | Enabled (`True`) |

All configurations utilize **Cosine Similarity** ($\text{Score} = \tau \sum w_s \cos(z_q^s, c_j^s)$ with $\tau=10.0$), ImageNet-pretrained ResNet-18 initialization, and cross-entropy classification on 5-way 5-shot episodes.

---

## 4. Folder Structure & Contents

```
Stacked-Cosine/
├── Source Code.ipynb                             # Complete end-to-end benchmark notebook
├── README.md                                     # This architecture, methodology & reproducibility guide
├── RESULTS.md                                    # Comprehensive results, tables, and comparative analysis
└── results/
    ├── experiment_summary.txt                    # Auto-generated runtime parameters and environment report
    ├── comparison_tables/
    │   ├── computational_efficiency.csv          # Parameter counts, VRAM, and per-episode timing
    │   ├── experiment_config.csv                 # Detailed configuration parameters and environment info
    │   ├── improvement_over_baseline.csv         # Absolute and relative accuracy gains over C0
    │   ├── main_comparison.csv                   # Comprehensive test metrics, 95% CIs, and validation acc
    │   └── per_epoch_metrics.csv                 # Epoch-by-epoch loss, accuracy, and learned parameters
    ├── parameter_tracking/
    │   ├── final_parameters.csv                  # Checkpoint and epoch-10 parameters (weights, gammas, p)
    │   └── parameter_trajectories.csv            # Step-by-step optimization trajectories across all 10 epochs
    └── plots/
        ├── 01_train_accuracy_vs_epoch.png        # Training accuracy curves across epochs
        ├── 02_train_loss_vs_epoch.png            # Training loss trajectories
        ├── 03_val_accuracy_vs_epoch.png          # Validation accuracy progression per epoch
        ├── 04_val_loss_vs_epoch.png              # Validation loss progression per epoch
        ├── 05_test_accuracy_comparison.png       # Main test accuracy comparison with 95% CIs
        ├── 06_training_time_comparison.png       # Total training duration comparison
        ├── 07_inference_time_comparison.png      # Per-episode inference latency comparison
        ├── 08_gpu_memory_comparison.png          # Peak GPU VRAM utilization across training & inference
        ├── 09_stage_weight_trajectories.png      # Evolution of multi-scale stage weights w_1 to w_5
        ├── 10_gamma_trajectories.png             # Evolution of attention scaling parameters gamma1 & gamma2
        ├── 11_gem_p_trajectories.png             # Learned GeM power parameters p_1 to p_5 across epochs
        └── 12_per_episode_accuracy_boxplot.png   # Boxplot distribution of 600 test episode accuracies
```

---

## 5. Summary of Key Results

| Experiment | Configuration Description | Test Accuracy (5w-5s) | 95% Confidence Interval | Absolute Gain vs. Baseline | Best Validation Accuracy | Inference Time / ep | Peak Train VRAM |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **C0** | Original Baseline | 82.0711% | [81.5527%, 82.5895%] | — | 82.5289% (Epoch 1) | **35.57 ms** | **3.88 GB** |
| **C1** | EMA + LayerNorm + GeM | 84.4311% | [83.9355%, 84.9267%] | +2.3600 pp (+2.88%) | 83.9778% (Epoch 1) | 40.27 ms | 4.67 GB |
| **C2** | **EMA + GeM** | **84.5311%** | **[84.0267%, 85.0355%]** | **+2.4600 pp (+3.00%)** | 84.1889% (Epoch 1) | 40.41 ms | 4.67 GB |
| **C3** | **EMA + LayerNorm** | **84.5311%** | **[84.0400%, 85.0222%]** | **+2.4600 pp (+3.00%)** | **84.4244% (Epoch 1)** | **35.51 ms** | 3.93 GB |

- **Substantial Performance Elevation:** All three enhanced configurations (**C1**, **C2**, **C3**) achieve a notable gain of **+2.36 pp to +2.46 pp** over the unaugmented Cosine MSENet baseline (**C0** at 82.07%).
- **Dual Champions in Cosine Space:**
  - **C3 (EMA + LayerNorm)** achieves **84.5311%** (+2.46 pp) with the highest validation accuracy (**84.42%**), fastest inference latency (**35.51 ms**), and negligible parameter addition (+2,048 parameters, +0.018%).
  - **C2 (EMA + GeM)** also achieves **84.5311%** (+2.46 pp) with only 5 additional scalar parameters (+0.00004% overhead).
  - **C1 (EMA + LayerNorm + GeM)** reaches **84.4311%** (+2.36 pp). While both individual modules yield ~+2.46 pp, stacking both under cosine normalization achieves near-identical high-accuracy regimes without negative interference.
- **Statistical Significance:** Paired 95% confidence intervals versus baseline strictly exclude zero for all models ($\text{CI95}_{\text{paired}} \approx \pm 0.24 - 0.28\text{ pp}$), establishing statistical significance at $p \ll 0.001$.

For comprehensive epoch-by-epoch progressions, parameter trajectories, paired significance tests, and hardware profiling, refer to [`RESULTS.md`](RESULTS.md).

---

## 6. Reproducibility & Environment Details

All experiments were executed in an isolated single-GPU environment with fixed deterministic episode seeds:
- **Operating Environment:** Linux (Ubuntu / Kaggle Kernel)
- **Python Version:** `3.13.16`
- **PyTorch Version:** `2.11.0+cu130`
- **Torchvision Version:** `0.26.0+cu130`
- **EasyFSL Version:** `1.5.0`
- **Hardware Acceleration:** NVIDIA Tesla T4 GPU (CUDA enabled, Automatic Mixed Precision / AMP active)
- **Seeds:**
  - Global SEED: `42`
  - Training Episode RNG: `20263`
  - Validation Episode RNG: `20261`
  - Test Episode RNG: `20262`
- **Optimization Parameters:**
  - Backbone Optimizer: Adam ($\text{lr} = 10^{-4}$)
  - Stage Weights & Gamma Optimizer: Adam ($\text{lr} = 5 \times 10^{-3}$)
  - GeM Exponent Optimizer: Adam ($\text{lr} = 10^{-3}$)
  - Learning Rate Schedule: Cosine Annealing over 10 epochs
  - Training Protocol: 360 episodes per epoch, 30-way 5-shot training episodes
  - Evaluation Protocol: 600 validation tasks and 600 test tasks, 5-way 5-shot with 15 queries/class (75 queries/task)
  - Distance Metric: Cosine ($\tau_{\text{init}} = 10.0$)
