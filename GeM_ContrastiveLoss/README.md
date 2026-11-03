# Generalized Mean (GeM) Pooling and Auxiliary Supervised Contrastive Loss for MSENet

This experiment investigates two architectural and objective modifications to **MSENet** (*Multi-Scale Embedding Network for Few-Shot Image Classification*):
1. **Generalized Mean (GeM) Pooling** with stage-wise learnable pooling exponents ($p$) to replace Global Average Pooling (GAP) following stage self-attention.
2. **Auxiliary Supervised Contrastive Loss (SupCon)** with a learnable temperature parameter ($\tau$) acting across multi-scale feature embeddings during episodic training.

---

## 1. Motivation & Problem Statement

### 1.1 The Limitation of Global Average Pooling (GAP) in MSENet
In standard few-shot metric learners like MSENet, feature representations from intermediate backbone stages (`maxpool`, `layer1`, `layer2`, `layer3`, `layer4`) pass through self-attention mechanisms before spatial pooling:
$$\mathbf{y} = \gamma \cdot \operatorname{Attn}(\mathbf{x}) + \mathbf{x}$$
Following this attention residual, spatial dimensions $(H \times W)$ are collapsed into an embedding vector via Global Average Pooling (GAP):
$$\operatorname{GAP}(\mathbf{y}_c) = \frac{1}{|\Omega|} \sum_{u \in \Omega} y_{c, u}$$
where $\Omega$ represents the spatial grid of size $H \times W$.

**Problem:** Standard GAP treats all spatial positions with uniform weighting. In few-shot image classification, salient discriminative foreground cues (e.g., bird beaks, animal eyes, localized object boundaries) often occupy compact spatial regions, while the remaining spatial grid comprises background clutter. GAP dampens and dilutes these critical peak activations by averaging them with non-informative regions. Conversely, pure Max-Pooling captures only the single maximum activation, discarding spatial context and destabilizing gradient flow.

### 1.2 The Limitation of Episodic Cross-Entropy Loss Alone
MSENet trains the backbone and multi-scale weighting parameters purely using episodic Cross-Entropy (CE) loss over query predictions:
$$\mathcal{L}_{\text{CE}} = -\sum_{i \in Q} \log \frac{\exp(-\sum_s w_s d_s(z_q^s, c_{y_i}^s))}{\sum_k \exp(-\sum_s w_s d_s(z_q^s, c_k^s))}$$

**Problem:** Episodic cross-entropy optimizes relative margins with respect to class prototypes within a specific random support set. However, it does not explicitly enforce metric compactness (intra-class clustering) and inter-class geometric separation across all support and query instances simultaneously at each stage. When sample counts per class are low (e.g., 1-shot or 5-shot), representations can overfit to specific prototype boundaries rather than learning a structured metric manifold.

---

## 2. Theoretical Concepts & Mathematical Formulations

To address these limitations, we evaluate four systematic configurations in an exhaustive $2 \times 2$ factorial ablation:
- **Baseline**: Original MSENet (GAP + Episodic Cross-Entropy).
- **GeM only**: Stage-wise Signed Generalized Mean Pooling + Episodic Cross-Entropy.
- **Contrastive only**: GAP + Episodic Cross-Entropy + Auxiliary Supervised Contrastive Loss.
- **GeM + Contrastive**: Signed GeM Pooling + Multi-task Joint Loss.

```
                          Input Images (Support & Query)
                                        │
                                        ▼
                     ResNet-18 Backbone Feature Extractor
       ┌───────────────┬────────────────┬───────────────┬───────────────┐
       ▼               ▼                ▼               ▼               ▼
Stage 0 (maxpool) Stage 1 (layer1) Stage 2 (layer2) Stage 3 (layer3) Stage 4 (layer4)
   (64, 21×21)     (64, 21×21)     (128, 11×11)     (256, 6×6)      (512, 3×3)
       │               │                │               │               │
       ▼               ▼                ▼               ▼               ▼
   Self-Attn       Self-Attn        Self-Attn       Self-Attn       Self-Attn
  y = γ·Attn+x    y = γ·Attn+x     y = γ·Attn+x    y = γ·Attn+x    y = γ·Attn+x
       │               │                │               │               │
       └───────────────┼────────────────┼───────────────┼───────────────┘
                       │
       ┌───────────────┴────────────────┐
       │ [Signed GeM Pooling vs. GAP]  │
       │   y+ = max(y, 0), y- = max(-y, 0)
       │   z_s = GeM_p(y+) - GeM_p(y-)  │
       └───────────────┬────────────────┘
                       │
       ┌───────────────┴────────────────────────────────┐
       │                                                │
       ▼                                                ▼
Metric Classification Pipeline           Auxiliary Metric Regularization
Prototypes c_k^s = Mean(z_{i \in S_k}^s)   z_all = [z_support; z_query]
Stage Distances d_s(z_q, c_k)            L2 Normalization: \hat{z} = z / ||z||_2
Weighted Score: - \sum_s w_s d_s         Pairwise Cosine Sim: \hat{z}_i · \hat{z}_j / τ
       │                                                │
       ▼                                                ▼
Episodic Cross-Entropy (L_CE)            Supervised Contrastive Loss (L_con)
       │                                                │
       └───────────────────────┬────────────────────────┘
                               ▼
            Total Loss: L = L_CE + λ · L_con  (λ = 0.1)
```

### 2.1 Signed Generalized Mean (GeM) Pooling

#### Standard Non-Negative GeM
Generalized Mean Pooling computes a parameterized norm over spatial positions $\Omega = \{1, \dots, H \times W\}$ for channel $c$:
$$\operatorname{GeM}(\mathbf{t}_c, p) = \left( \frac{1}{|\Omega|} \sum_{u \in \Omega} t_{c, u}^p \right)^{\frac{1}{p}}$$
- When $p = 1$, GeM is mathematically identical to Global Average Pooling (GAP).
- As $p \to \infty$, GeM approaches Global Max Pooling (GMP).
- For $p > 1$, GeM emphasizes salient, highly-activated spatial features without completely ignoring background context.

Numerically stable implementation via Log-Sum-Exp:
$$\operatorname{GeM}(\mathbf{t}_c, p) = \exp\left( \frac{\operatorname{logsumexp}(p \cdot \log(\max(\mathbf{t}_c, \epsilon))) - \ln |\Omega|}{p} \right)$$
where $\epsilon = 10^{-6}$.

#### The Signed GeM Extension for Attention Residuals
In MSENet, the tensor entering spatial pooling is the residual output of stage-wise self-attention:
$$\mathbf{y} = \gamma \cdot \operatorname{Attn}(\mathbf{x}) + \mathbf{x}$$
Because $\mathbf{x}$ stems from intermediate convolution/normalization layers and $\gamma$ can be negative or positive, $\mathbf{y}$ contains negative values (empirically ~19.7% of all activations across test batches). Directly clamping negative activations to $\epsilon$ would destroy critical inhibitory signal.

To solve this, we implement **Signed GeM Pooling** by decomposing $\mathbf{y}$ into positive and negative rectifier branches:
$$\mathbf{y}^+ = \max(\mathbf{y}, 0), \qquad \mathbf{y}^- = \max(-\mathbf{y}, 0)$$
Signed GeM computes the difference of the non-negative GeM pools:
$$\operatorname{SignedGeM}(\mathbf{y}, p) = \operatorname{GeM}(\mathbf{y}^+, p, \epsilon) - \operatorname{GeM}(\mathbf{y}^-, p, \epsilon)$$

**Properties:**
1. **Sign and Channel Preservation:** Preserves both positive activations and negative inhibitory signals without clipping.
2. **Strict GAP Equivalence at $p = 1$:**
   $$\operatorname{SignedGeM}(\mathbf{y}, 1) = \frac{1}{|\Omega|}\sum \mathbf{y}^+ - \frac{1}{|\Omega|}\sum \mathbf{y}^- = \frac{1}{|\Omega|}\sum (\mathbf{y}^+ - \mathbf{y}^-) = \frac{1}{|\Omega|}\sum \mathbf{y} = \operatorname{GAP}(\mathbf{y})$$
3. **Epsilon Cancellation:** For near-zero feature channels, the baseline $\epsilon$ offsets of the positive and negative branches cancel out.
4. **Learnable Exponents:** Each stage $s \in \{1, 2, 3, 4, 5\}$ maintains its own learnable exponent parameter $p_s$:
   $$p_s = \operatorname{Softplus}(\text{raw}_s) + p_{\text{floor}}$$
   where $p_{\text{floor}} = 0.01$, ensuring $p_s > 0$ strictly at all times. Initialized at $p_s = 3.0$ ($\text{raw}_0 \approx 2.9947$).

---

### 2.2 Auxiliary Supervised Contrastive Loss (SupCon)

To encourage robust multi-scale metric clustering, we incorporate an auxiliary Supervised Contrastive Loss (Khosla et al., 2020) over pooled support and query embeddings across all 5 stages.

For each stage $s$, pooled embeddings $\mathbf{z}^s \in \mathbb{R}^{(N_{\text{support}} + N_{\text{query}}) \times C_s}$ are $L_2$-normalized:
$$\hat{\mathbf{z}}_i^s = \frac{\mathbf{z}_i^s}{\|\mathbf{z}_i^s\|_2}$$

For anchor sample $i$ with label $y_i$, let $P(i) = \{p \neq i \mid y_p = y_i\}$ be the set of all positive indices sharing the same class label within the episode. The supervised contrastive loss for stage $s$ is:
$$\mathcal{L}_{\text{con}}^s = -\sum_{i} \frac{1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(\hat{\mathbf{z}}_i^s \cdot \hat{\mathbf{z}}_p^s / \tau)}{\sum_{a \neq i} \exp(\hat{\mathbf{z}}_i^s \cdot \hat{\mathbf{z}}_a^s / \tau)}$$

The total auxiliary contrastive loss is the average over all 5 stages:
$$\mathcal{L}_{\text{con}} = \frac{1}{5} \sum_{s=1}^5 \mathcal{L}_{\text{con}}^s$$

#### Bounded Learnable Temperature ($\tau$)
Temperature $\tau$ regulates the sensitivity of contrastive penalties. Rather than fixing $\tau$, it is parameterized as learnable:
$$\tau = \exp(\operatorname{clamp}(\log \tau_{\text{raw}}, \ln(0.01), \ln(1.0)))$$
Initialized at $\tau_0 = 0.1$.

#### Joint Multi-Task Objective
During training, the overall loss optimized via backpropagation is:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda \cdot \mathcal{L}_{\text{con}}$$
where $\lambda = 0.1$ (`CON_COEF`).

---

## 3. Architectural & Code Modifications

Compared to the base MSENet repository, all changes are strictly isolated into modular components while preserving original backbone feature extraction and stage distance calculation:

### 3.1 Pooling Replacement (`_attend_and_pool`)
- In `PrototypicalNetworks`, `_attend_and_pool` originally ran:
  ```python
  self.global_avg_pool(module(features, gamma)).flatten(1)
  ```
- In `MSENetVariant`, this is overridden to conditionally route features through `_gem_stage`:
  ```python
  def _gem_stage(self, features, stage_idx, gamma):
      module = self.attention_modules[stage_idx]
      p = self.gem_p[stage_idx]
      outs = []
      for chunk in features.split(self.attention_chunk_size, dim=0):
          a = module(chunk, gamma)  # y = gamma * attn + x
          outs.append(gem_pool(a, p))
      return torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]
  ```

### 3.2 Auxiliary Head & Parameter Groups
- **Learnable Exponents:** `gem_raw` parameter tensor of shape `(5,)`, initialized to yield $p=3.0$ via softplus. Trained with Adam at `lr = 1e-3`.
- **Learnable Contrastive Temperature:** `con_log_tau` scalar parameter, initialized to $\ln(0.1)$, trained with Adam at `lr = 1e-3`.
- **Parameter Overhead:**
  - Baseline: 11,618,120 parameters.
  - GeM only: 11,618,125 parameters (+5 parameters for 5 stage exponents).
  - Contrastive only: 11,618,121 parameters (+1 parameter for $\tau$).
  - GeM + Contrastive: 11,618,126 parameters (+6 parameters).

---

## 4. Implementation Details & Hyperparameters

| Hyperparameter | Symbol / Variable | Value | Description |
|:---|:---|:---|:---|
| Initial GeM Exponent | `GEM_P_INIT` | 3.0 | Initial value for $p_s$ |
| GeM Epsilon | `GEM_EPS` | $10^{-6}$ | Floor for non-negative log-sum-exp |
| GeM Exponent Floor | `GEM_P_FLOOR` | 0.01 | Lower bound guarantee for $p_s$ |
| GeM Optimizer LR | `GEM_LR` | $10^{-3}$ | Dedicated Adam learning rate for $p_s$ |
| Contrastive Loss Weight | `CON_COEF` ($\lambda$) | 0.1 | Balancing weight in joint loss |
| Initial Temperature | `CON_TAU_INIT` | 0.1 | Initial value for SupCon $\tau$ |
| Temperature Bounds | `CON_TAU_MIN`, `CON_TAU_MAX` | 0.01, 1.0 | Clamping range for $\tau$ |
| Temperature Optimizer LR | `CON_TAU_LR` | $10^{-3}$ | Dedicated Adam learning rate for $\tau$ |
| Backbone Learning Rate | `backbone_lr` | $10^{-4}$ | Learning rate for ResNet-18 & Attention |
| Stage Weights LR | `weights_lr` | $5 \times 10^{-3}$ | Learning rate for stage weights & gammas |
| Initial Stage Weights | `STAGE_WEIGHTS` | (1.0, 1.1, 1.2, 1.3, 1.4) | Multi-scale stage weight priors |
| Initial Gammas | `gamma_init` | 0.2 | Self-attention scale initialization |
| Metric Function | `DISTANCE_METRIC` | Euclidean | $L_2$ prototype distance metric |
| Precision | `USE_AMP` | True | PyTorch Automatic Mixed Precision (`fp16`/`fp32`) |
| Seed | `SEED` | 42 | Global pseudo-random initialization |

---

## 5. Experimental Setup & Evaluation Methodology

### 5.1 Dataset & Task Configuration
- **Dataset:** MiniImageNet (84x84 resolution, ImageNet pre-trained ResNet-18 backbone).
- **Training Episodes:** 360 episodes per epoch, 10 epochs (3,600 total episodes).
  - Setup: 30-way 5-shot training with 15 queries per class (150 support images, 450 query images per episode).
- **Validation:** 600 validation tasks per epoch (5-way 5-shot, 15 queries/class, seed = 2024). Checkpoint with highest validation accuracy is saved for final testing.
- **Testing:** 600 episodes each for:
  - 5-way 5-shot (seed = 2025, 75 query images/episode).
  - 5-way 1-shot (seed = 2026, 75 query images/episode).
- **Benchmarking:** 100 test episodes (10 warmup, seed = 2027) measured in fp32 for latency and peak VRAM.

### 5.2 Deterministic Pairing & Exact Reproducibility
- **BLAKE2b Episode Digest Verification:** Every episode's support/query image bytes and labels are verified via a BLAKE2b cryptographic hash digest. All 4 configurations evaluate on the **exact same episodes** in identical sequence.
- **Identical Pretrained Backbone Check:** Before training, the backbone feature extractor and attention modules are cryptographically verified against a master snapshot.

---

## 6. Directory Structure

```
GeM_ContrastiveLoss/
├── README.md                                # This document (theoretical framing & setup)
├── RESULTS.md                               # Complete experimental analysis & metrics
├── Source Code.ipynb                        # Executable Jupyter notebook (Kaggle/Colab)
└── results/
    ├── final_parameters.json                # Complete parameter snapshots & activation diagnostics
    ├── run_info.json                        # Runtime environment, hardware & hyperparameter metadata
    ├── comparison_tables/
    │   ├── comparison_summary.csv           # Master summary of accuracy & efficiency metrics
    │   ├── test_accuracy.csv                # Detailed 1-shot and 5-shot test metrics with 95% CIs
    │   ├── efficiency_metrics.csv           # Latency, memory, FLOPs, and parameter counts
    │   ├── paired_vs_baseline.csv           # Paired difference t-statistics vs. baseline
    │   ├── per_episode_test_accuracy.csv    # Episode-level test accuracies (600 episodes)
    │   ├── final_parameters.csv             # Checkpoint and epoch 10 parameter values
    │   ├── parameter_trajectories.csv       # Step-by-step evolution of weights, gammas, p, and tau
    │   ├── epoch_metrics_all.csv            # Combined epoch-by-epoch training/validation logs
    │   ├── epoch_metrics_baseline.csv       # Baseline training progression
    │   ├── epoch_metrics_gem.csv            # GeM configuration training progression
    │   ├── epoch_metrics_contrastive.csv    # Contrastive configuration training progression
    │   └── epoch_metrics_gem_contrastive.csv# Joint GeM + Contrastive training progression
    └── plots/
        ├── plot5_test_accuracy_bar.png      # Primary test accuracy comparison bar plot
        ├── plot6_test_accuracy_ci.png       # Test accuracy with 95% confidence intervals
        ├── plot1_training_time.png          # Total and per-epoch training wall-clock time
        ├── plot2_inference_time.png         # Inference latency per episode and per query
        ├── plot3_gpu_memory.png             # GPU peak allocated and reserved memory
        ├── plot4_parameter_trajectories.png # Trajectories of p_s, tau, stage weights, and gammas
        ├── extra_learning_curves.png        # Training loss, validation accuracy curves
        └── extra_paired_difference_vs_baseline.png # Paired episode difference distributions
```

---

## 7. Reproducibility Guide

To reproduce these experiments:
1. Open [`Source Code.ipynb`](../DL Lab/few-shot-image-class/GeM_ContrastiveLoss/Source%20Code.ipynb) on Kaggle or a GPU workstation with Python 3.10+ and PyTorch 2.0+ (CUDA enabled).
2. Set `USE_IMAGENET_PRETRAINED = True`, `DISTANCE_METRIC = 'euclidean'`, and run all cells.
3. The code will execute all four runs sequentially under fixed seeds (`SEED=42`, `VAL_SEED=2024`, `TEST_SEED_5SHOT=2025`, `TEST_SEED_1SHOT=2026`, `BENCH_SEED=2027`) and output identical comparison tables and figures to `results/`.
