# MSENet: Multi-Scale Embedding Network (Baseline Model)

## 1. Overview & Research Paper Foundation

This directory houses the clean reference implementation of the canonical **MSENet** (Multi-Scale Embedding Network) for few-shot image classification, directly based on the research paper:

> **"Enhancing Few-Shot Image Classification through Learnable Multi-Scale Embedding and Attention Mechanisms"**  
> *Fatemeh Askari, Amirreza Fateh, Mohammad Reza Mohammadi* (GitHub: [FatemehAskari/MSENet](https://github.com/FatemehAskari/MSENet))

The purpose of this standalone folder is to establish a rigorous, reproducible benchmark for the unperturbed baseline against which all subsequent novelties and architectural extensions in this repository are evaluated.

---

## 2. Baseline Architecture & Mechanics

Standard metric-based few-shot learners (such as Prototypical Networks) map images into a single embedding space using the final layer of a backbone network, discarding intermediate features. MSENet addresses this limitation by tapping features at multiple depths of a ResNet-18 backbone.

```
                       Input Image (84 x 84)
                                │
                                ▼
                       ResNet-18 Backbone
                                │
        ┌──────────────┬────────┼──────────────┬──────────────┐
        ▼              ▼        ▼              ▼              ▼
     Stage 0        Stage 1   Stage 2       Stage 3        Stage 4
    (maxpool)      (layer1)   (layer2)      (layer3)       (layer4)
    64x21x21       64x21x21   128x11x11     256x6x6        512x3x3
        │              │        │              │              │
        ▼              ▼        ▼              ▼              ▼
    Self-Attn      Self-Attn  Self-Attn     Self-Attn      Self-Attn
   (gamma=0.2)    (gamma=0.2)(gamma=0.2)   (gamma=0.2)    (gamma=0.2)
        │              │        │              │              │
        ▼              ▼        ▼              ▼              ▼
       GAP            GAP      GAP            GAP            GAP
      (64)           (64)     (128)          (256)          (512)
        │              │        │              │              │
        └──────────────┴────────┼──────────────┴──────────────┘
                                ▼
               Support Prototype & Query Distance:
               d_p(z_q, c_j) = || z_q^p - c_j^p ||_2 (Euclidean)
                                │
                                ▼
                  Learnable Stage Weights (w_p):
                  Logits = - Sum_p [ w_p * d_p(z_q, c_j) ]
                                │
                                ▼
                        Cross-Entropy Loss
```

### 2.1 Multi-Scale Feature Extraction
Features are extracted at five discrete transition stages:
- **Stage 0 (`maxpool`):** Channels $C_0 = 64$, resolution $21 \times 21$ (edges, low-level textures).
- **Stage 1 (`layer1`):** Channels $C_1 = 64$, resolution $21 \times 21$ (contours, basic shape primitives).
- **Stage 2 (`layer2`):** Channels $C_2 = 128$, resolution $11 \times 11$ (localized visual patterns).
- **Stage 3 (`layer3`):** Channels $C_3 = 256$, resolution $6 \times 6$ (part-level representations).
- **Stage 4 (`layer4`):** Channels $C_4 = 512$, resolution $3 \times 3$ (high-level semantic abstractions).

### 2.2 Stage-Wise Self-Attention (SA)
At each stage $p \in \{0, 1, 2, 3, 4\}$, spatial features are refined via a non-local self-attention module:
1. $1 \times 1$ convolutions project the feature map into query ($q$), key ($k$), and value ($v$) representations:
   $$q = W_q x, \quad k = W_k x, \quad v = W_v x$$
   where $W_q, W_k \in \mathbb{R}^{\frac{C_p}{8} \times C_p}$ reduce dimensionality for efficient attention computation.
2. Spatial attention weights $\beta \in \mathbb{R}^{N \times N}$ ($N = H_p \cdot W_p$) capture pairwise spatial correlations:
   $$\beta = \operatorname{Softmax}\left(q \cdot k^\top\right)$$
3. The attention output is scaled by parameter $\gamma$ and added back through a residual skip connection:
   $$y = \gamma \cdot \left(v \cdot \beta^\top\right) + x$$
   In the baseline, $\gamma$ is held fixed at $\gamma_1 = \gamma_2 = 0.2$ for both support and query inputs.

### 2.3 Global Average Pooling (GAP) & Prototype Computation
Spatial feature maps are pooled across spatial dimensions into 1D vectors:
$$z^p = \operatorname{GAP}(y^p) \in \mathbb{R}^{C_p}$$

Given support set $S_j$ for class $j$, the stage prototype $c_j^p$ is the arithmetic mean:
$$c_j^p = \frac{1}{|S_j|} \sum_{x \in S_j} z^p(x)$$

### 2.4 Euclidean Distance Metric ($L_2$)
The baseline model measures sample-to-prototype similarity strictly using Euclidean distance:
$$d_{i,j}^p = \left\| z_i^{q,p} - c_j^p \right\|_2 = \sqrt{\sum_{k=1}^{C_p} \left(z_{i,k}^{q,p} - c_{j,k}^p\right)^2}$$

### 2.5 Learnable Global Stage Weights
Stage distances are weighted using a learnable parameter vector $w = [w_0, w_1, w_2, w_3, w_4]$ (initialized to $[1.0, 1.1, 1.2, 1.3, 1.4]$):
$$\text{Logit}(x_i^q, j) = -\sum_{p=0}^4 w_p \cdot d_{i,j}^p$$

Predictions are normalized with Softmax, and parameters are optimized via standard Cross-Entropy Loss:
$$\mathcal{L}_{\text{CE}} = -\sum_{i} \log \frac{\exp\left(\text{Logit}(x_i^q, y_i)\right)}{\sum_j \exp\left(\text{Logit}(x_i^q, j)\right)}$$

---

## 3. Inherent Limitations & Problems with the Baseline

Through empirical analysis of the baseline training dynamics, we identified several core deficiencies:

1. **Dimensionality & Magnitude Imbalance in Euclidean Metric:**
   - Euclidean distance computes squared differences summed across all channels. Because Stage 4 has 512 channels while Stage 0 has only 64, distance values across stages naturally vary by an order of magnitude.
   - Euclidean distance is sensitive to vector norms and feature scaling rather than directional alignment, causing larger feature scales to disproportionately influence gradient updates.

2. **Rigidity of Static Global Stage Weights:**
   - The stage weight vector $w$ is global and static across all evaluation tasks. However, few-shot tasks vary dramatically in visual granularity: fine-grained tasks (e.g., dog breeds) need low-level spatial texture cues, whereas coarse-grained tasks (e.g., vehicles vs. mammals) rely on deep semantic abstractions. A single static weight vector cannot adapt dynamically.

3. **Overfitting & Validation Drift in Episodic Training:**
   - With small support sets (e.g., 5 examples per class), standard Batch Normalization statistics drift significantly during episodic training.
   - Without regularization (such as EMA, LayerNorm, or label smoothing), validation accuracy peaks early (Epoch 1) and degrades over subsequent training epochs despite train loss continuing to drop.

4. **Uniform Spatial Averaging (GAP Information Loss):**
   - Standard Global Average Pooling treats all spatial positions uniformly, diluting compact, discriminative object parts with background clutter.

---

## 4. Execution Details

The baseline notebook [`msenet-base1-final.ipynb`](file:///c:/Users/us/Downloads/few-shot-image-class/msenet_baseline/msenet-base1-final.ipynb) runs the complete training pipeline:
- **Dataset:** MiniImageNet ($84 \times 84$, 64 train, 16 val, 20 test classes)
- **Episodes:** 360 train episodes/epoch (30-way 5-shot), 600 validation episodes/epoch (5-way 5-shot), 600 test episodes (5-way 5-shot)
- **Optimizer:** Adam (backbone lr: $10^{-4}$, stage weights lr: $5 \times 10^{-3}$)
- **Checkpointing:** Validated after every epoch; best validation checkpoint loaded for final evaluation on held-out test classes.

For complete numerical outputs, epoch progression, and final parameter values from this run, see [`RESULTS.md`](file:///c:/Users/us/Downloads/few-shot-image-class/msenet_baseline/RESULTS.md).
