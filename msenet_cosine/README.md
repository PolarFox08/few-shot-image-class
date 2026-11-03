# MSENet with Cosine and Hybrid Distance Metrics

## 1. Background: Original MSENet Paper

This work builds upon and improves the method proposed in:
> **"Enhancing Few-Shot Image Classification through Learnable Multi-Scale Embedding and Attention Mechanisms"**  
> *Fatemeh Askari, Amirreza Fateh, Mohammad Reza Mohammadi* (GitHub: [FatemehAskari/MSENet](https://github.com/FatemehAskari/MSENet))

### Summary of Original Architecture
Traditional metric-based few-shot learning methods (such as Prototypical Networks) rely on a single feature embedding extracted from the final layer of a backbone, typically comparing query features to class prototypes using standard Euclidean distance. This discards intermediate representations containing rich low-level spatial and texture details.

To address this, **MSENet** introduced:
1. **Multi-Scale Feature Extraction:** Extracts feature representations across all five convolutional stages of ResNet-18:
   - Stage 0: `maxpool` ($64 \times 21 \times 21$)
   - Stage 1: `layer1` ($64 \times 21 \times 21$)
   - Stage 2: `layer2` ($128 \times 11 \times 11$)
   - Stage 3: `layer3` ($256 \times 6 \times 6$)
   - Stage 4: `layer4` ($512 \times 3 \times 3$)
2. **Stage-Wise Self-Attention (SA):** Refines spatial features at each stage using self-attention scaled by parameter $\gamma$:
   $$y_{\text{Conv}-p} = \gamma \cdot \text{Attention}(f_{\text{Conv}-p}) + f_{\text{Conv}-p}$$
3. **Global Average Pooling (GAP):** Pools attended spatial feature maps into 1D vectors $z \in \mathbb{R}^{C_p}$.
4. **Prototypes:** Support representations are averaged per class $j$:
   $$c_j^p = \frac{1}{|S_j|} \sum_{x \in S_j} z^p(x)$$
5. **Euclidean Distance:** Original MSENet computes the $L_2$ distance at each stage:
   $$d_{i,j}^p = \|z_i^{q,p} - c_j^p\|_2$$
6. **Learnable Stage Weights:** Combines stage distances using trainable weights $w = [w_0, w_1, w_2, w_3, w_4]$:
   $$\text{Score}(x_i^q, j) = -\sum_{p=0}^4 w_p \cdot d_{i,j}^p$$
   The true class negative log-probability is minimized via Cross-Entropy loss.

---

## 2. Motivation & What This Folder Adds

While Euclidean distance works well in low-dimensional isotropic spaces, it presents notable limitations in deep multi-scale architectures:
- **Curse of Dimensionality & Magnitude Sensitivity:** Euclidean distance is strongly sensitive to feature vector magnitudes and varying channel dimensions ($64$ vs $512$). High-magnitude features or outlier activations can dominate distance calculations.
- **Hyperspherical Geometry:** In deep metric learning, direction and angular orientation on a normalized sphere often capture semantic category boundaries far more robustly than absolute vector lengths.

To investigate and alleviate these issues, the implementations in `msenet_cosine/` introduce two major distance innovations and systematic ablation of the attention residual parameter $\gamma$:

### Novelty A: Cosine Distance Metric (`distance_metric='cosine'`)
Features and class prototypes are projectively mapped onto the unit hypersphere via $L_2$ normalization:
$$\hat{z}_i^q = \frac{z_i^q}{\|z_i^q\|_2}, \qquad \hat{c}_j = \frac{c_j}{\|c_j\|_2}$$

Cosine similarity between the query and prototype is:
$$\operatorname{Sim}_{\cos}(\hat{z}_i^q, \hat{c}_j) = \hat{z}_i^q \cdot \hat{c}_j^\top$$

The corresponding bounded distance $[0, 2]$ is scaled by a learnable temperature parameter $T_{\cos}$ (`cosine_temp`, initialized to $10.0$):
$$d_{\cos}(z_i^q, c_j) = T_{\cos} \cdot \left(1 - \operatorname{Sim}_{\cos}(\hat{z}_i^q, \hat{c}_j)\right)$$

> **Why Temperature Scaling?** Standard cosine distances fall into $[0, 2]$. Directly passing negative distances into Softmax produces extremely soft probability distributions with underconfident cross-entropy gradients. Scaling by $T_{\cos}$ sharpens the logits and allows gradient descent to regulate prediction confidence.

---

### Novelty B: Hybrid Distance Metric (`distance_metric='hybrid'`)
Combines the complementary properties of both Euclidean distance (magnitude & cluster separation) and Cosine distance (angular alignment):
$$d_{\text{hybrid}}(z_i^q, c_j) = 0.5 \cdot d_{\text{euclidean}}(z_i^q, c_j) + 0.5 \cdot d_{\cos}(z_i^q, c_j)$$
$$d_{\text{hybrid}}(z_i^q, c_j) = 0.5 \cdot \|z_i^q - c_j\|_2 + 0.5 \cdot T_{\cos} \left(1 - \frac{z_i^q}{\|z_i^q\|_2} \cdot \frac{c_j^\top}{\|c_j\|_2}\right)$$

This allows the network to benefit from angular orientation while preserving spatial magnitude cues across the multi-scale hierarchy.

---

### Novelty C: Attention Scaling Factor Dynamics (Fixed vs. Learnable $\gamma$)
The self-attention module scales non-local context by $\gamma$ before adding residual backbone features:
$$\operatorname{out} = \gamma \cdot \operatorname{Attention}(x) + x$$

In our architecture, separate parameters are maintained for support and query branches:
- $\gamma_1$: Attention scaling factor for support feature maps ($z_{\text{support}}$).
- $\gamma_2$: Attention scaling factor for query feature maps ($z_{\text{query}}$).

We evaluate two regimes:
1. **Fixed Gamma (`fixedGamma`):** $\gamma_1 = \gamma_2 = 0.2$ are kept static (`gammalearnable=False`), preserving the handcrafted residual balance.
2. **Learnable Gamma (`learnableGamma`):** $\gamma_1, \gamma_2$ are registered as `nn.Parameter` (`gammalearnable=True`) and trained with Adam ($\text{lr} = 0.005$). This enables the model to dynamically discover the optimal balance of self-attention independently for prototypes and queries.

---

## 3. Notebook Files Overview

This folder contains 4 distinct experimental configurations corresponding to the $2 \times 2$ matrix of distance metric and gamma learning:

| Notebook File | Distance Metric | Attention Gamma $\gamma$ | Learnable Parameters |
|:---|:---|:---|:---|
| [`mse-withCosine_fixedGamma.ipynb`](file:///c:/Users/us/Downloads/few-shot-image-class/msenet_cosine/mse-withCosine_fixedGamma.ipynb) | Cosine Distance ($d_{\cos}$) | Fixed ($\gamma_1 = \gamma_2 = 0.2$) | Backbone, Stage Weights ($w$), $T_{\cos}$ |
| [`mse-withCosine_learnableGamma.ipynb`](file:///c:/Users/us/Downloads/few-shot-image-class/msenet_cosine/mse-withCosine_learnableGamma.ipynb) | Cosine Distance ($d_{\cos}$) | Learnable ($\gamma_1, \gamma_2$ init 0.2) | Backbone, Stage Weights ($w$), $T_{\cos}$, $\gamma_1, \gamma_2$ |
| [`mse-withHybrid_fixedGamma.ipynb`](file:///c:/Users/us/Downloads/few-shot-image-class/msenet_cosine/mse-withHybrid_fixedGamma.ipynb) | Hybrid ($0.5 \cdot d_{\text{euc}} + 0.5 \cdot d_{\cos}$) | Fixed ($\gamma_1 = \gamma_2 = 0.2$) | Backbone, Stage Weights ($w$), $T_{\cos}$ |
| [`mse-withHybrid_learnableGamma.ipynb`](file:///c:/Users/us/Downloads/few-shot-image-class/msenet_cosine/mse-withHybrid_learnableGamma.ipynb) | Hybrid ($0.5 \cdot d_{\text{euc}} + 0.5 \cdot d_{\cos}$) | Learnable ($\gamma_1, \gamma_2$ init 0.2) | Backbone, Stage Weights ($w$), $T_{\cos}$, $\gamma_1, \gamma_2$ |

---

## 4. Experimental Setup

All 4 notebooks share an identical, strictly controlled training protocol to ensure fair comparison:
- **Dataset:** MiniImageNet ($84 \times 84$ resolution, 64 training classes, 16 validation classes, 20 test classes).
- **Backbone:** ResNet-18 (ImageNet-pretrained).
- **Training Protocol:**
  - Episodes per epoch: 360 episodes.
  - Episode configuration: 30-way 5-shot training with 15 query images per class.
  - Validation: 600 tasks, 5-way 5-shot with 15 queries per task evaluated after each epoch.
  - Testing: 600 tasks, 5-way 5-shot on held-out test classes using the best validation checkpoint.
- **Optimizer:** Adam with Cosine Annealing learning rate schedule over 10 epochs ($\eta_{\min} = 10^{-6}$).
  - Backbone learning rate: $1 \times 10^{-4}$
  - Stage weights learning rate: $5 \times 10^{-3}$
  - Cosine temperature learning rate: $5 \times 10^{-3}$
  - Gamma learning rate (when learnable): $5 \times 10^{-3}$
- **Initial Stage Weights:** $[1.0, 1.1, 1.2, 1.3, 1.4]$ for Stages 0 through 4.
- **Initial Cosine Temperature ($T_{\cos}$):** $10.0$.
- **Acceleration:** Automatic Mixed Precision (`torch.cuda.amp.autocast`) on NVIDIA Tesla T4.

For detailed quantitative results, epoch-wise loss/accuracy tables, learned parameter values, and comparative analysis, see [`RESULTS.md`](file:///c:/Users/us/Downloads/few-shot-image-class/msenet_cosine/RESULTS.md).
