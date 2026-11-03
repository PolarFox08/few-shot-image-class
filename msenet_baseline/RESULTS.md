# Experimental Results: MSENet Baseline (Euclidean Distance)

This document details the quantitative outcomes obtained from training and evaluating the standalone canonical **MSENet baseline** in [`msenet-base1-final.ipynb`](file:///c:/Users/us/Downloads/few-shot-image-class/msenet_baseline/msenet-base1-final.ipynb) on the **MiniImageNet** dataset (ResNet-18 backbone, 5-way 5-shot evaluation over 600 fixed test episodes).

---

## 1. Executive Summary: Baseline Performance Metrics

| Metric | Measured Baseline Value | Configuration / Description |
|:---|:---:|:---|
| **Backbone Architecture** | ResNet-18 (ImageNet-pretrained) | 5 feature stages (`maxpool`, `layer1`, `layer2`, `layer3`, `layer4`) |
| **Distance Metric** | Euclidean ($L_2$) | Pairwise unnormalized distance: $d(z_q, c_j) = \|z_q - c_j\|_2$ |
| **Stage Weighting Scheme** | Static Learnable ($w$) | Global weight vector $w \in \mathbb{R}^5$ optimized by Adam ($\text{lr}=5 \times 10^{-3}$) |
| **Attention Residual Parameter ($\gamma$)** | Fixed ($\gamma_1 = \gamma_2 = 0.2$) | Non-trainable scalar across support and query sets |
| **Best Validation Accuracy** | **81.25%** | Reached at **Epoch 1** |
| **Final Test Accuracy (5-way 5-shot)** | **82.23%** | Evaluated on 600 unseen test tasks |
| **Initial Training Loss** | 2.1115 | Cross-Entropy loss at Epoch 1 |
| **Final Training Loss** | 0.0837 | Converged Cross-Entropy loss at Epoch 10 |
| **Initial Stage Weights** | `[1.0, 1.1, 1.2, 1.3, 1.4]` | Monotonic increasing initialization |
| **Final Learned Stage Weights** | `[0.2984, 1.1780, 1.5044, 3.2669, 0.9636]` | Stage 3 (`layer3`) receives highest weighting |
| **Average Epoch Time** | ~8.6 – 8.7 min | Total training time: ~86.5 min (10 epochs on Tesla T4) |

---

## 2. Full Epoch-by-Epoch Training & Validation Progression

The training loop was executed for 10 epochs (360 training episodes/epoch at 30-way 5-shot; 600 validation episodes/epoch at 5-way 5-shot). Validation accuracy was measured after each epoch, and the best-performing state was saved.

| Epoch | Training Loss (Cross-Entropy) | Validation Accuracy (%) | Epoch Runtime (min) | Checkpoint Status |
|:---:|:---:|:---:|:---:|:---:|
| **1** | **2.1115** | **81.25%** | **8.6 min** | **Saved Best Checkpoint (`best_model_5shot.pth`)** |
| 2 | 1.3785 | 81.05% | 8.7 min | |
| 3 | 0.8940 | 79.43% | 8.6 min | |
| 4 | 0.5800 | 77.72% | 8.7 min | |
| 5 | 0.3836 | 75.52% | 8.7 min | |
| 6 | 0.2476 | 76.47% | 8.6 min | |
| 7 | 0.1816 | 75.39% | 8.7 min | |
| 8 | 0.1338 | 76.29% | 8.6 min | |
| 9 | 0.1029 | 75.47% | 8.7 min | |
| 10 | 0.0837 | 74.42% | 8.6 min | |

### Empirical Training Dynamics
- **Rapid Initial Convergence:** The model converges quickly on the training distribution, with cross-entropy loss dropping from $2.1115$ in Epoch 1 to $0.0837$ by Epoch 10.
- **Validation Generalization Peak:** Generalization to validation classes peaks immediately at Epoch 1 (**81.25%**). From Epoch 2 onwards, validation accuracy gradually declines, reflecting the tendency of unregularized episodic few-shot training to overfit to the 64 training classes.
- **Evaluation Checkpoint:** As per standard protocol, the Epoch 1 checkpoint was loaded to evaluate performance on the held-out test split, yielding **82.23%**.

---

## 3. Stage Weight Allocation & Parameter Analysis

### 3.1 Learned Stage Weights Breakdown

| Stage Index | Architecture Tap | Channel Dimension | Spatial Map Size | Initial Weight | Final Learned Weight | Relative Share (%) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **Stage 0** | `maxpool` | 64 | $21 \times 21$ | 1.0000 | **0.2984** | 4.14% |
| **Stage 1** | `layer1` | 64 | $21 \times 21$ | 1.1000 | **1.1780** | 16.34% |
| **Stage 2** | `layer2` | 128 | $11 \times 11$ | 1.2000 | **1.5044** | 20.86% |
| **Stage 3** | `layer3` | 256 | $6 \times 6$ | 1.3000 | **3.2669** | **45.30%** |
| **Stage 4** | `layer4` | 512 | $3 \times 3$ | 1.4000 | **0.9636** | 13.36% |
| **Total Sum** | | | | 6.0000 | **7.2113** | 100.00% |

### 3.2 Key Observations from Learned Parameters
1. **Dominance of Intermediate-High Features (Stage 3):** Under Euclidean distance, Stage 3 (`layer3`) emerges as the primary decision driver, accounting for nearly half ($45.30\%$) of the cumulative distance weight ($3.2669$).
2. **Attenuation of Shallow Stage (Stage 0):** Stage 0 (`maxpool`) is down-scaled from $1.0$ to $0.2984$, demonstrating that while low-level edge features provide supplementary regularization, they are insufficient on their own for category differentiation.
3. **Stage 4 vs. Stage 3 Interplay:** While Stage 4 (`layer4`) possesses the highest semantic abstraction, its 512-channel vector produces larger raw Euclidean distances; the network scales Stage 3 to a higher weight ($3.2669$ vs. $0.9636$) to balance sensitivity across feature scales.
4. **Attention Scaling Parameter:** Both attention parameters remain fixed at their initialization:
   - $\gamma_1$ (Support branch): **0.2000**
   - $\gamma_2$ (Query branch): **0.2000**
