# DL Lab Project - Changes Summary

---
Summary of changes made by Ashwin.

Notebook 1 : Attempted adjacent and bidirectional cross attention  

Notebook 2 : Attempted LayerNorm, Learnable Temperature both separately first, and then combined  

Notebook 3 : Attempted GeM, Contrastive Loss both separately first, and then combined  


# Cross-Attention in MSENet

Cross-attention allows one feature representation to selectively use information from another. Unlike self-attention, where queries, keys, and values come from the same feature map, cross-attention takes the query from one stage and the keys and values from a different stage. This enables information exchange between ResNet stages that capture features at different levels of abstraction.

For a query feature map $X$ and a source feature map $Y$, the operation is:

$$
Q = W_Q X,\qquad K = W_K Y,\qquad V = W_V Y
$$

$$
\operatorname{CA}(X,Y) =
\operatorname{softmax}\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
$$

Here, $W_Q$, $W_K$, and $W_V$ are learnable projections, and $d_k$ is the key dimension. In practice, feature maps from different stages are projected to compatible dimensions and their spatial positions are aligned as needed. The attention output is added to the original query features through a residual connection.

## 1. Adjacent Cross-Attention

Adjacent cross-attention performs one-way information exchange between each consecutive pair of stages. The shallower stage supplies the queries, while the immediately deeper stage supplies the keys and values. For example, Stage 1 attends to Stage 2, Stage 2 attends to Stage 3, and so on.

$$
X'_i = X_i + \operatorname{CA}(X_i, X_{i+1})
$$

This lets shallower stages incorporate relevant higher-level semantic information while retaining their original low-level features. The deepest stage has no deeper adjacent stage to attend to.

## 2. Bidirectional Cross-Attention

Bidirectional cross-attention performs the exchange in both directions for every adjacent pair. The shallower stage attends to the deeper stage, and the deeper stage also attends to the shallower stage.

$$
X'_i = X_i + \operatorname{CA}(X_i, X_{i+1})
$$

$$
X'_{i+1} = X_{i+1} + \operatorname{CA}(X_{i+1}, X_i)
$$

This allows both stages to benefit from each other's information: shallow features can gain semantic context, while deeper features can access lower-level detail. It offers richer cross-scale interaction than the one-way adjacent variant, at the cost of additional computation.

In both variants, cross-attention is applied after the existing stage-wise self-attention and before global average pooling and prototype computation. The resulting stage features are then used by the existing Euclidean-distance and weighted multi-stage classification pipeline.
## 3. LayerNorm (Layer Normalization)

LayerNorm is introduced after Global Average Pooling (GAP) at each of the five ResNet-18 feature stages, before prototype computation and distance calculation.

It normalizes the feature embedding across its channel dimensions, ensuring that features have a consistent scale and distribution.

For an embedding $x$:

$$
\hat{x} = \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}
$$

$$
y = \gamma\hat{x}+\beta
$$

Here, $\mu$ and $\sigma^2$ represent the mean and variance of the embedding, while $\gamma$ and $\beta$ are learnable scaling and shifting parameters.

**Purpose:** To reduce variations in feature scales across channels, potentially improving feature consistency and making Euclidean distance comparisons more reliable. Separate LayerNorm modules are applied to each stage.

## 4. Learnable Temperature Scaling

The original MSENet calculates weighted Euclidean distances across its five feature stages and directly converts the negative distances into classification probabilities using Softmax.

We introduce a learnable temperature parameter $T$, initialized to 1.0, to control the sharpness of the resulting probability distribution.

**Original:**

$$
P(y=j\mid x)=\operatorname{Softmax}(-d_j)
$$

**Modified:**

$$
P(y=j\mid x)=
\operatorname{Softmax}\left(\frac{-d_j}{T}\right)
$$

To ensure positivity, temperature is parameterized using Softplus:

$$
T=\operatorname{Softplus}(t)
$$

where $t$ is a learnable parameter.

**Purpose:** Temperature controls how strongly the model distinguishes between class distances. Lower temperatures produce sharper probability distributions, while higher temperatures produce softer distributions. The model learns the appropriate temperature during training.

## 5. Generalized Mean (GeM) Pooling

GeM pooling is an alternative to Global Average Pooling (GAP). Instead of simply averaging spatial feature values, it uses a power parameter `p` to control how strongly larger activations influence the pooled representation.

$$
\mathrm{GeM}(x_c) = \left( \frac{1}{|\Omega|} \sum_{u \in \Omega} x_{c,u}^{p} \right)^{\frac{1}{p}}
$$


Where:
- `x_c,u` is the activation at spatial position `u` in channel `c`.
- `|Ω|` is the total number of spatial positions.
- `p` controls the pooling behaviour and can be learnable.

When `p = 1`, GeM behaves like average pooling. As `p` increases, it gives greater importance to stronger activations, approaching max pooling as `p` → ∞.

**Purpose:** GeM allows the model to learn how much to emphasize prominent spatial features instead of relying only on their average. This may produce more discriminative feature representations for few-shot classification.

## 6. Contrastive Loss

Contrastive Loss is a metric-learning loss that trains a model to produce similar embeddings for samples belonging to the same class and dissimilar embeddings for samples from different classes.

It works with pairs of samples:
- **Positive pair:** Both samples belong to the same class.
- **Negative pair:** Samples belong to different classes.

**Formula:**

L = y × d² + (1 − y) × max(0, m − d)²

Where:
- `y = 1` for a positive pair and `y = 0` for a negative pair.
- `d` is the distance between the two sample embeddings.
- `m` is the margin that controls the minimum desired separation between negative pairs.

**Purpose:** The loss pulls same-class embeddings closer together and pushes different-class embeddings apart, provided their distance is below the margin. This encourages a more discriminative embedding space, potentially helping few-shot classification distinguish classes using only a few labelled examples.

---
# RESULTS : 

### Notebook 1 - (baseline - 83.06)
### Adjacent Cross Attention - 83.25
### Bidirectional Cross Attention - 83.06  
---

### Notebook 2 - (baseline - 82.48)
### LayerNorm - 83.06
### Learnable Temperature - 82.47
### LayerNorm + Learnable Temperature - 83.18  
---

### Notebook 3 - (baseline - 82.24)
### GeM - 83.06
### Contrastive Loss - 82.26
### GeM + Contrastive Loss - 82.68 
---