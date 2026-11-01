# DL Lab Project - Changes Summary (Set 2)

### A1 - Freeze Batch Normalization Statistics

**What it does:**  
Sets all Batch Normalization (`BatchNorm2d`) layers to evaluation mode after every `model.train()` call. This prevents the running mean and variance from being updated during training. However, the learnable affine parameters (gamma and beta) remain trainable.

**Purpose:**  
Prevents changes in batch statistics from destabilizing pretrained feature representations, particularly when training with small few-shot batches.

**Formula:**

$$
\hat{x} = \frac{x - \mu_{\text{pretrained}}}
{\sqrt{\sigma^2_{\text{pretrained}} + \epsilon}}
$$

The pretrained running statistics remain fixed, while gamma and beta can still be optimized.

---

### A2 - Differential Learning Rates

**What it does:**  
Assigns different learning rates to different parts of the network:

- Stem and Layer 1: Frozen.
- Layer 2: 1e-6.
- Layer 3: 3e-6.
- Layer 4: 1e-5.
- Attention heads and special parameters: 5e-3.

**Purpose:**  
Allows newly initialized attention modules to learn faster while making smaller updates to pretrained backbone layers. This reduces the risk of destroying useful pretrained representations.

**Formula:**

$$
\theta_i^{(t+1)} =
\theta_i^{(t)} - \eta_i \nabla_{\theta_i} L
$$

Here, each parameter group has its own learning rate, denoted by eta.

---

### A3 - L2-SP Regularization

**What it does:**  
Adds a regularization term that penalizes changes in the backbone parameters relative to their pretrained values. The regularization coefficient is set to 1e-3.

**Purpose:**  
Encourages the model to retain useful knowledge from ImageNet pretraining instead of drifting too far from its original parameters during few-shot training.

**Formula:**

$$
L_{\text{total}} =
L_{\text{task}} +
\frac{\alpha}{2}
\left\| \theta_{\text{backbone}} -
\theta_{\text{pretrained}} \right\|_2^2
$$

Where:
- `L_task` is the original training loss.
- `alpha` is the regularization coefficient (1e-3).
- `theta_backbone` represents the current backbone parameters.
- `theta_pretrained` represents the original pretrained parameters.

---

### A4 - Exponential Moving Average (EMA)

**What it does:**  
Maintains a moving average of the model's state after every optimizer update, using a decay factor of 0.99. The averaged model state is used during validation and testing.

**Purpose:**  
Reduces fluctuations in model parameters during training and produces more stable predictions by combining information from multiple training steps.

**Formula:**

$$
\theta_{\text{EMA}}^{(t)} =
\beta \theta_{\text{EMA}}^{(t-1)}
+ (1-\beta)\theta^{(t)}
$$

Where:
- `beta` = 0.99.
- `theta` represents the current model state.
- `theta_EMA` represents the moving-averaged model state.

The implementation also maintains EMA values for floating-point Batch Normalization statistics.

---

### A5 - Label Smoothing

**What it does:**  
Uses Cross Entropy Loss with a label smoothing factor of 0.1 instead of assigning the entire target probability to the correct class.

**Purpose:**  
Reduces overconfidence by distributing a small portion of the target probability across incorrect classes. This encourages better generalization, especially when training data is limited.

**Formula:**

For a classification task with K classes:

$$
q_i =
\begin{cases}
1-\epsilon+\frac{\epsilon}{K}, & \text{if } i=y \\
\frac{\epsilon}{K}, & \text{otherwise}
\end{cases}
$$

The resulting cross-entropy loss is:

$$
L = -\sum_{i=1}^{K} q_i \log(p_i)
$$

Where:
- `epsilon` = 0.1.
- `y` is the correct class.
- `q_i` is the smoothed target probability.
- `p_i` is the model's predicted probability.

---
## Results :

### Baseline - 82.24
### A1 - 82.33
### A2 - 78.89
### A3 - 82.24
### A4 - 84.12
### A5 - 82.52