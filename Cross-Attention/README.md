# Multi-Scale Stage-to-Stage Cross-Attention for MSENet

## 1. Motivation: Enhancing Inter-Scale Information Flow

In the original MSENet paper:
> **"Enhancing Few-Shot Image Classification through Learnable Multi-Scale Embedding and Attention Mechanisms"**  
> *Fatemeh Askari, Amirreza Fateh, Mohammad Reza Mohammadi* (GitHub: [FatemehAskari/MSENet](https://github.com/FatemehAskari/MSENet))

The multi-scale architecture extracts feature representations from five stages of a ResNet-18 backbone (`maxpool`, `layer1`, `layer2`, `layer3`, `layer4`). Each stage is processed through an independent intra-stage spatial self-attention module before being globally average-pooled into a flat representation vector:

$$\mathbf{F}_p' = \mathbf{F}_p + \gamma \cdot \operatorname{SelfAttn}(\mathbf{F}_p), \quad p \in \{0, 1, 2, 3, 4\}$$

After pooling, stage-specific prototype distances are computed and linearly combined using learnable stage weights $w = [w_0, w_1, w_2, w_3, w_4]$:

$$\operatorname{Score}(x_i^q, j) = -\sum_{p=0}^4 w_p \cdot d_{i, j}^p$$

### The Limitation of Independent Stage Processing
While MSENet captures hierarchical representations from shallow edges to deep semantics, **the stages evolve completely in isolation** until the final score aggregation:
1. **Isolated Stage Refinement:** Intra-stage self-attention refines spatial positions only within the same scale. Finer-grained stages (`maxpool`, `layer1`) lack the high-level semantic context needed to resolve spatial ambiguities.
2. **Missing Inter-Scale Communication:** Coarser-grained semantic stages (`layer3`, `layer4`) capture global categorical context but suffer from downsampled spatial resolution ($6 \times 6$ and $3 \times 3$). They cannot reference high-resolution edge/texture boundaries present in shallower feature maps.
3. **Late-Only Stage Fusion:** Because information fusion is deferred entirely to scalar-weighted prototype distance combination at the output, the intermediate representations cannot mutually disambiguate or contextualize one another during feature extraction.

---

## 2. The Novelty: Multi-Scale Stage Cross-Attention

This directory investigates whether introducing **direct token-to-token cross-attention between consecutive multi-scale stages** improves few-shot image classification performance on MiniImageNet.

```
                      Input Feature Maps from ResNet-18 Backbone
                Stage 1 (maxpool)   Stage 2 (layer1)   Stage 3 (layer2)   Stage 4 (layer3)   Stage 5 (layer4)
                  [64, 21, 21]       [64, 21, 21]      [128, 11, 11]       [256, 6, 6]        [512, 3, 3]
                        │                  │                  │                  │                  │
                        ▼                  ▼                  ▼                  ▼                  ▼
                 Self-Attention     Self-Attention     Self-Attention     Self-Attention     Self-Attention
                        │                  │                  │                  │                  │
                        ├──────────────────┴──────────────────┴──────────────────┴──────────────────┤
                        │                  Baseline MSENet: No Cross-Attention                     │
                        │                  Adjacent CA: Shallow queries Deeper (S_i -> S_{i+1})      │
                        │                  Bidirectional CA: Both forward and reverse links         │
                        ▼                                                                           ▼
           ┌────────────────────────┐                                                 ┌────────────────────────┐
           │ Adjacent Cross-Attn    │                                                 │ Bidirectional Cross-At │
           │ S1 -> S2 (441 -> 441)  │                                                 │ S1 <-> S2 (441 <-> 441)│
           │ S2 -> S3 (441 -> 121)  │                                                 │ S2 <-> S3 (441 <-> 121)│
           │ S3 -> S4 (121 -> 36)   │                                                 │ S3 <-> S4 (121 <-> 36) │
           │ S4 -> S5 (36 -> 9)     │                                                 │ S4 <-> S5 (36 <-> 9)   │
           └───────────┬────────────┘                                                 └───────────┬────────────┘
                       │                                                                          │
                       └────────────────────────────────────┬─────────────────────────────────────┘
                                                            ▼
                                              Global Average Pooling (GAP)
                                                            ▼
                                           Multi-Scale Prototype Distances
                                                            ▼
                                            Linear Stage Weight Fusion:
                                               Score = - Σ w_p * d_p
```

### Architectural Formulation

Cross-attention modules are positioned **immediately after the baseline spatial self-attention and immediately before Global Average Pooling (GAP)**. 

#### 1. Scaled Dot-Product Cross-Attention (`CrossAttn`)
Given a query feature map $\mathbf{X}_q \in \mathbb{R}^{B \times C_q \times H_q \times W_q}$ and key/value feature map $\mathbf{X}_{kv} \in \mathbb{R}^{B \times C_{kv} \times H_k \times W_k}$:
- Number of query tokens: $N_q = H_q \cdot W_q$
- Number of key/value tokens: $N_k = H_k \cdot W_k$
- Bottleneck projection dimension: $d = \lfloor C_q / 8 \rfloor$ (matching MSENet's self-attention channel convention)

Linear projections are implemented via $1 \times 1$ 2D convolutions:
$$\mathbf{Q} = \operatorname{Conv1x1}_{C_q \to d}(\mathbf{X}_q) \in \mathbb{R}^{B \times d \times N_q} \xrightarrow{\text{transpose}} \mathbb{R}^{B \times N_q \times d}$$
$$\mathbf{K} = \operatorname{Conv1x1}_{C_{kv} \to d}(\mathbf{X}_{kv}) \in \mathbb{R}^{B \times d \times N_k}$$
$$\mathbf{V} = \operatorname{Conv1x1}_{C_{kv} \to C_q}(\mathbf{X}_{kv}) \in \mathbb{R}^{B \times C_q \times N_k}$$

The attention weight matrix $\mathbf{A} \in \mathbb{R}^{B \times N_q \times N_k}$ is calculated via scaled dot-product:
$$\mathbf{A} = \operatorname{Softmax}\left(\frac{\mathbf{Q} \mathbf{K}}{\sqrt{d}}\right)$$

The cross-attention update is projected onto the query spatial grid and modulated by a learnable scalar gate $\gamma$:
$$\operatorname{Out} = \mathbf{V} \mathbf{A}^\top \in \mathbb{R}^{B \times C_q \times H_q \times W_q}$$
$$\Delta \mathbf{X}_q = \gamma \cdot \operatorname{Out}$$

#### 2. Native Token Spatial Resolution Handling
Because consecutive stages have different spatial dimensions (e.g., $21 \times 21 \to 11 \times 11$, $11 \times 11 \to 6 \times 6$, $6 \times 6 \to 3 \times 3$), standard element-wise addition is impossible without interpolation. Token-based cross-attention completely eliminates lossy bilinear interpolation or pooling:
- Keys and values remain at their **native token resolution** ($N_k$).
- The output $\mathbf{V} \mathbf{A}^\top$ naturally collapses along the key dimension $N_k$, landing directly onto the query grid $N_q = H_q \times W_q$.
- The update is residual: $\mathbf{F}_q' = \mathbf{F}_q + \Delta \mathbf{X}_q$.

#### 3. Parallel Stage Update (No Cascading)
All keys and values are drawn strictly from the **self-attention-refined features prior to any cross-attention updates**. This guarantees parallel execution without sequential dependencies or cascading artifacts.

---

## 3. Investigated Variants

Three model configurations are compared under identical training and evaluation pipelines:

1. **Original MSENet (Baseline):**
   - Standard 5-stage ResNet-18 architecture.
   - Independent spatial self-attention on each stage; no inter-stage cross-attention.
   - Stage weights: $[w_1, w_2, w_3, w_4, w_5]$ initialized to $[1.0, 1.1, 1.2, 1.3, 1.4]$.
   - Attention scaling scalars: $\gamma_1$ (support) and $\gamma_2$ (query), both initialized to $0.2$.

2. **Adjacent Cross-Attention:**
   - Adds 4 unidirectional forward cross-attention links where each shallower stage queries its immediately deeper neighbor:
     - Link 0: Stage 1 queries Stage 2 ($441 \to 441$ tokens, $d=8$)
     - Link 1: Stage 2 queries Stage 3 ($441 \to 121$ tokens, $d=8$)
     - Link 2: Stage 3 queries Stage 4 ($121 \to 36$ tokens, $d=16$)
     - Link 3: Stage 4 queries Stage 5 ($36 \to 9$ tokens, $d=32$)
   - Parameter overhead: $+210,052$ parameters ($+210,048$ conv weights, $+4$ learnable scalar gates).

3. **Bidirectional Cross-Attention:**
   - Incorporates all 4 forward links from the Adjacent model **plus** 4 backward links where deeper stages query shallower neighbors:
     - Backward Link 0: Stage 2 queries Stage 1 ($441 \to 441$ tokens, $d=8$)
     - Backward Link 1: Stage 3 queries Stage 2 ($121 \to 441$ tokens, $d=16$)
     - Backward Link 2: Stage 4 queries Stage 3 ($36 \to 121$ tokens, $d=32$)
     - Backward Link 3: Stage 5 queries Stage 4 ($9 \to 36$ tokens, $d=64$)
   - Parameter overhead: $+452,920$ parameters ($+452,912$ conv weights, $+8$ learnable scalar gates).

---

## 4. Experimental Setup & Evaluation Methodology

The experimental framework maintains strict fairness controls across all three runs:
- **Benchmark Dataset:** MiniImageNet ($84 \times 84$ input resolution, 64 training classes, 16 validation classes, 20 test classes).
- **Episodic Training:** 10 epochs, 360 episodes per epoch, 30-way 5-shot training with 15 query images per class.
- **Backbone & Initialization:** ImageNet-pretrained ResNet-18 backbone. Shared base parameters are identically initialized across runs using fixed random seed (`SEED=42`).
- **Optimization:** Adam optimizer with decoupled learning rates:
  - Backbone parameters: $\eta_{\text{backbone}} = 10^{-4}$
  - Attention projections, scalar weights, and gates: $\eta_{\text{weights}} = 5 \times 10^{-3}$
  - Cosine annealing learning rate schedule across 10 epochs.
  - Automatic Mixed Precision (`torch.cuda.amp.autocast`) enabled.
- **Evaluation Protocol:**
  - 5-way 5-shot evaluation with 15 queries per class (75 total queries per episode).
  - Validation: 600 fixed episodes per epoch.
  - Testing: 600 fixed test episodes evaluated on the **best-validation checkpoint** (Epoch 1 for all models).
  - Test distance metric: Euclidean distance ($d(z_q, c_j) = \|z_q - c_j\|_2$).
  - Paired two-tailed Student's t-test over identical per-episode accuracy vectors.

---

## 5. Folder Structure & Contents

```
Cross-Attention/
├── Source Code - Initial.ipynb                         # Pre-run notebook draft
├── Source Code.ipynb                                   # Executed end-to-end benchmark notebook
├── README.md                                           # Architectural guide, formulation, and setup
├── RESULTS.md                                          # Empirical results, CSV summaries, and analysis
└── results/
    ├── README.txt                                      # Overview text file
    ├── comparison_tables/
    │   ├── table1_accuracy_comparison.csv              # Test accuracy, 95% CIs, paired differences, p-values
    │   ├── table2_efficiency_comparison.csv            # Parameters, runtimes, latency, memory, FLOPs
    │   ├── table3_parameter_breakdown.csv              # Component-wise parameter counts
    │   ├── table4_training_history.csv                 # Per-epoch training/val loss, accuracy, and weights
    │   ├── table5_learned_stage_weights_and_gammas.csv # Checkpoint vs final learned parameter values
    │   ├── table6_cross_attention_diagnostics.csv      # Entropy, max attention weights, update ratios, norms
    │   └── table7_per_episode_test_accuracy.csv        # Raw per-episode accuracy across all 600 test tasks
    ├── parameter_tracking/
    │   ├── learned_params_and_stats_adjacent.json      # Learned parameters and layer-wise weight statistics
    │   ├── learned_params_and_stats_baseline.json      # Baseline learned parameters and weight statistics
    │   ├── learned_params_and_stats_bidirectional.json # Bidirectional learned parameters and diagnostics
    │   ├── raw_results_adjacent.json                   # Complete serialised output dictionary for Adjacent
    │   ├── raw_results_baseline.json                   # Complete serialised output dictionary for Baseline
    │   ├── raw_results_bidirectional.json              # Complete serialised output dictionary for Bidirectional
    │   └── run_config.json                             # Global configuration dictionary
    └── plots/
        ├── fig1_test_accuracy.png                      # Primary test accuracy bar chart with 95% CI
        ├── fig2a_training_accuracy.png                 # Training accuracy progression across 10 epochs
        ├── fig2b_training_loss.png                     # Training loss progression across 10 epochs
        ├── fig2c_validation_accuracy.png               # Validation accuracy progression across 10 epochs
        ├── fig2d_validation_loss.png                   # Validation loss progression across 10 epochs
        ├── fig3a_parameters.png                        # Total parameter count comparison
        ├── fig3b_training_time.png                     # Total training time per model
        ├── fig3c_inference_time.png                    # Inference latency per 5-way 5-shot episode
        ├── fig3d_gpu_memory.png                        # Peak training and inference GPU VRAM footprint
        ├── fig4_accuracy_vs_cost.png                   # Scatter plot of accuracy vs latency and FLOPs
        ├── fig5_learned_weights_and_gammas.png         # Bar plot of learned stage weights and gate values
        ├── fig6_per_episode_accuracy_distribution.png  # Box plot of per-episode test accuracy
        └── fig7_cross_attention_diagnostics.png        # Bar charts of attention entropy and update ratios
```

---

## 6. Summary of Key Results

| Model | Parameters | Training Time | Test Acc (%) | 95% CI (%) | $\Delta$ vs Baseline | Paired $p$-value | Best Val Acc (%) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Original MSENet (Baseline)** | 11,618,120 | 26.29 min | 83.06% | $\pm$0.50% | — | — | **82.38%** (Ep 1) |
| **Adjacent Cross-Attention** | 11,828,172 | 33.57 min | **83.25%** | $\pm$0.52% | **+0.19%** | **0.0409** | 82.26% (Ep 1) |
| **Bidirectional Cross-Attention** | 12,071,040 | 41.42 min | 83.06% | $\pm$0.51% | -0.00% | 0.9814 | 82.24% (Ep 1) |

- **Statistically Significant Modest Gain:** Adjacent Cross-Attention achieves **83.25%** test accuracy (+0.19 pp over baseline) with a statistically significant paired $t$-test $p$-value of **0.0409** ($p < 0.05$).
- **Bidirectional Inefficacy:** Extending cross-attention to bidirectional stage communication yields **83.06%**, matching the baseline exactly ($\Delta = -0.00\%$, $p = 0.9814$) while incurring +57.6% training overhead.
- **Diagnostic Finding:** Cross-attention attention maps exhibit high normalized entropy ($>0.996$), behaving predominantly as diffuse, learnable spatial feature smoothers rather than sharp object-part matchers.

For detailed diagnostic metrics, epoch progression histories, efficiency benchmarks, and parameter analyses, refer to [`RESULTS.md`](RESULTS.md).
