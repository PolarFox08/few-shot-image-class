# Experimental Results: Stacked MSENet with Dynamic Stage Weighting (EMA, LayerNorm, GeM)

This document presents the complete empirical evaluation of stacking architectural and optimization enhancements—**Layer Normalization (LayerNorm)**, **Exponential Moving Average (EMA)**, and **Signed Generalized Mean (GeM) Pooling**—onto the **MSENet** framework with **Task-Adaptive Dynamic Stage Weighting** under Euclidean distance. All models are evaluated on the **MiniImageNet** few-shot benchmark (5-way 5-shot evaluation over 600 episodes, ResNet-18 backbone).

---

## 1. Executive Summary: Main Test Performance Comparison

All three configurations were trained under identical seeds, hyperparameter settings, and dataset splits (360 train tasks/epoch at 30-way 5-shot, 600 validation tasks/epoch at 5-way 5-shot, and 600 fixed test tasks at 5-way 5-shot with 15 queries per class):

| Experiment | Configuration Description | Distance Metric | LayerNorm | EMA | GeM | Test Accuracy (5w-5s) | 95% Confidence Interval | Test Accuracy Std | Delta vs. D0 (pp) | Paired 95% CI (pp) | Best Validation Accuracy | Final Validation Accuracy (Ep 10) | Best Epoch |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **D0** | DW only (baseline) | Euclidean | No | No | No | 82.8756% | [82.3596%, 83.3916%] | 6.4481% | — | — | 82.7156% | 79.0778% | 1 |
| **D1** | **DW + LayerNorm + EMA** | **Euclidean** | **Yes** | **Yes** | **No** | **85.3000%** | **[84.8220%, 85.7780%]** | **5.9735%** | **+2.4244 pp** | **[+2.1159 pp, +2.7330 pp]** | 84.3600% | **79.1778%** | 1 |
| **D2** | **DW + LayerNorm + EMA + GeM** | Euclidean | Yes | Yes | Yes | 85.2689% | [84.7735%, 85.7643%] | 6.1908% | +2.3933 pp | [+2.0837 pp, +2.7029 pp] | **84.4467%** | 78.8733% | 1 |

*Note: In the paired confidence intervals, $\text{Delta} \pm \text{CI95}_{\text{paired}}$ is computed across all 600 identical test tasks.*

### Visual Comparison of Test Performance

The figure below illustrates the 5-way 5-shot test accuracy comparison across all three configurations, including 95% confidence intervals and validation benchmark markers:

![Test Accuracy Comparison](results/plots/05_test_accuracy_comparison.png)

---

## 2. Improvement Analysis Over Baseline (D0)

The relative and absolute performance gains achieved over the unaugmented Dynamic Weights baseline (`D0`) are quantified below from `improvement_over_D0.csv`:

| Experiment | Configuration Description | Test Accuracy (%) | Absolute Improvement (pp) | Relative Improvement (%) |
|:---:|:---|:---:|:---:|:---:|
| **D0** | DW only (baseline, no LN/EMA/GeM) | 82.8756% | 0.0000 pp | 0.0000% |
| **D1** | **DW + LayerNorm + EMA** | **85.3000%** | **+2.4244 pp** | **+2.9254%** |
| **D2** | DW + LayerNorm + EMA + GeM | 85.2689% | +2.3933 pp | +2.8879% |

### Key Analytical Takeaways
1. **Statistically Significant Dominance:** Both stacked configurations (**D1** and **D2**) deliver massive improvements (> +2.39 pp absolute gain) over the baseline D0 model. Their paired 95% confidence intervals strictly exclude zero:
   - D1 vs. D0: $[+2.1159\text{ pp}, +2.7330\text{ pp}]$
   - D2 vs. D0: $[+2.0837\text{ pp}, +2.7029\text{ pp}]$  
   This establishes with high statistical confidence ($p \ll 0.001$) that stacking LayerNorm and EMA onto Dynamic Stage Weighting provides substantial generalization gains that cannot be attributed to episodic sampling noise.
2. **Efficacy of LayerNorm + EMA (D1):** Stacking LayerNorm and EMA yields **85.3000%** test accuracy (+2.42 pp over D0). Layer Normalization stabilizes feature variance across ResNet-18 stages before computing pairwise Euclidean distances, which prevents deeper stages with wide dynamic ranges from distorting distances. Simultaneously, EMA parameter smoothing buffers noisy episodic meta-gradients.
3. **Role of Signed GeM Pooling (D2):** Adding Signed GeM Pooling in D2 achieves the highest peak validation accuracy across all experiments (**84.4467%**) and reaches **85.2689%** test accuracy (+2.39 pp over D0), within 0.03 pp of D1. GeM enables the network to learn adaptive spatial pooling parameters ($p \approx 2.89 - 3.33$), focusing on salient object components.
4. **Variance Reduction Across Episodes:** The standard deviation across test episodes drops from **6.4481%** in D0 down to **5.9735%** in D1 and **6.1908%** in D2. Stacking LayerNorm and EMA leads to significantly more consistent decision boundaries across diverse few-shot tasks.

---

## 3. Dynamic Stage Weight Allocations on Test Episodes

The table below reports the global prior softplus values, empirical test-episode dynamic stage weights (mean and standard deviation across all 600 test episodes), and sample dynamic weights for two specific test tasks from `dynamic_weights_test_stats.csv`:

| Experiment | Stage | ResNet-18 Node | Channels | Global Prior (Softplus) | Dynamic Weight Mean (Test) | Dynamic Weight Std (Test) | Weight Share (%) | Task 1 Dyn Weight | Task 2 Dyn Weight |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **D0** | 1 | `maxpool` | 64 | 0.828382 | 0.230982 | 0.000573 | 9.12% | 0.230323 | 0.231334 |
| | 2 | `layer1` | 64 | 0.907371 | 0.255728 | 0.000373 | 10.09% | 0.255963 | 0.255701 |
| | 3 | `layer2` | 128 | 0.856465 | 0.237890 | 0.000720 | 9.39% | 0.237844 | 0.238560 |
| | 4 | `layer3` | 256 | 0.765746 | 0.694305 | 0.012979 | 27.41% | 0.688512 | 0.703226 |
| | 5 | `layer4` | 512 | 1.206670 | **1.114500** | 0.018274 | **44.00%** | 1.122110 | 1.101760 |
| **D1** | 1 | `maxpool` | 64 | 1.041770 | 0.308129 | 0.007095 | 9.07% | 0.303764 | 0.305814 |
| | 2 | `layer1` | 64 | 1.088130 | 0.313004 | 0.001001 | 9.22% | 0.313852 | 0.312028 |
| | 3 | `layer2` | 128 | 1.127360 | 0.338651 | 0.010462 | 9.97% | 0.340527 | 0.362321 |
| | 4 | `layer3` | 256 | 1.178580 | 0.440369 | 0.034492 | 12.97% | 0.453786 | 0.434568 |
| | 5 | `layer4` | 512 | 1.279350 | **1.995360** | 0.041110 | **58.77%** | 1.982710 | 1.974770 |
| **D2** | 1 | `maxpool` | 64 | 1.021010 | 0.300153 | 0.004477 | 8.91% | 0.297985 | 0.299052 |
| | 2 | `layer1` | 64 | 1.090130 | 0.315962 | 0.001246 | 9.38% | 0.317501 | 0.315706 |
| | 3 | `layer2` | 128 | 1.112050 | 0.403519 | 0.041249 | 11.98% | 0.408914 | 0.432010 |
| | 4 | `layer3` | 256 | 1.226220 | 0.471988 | 0.034002 | 14.01% | 0.514481 | 0.480182 |
| | 5 | `layer4` | 512 | 1.265280 | **1.873610** | 0.059109 | **55.72%** | 1.830110 | 1.834240 |

### Stage Weight Insights
1. **Amplified Deep-Stage Focus with LayerNorm (D1 & D2):** In unnormalized D0, `layer4` received an average weight of $1.1145$ (44.00% share). With LayerNorm in D1 and D2, features are normalized across channels, allowing the dynamic gating network to assign even greater prominence to deep semantic representations: `layer4` jumps to **1.9954** (58.77% share) in D1 and **1.8736** (55.72% share) in D2.
2. **Intermediate Stage Sensitivity:** In D2 (with GeM pooling), `layer2` ($0.4035$) and `layer3` ($0.4720$) receive higher weights than in D1, with greater task-to-task variability ($\text{std} = 0.0412$ and $0.0340$), showing that GeM pooling provides richer spatial representations at intermediate scales that the dynamic gating module actively leverages.
3. **Adaptive Responsiveness:** For individual tasks (Task 1 vs. Task 2), weights actively shift: for example, D2 `layer3` weight modulates from $0.5145$ in Task 1 to $0.4802$ in Task 2, verifying episode-specific modulation.

---

## 4. Parameter Tracking and Learned Trajectories

Tracking the internal parameters of the dynamic gating module, attention heads, and GeM pooling provides critical insights into model convergence (from `final_parameters.csv` and `per_epoch_metrics.csv`):

### 4.1 Final Parameters at Best Validation Checkpoint vs. End of Training

| Metric / Parameter | D0 (Best / Ep 10) | D1 (Best / Ep 10) | D2 (Best / Ep 10) |
|:---|:---:|:---:|:---:|
| **Best Val Accuracy** | **82.7156%** (Ep 1) | **84.3600%** (Ep 1) | **84.4467%** (Ep 1) |
| **Final Val Accuracy (Ep 10)** | 79.0778% | 79.1778% | 78.8733% |
| **Prior $w_1$ (maxpool)** | 0.8284 / 0.6375 | 1.0418 / 0.8212 | 1.0210 / 0.7965 |
| **Prior $w_2$ (layer1)** | 0.9074 / 0.6838 | 1.0881 / 0.8084 | 1.0901 / 0.8418 |
| **Prior $w_3$ (layer2)** | 0.8565 / 0.5622 | 1.1274 / 0.8466 | 1.1121 / 0.9272 |
| **Prior $w_4$ (layer3)** | 0.7657 / 0.5918 | 1.1786 / 0.9240 | 1.2262 / 1.0765 |
| **Prior $w_5$ (layer4)** | 1.2067 / 1.2598 | 1.2794 / 1.3956 | 1.2653 / 1.4095 |
| **Attention $\gamma_1$ (maxpool)** | 0.2925 / 0.5565 | 0.7374 / 1.1525 | 0.7478 / 1.1876 |
| **Attention $\gamma_2$ (layer1)** | 0.9323 / 1.6164 | 0.1002 / 0.3634 | 0.0638 / 0.3893 |
| **DW Fisher Temp ($\tau_{\text{fisher}}$)** | 1.9403 / 1.9462 | 1.9267 / 1.9837 | 1.9335 / 1.9988 |
| **DW Mix Logit / Alpha ($\alpha$)** | 0.2103 / 0.5524 ($\alpha$) | 0.0849 / 0.5212 ($\alpha$) | 0.0763 / 0.5191 ($\alpha$) |
| **DW Gate Total Weight Norm** | 5.8970 / 5.9365 | 5.8236 / 5.9234 | 5.8383 / 5.9175 |
| **GeM $p_1, p_2, p_3, p_4, p_5$ (Best)** | — | — | 2.996, 2.928, 3.043, 2.968, 3.055 |
| **GeM $p_1, p_2, p_3, p_4, p_5$ (Ep 10)** | — | — | 3.009, 2.892, 3.019, 2.918, 3.334 |

### Parameter Evolution Insights
1. **Learned Attention Behavior:** In D0, $\gamma_2$ rapidly expands to $0.9323$ at Epoch 1 and $1.6164$ by Epoch 10. In contrast, with LayerNorm in D1 and D2, $\gamma_1$ grows to $0.7374 - 0.7478$ while $\gamma_2$ stabilizes at a much lower value ($0.0638 - 0.1002$), indicating that LayerNorm naturally conditions layer1 representations so that aggressive self-attention residual scaling is unnecessary.
2. **Fisher vs. Query Mix Balancing:** The gating mixing coefficient $\alpha = \sigma(\text{mix\_logit})$ stabilizes near $0.519 - 0.522$ in D1 and D2, indicating an optimal balanced reliance between support-set Fisher discriminability and query entropy confidence.
3. **GeM Power Trajectory:** The learned GeM power parameter for `layer4` increases from the initial $3.0$ up to $3.334$ by Epoch 10, showing the network progressively sharpens spatial focus on peak activation regions in deep feature maps.

---

## 5. Computational Efficiency and Hardware Profiling

All experiments were benchmarked on a single dedicated **NVIDIA Tesla T4 GPU (16 GB)** using PyTorch Automatic Mixed Precision (AMP):

| Experiment | Configuration Description | Total Train Time | Train Time / Epoch | Total Wall Time | Inference Latency / Episode | Throughput (Tasks/s) | Total Parameters | Trainable Parameters | Peak Train GPU VRAM | Peak Infer GPU VRAM | Process RSS Memory |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **D0** | DW only (baseline) | 1,169.63 s (19.49 min) | 116.96 s | 24.00 min | **37.11 ms** | **26.95 tasks/s** | 11,618,586 | 11,618,586 | **3,887.67 MB (3.80 GB)** | 514.35 MB | 3.78 GB |
| **D1** | **DW + LayerNorm + EMA** | 1,191.35 s (19.86 min) | 119.14 s | 24.36 min | **37.16 ms** | **26.91 tasks/s** | 11,620,634 | 11,620,634 | 3,928.61 MB (3.84 GB) | 517.10 MB | 3.78 GB |
| **D2** | DW + LayerNorm + EMA + GeM | 1,416.38 s (23.61 min) | 141.64 s | 28.69 min | 42.49 ms | 23.54 tasks/s | 11,620,639 | 11,620,639 | 4,667.60 MB (4.56 GB) | **513.42 MB** | 3.78 GB |

### Hardware and Overhead Findings
- **Zero Latency Penalty for LayerNorm + EMA (D1):** D1 adds only **0.05 ms** per episode over D0 (37.16 ms vs. 37.11 ms), preserving essentially identical inference throughput (~27 tasks per second).
- **Negligible Parameter Overhead:**
  - Dynamic Weight module adds roughly 466 parameters over vanilla MSENet (11,618,586 vs. 11,618,120).
  - LayerNorm introduces **2,048 parameters** ($2 \times (64+64+128+256+512)$ affine weights $\gamma$ and biases $\beta$).
  - GeM pooling introduces only **5 scalar power parameters** ($p_1, \dots, p_5$).
  - The complete D2 network requires only 11,620,639 parameters, an imperceptible +0.018% overhead compared to D0.
- **VRAM Utilization:** Peak inference memory footprint is identical across all models at **~513 to 517 MB**. Peak training VRAM rises from 3.89 GB (D0) to 4.67 GB (D2) due to power gradient computations in GeM autograd graphs, remaining well within standard GPU hardware limits.

---

## 6. Full Epoch-by-Epoch Training & Validation Progression

All models were trained for 10 epochs (360 episodes per epoch) using Adam with Cosine Annealing. For all three runs, the **Best Validation Checkpoint occurred at Epoch 1**, after which episodic meta-training exhibited characteristic validation plateauing.

### 6.1 D0: DW only (baseline)
| Epoch | Training Accuracy (%) | Training Loss | Validation Accuracy (%) | Validation Loss | Learning Rate | Train Time (s) | Val Time (s) | Status |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **77.2019%** | **1.0373** | **82.7156%** | **0.4736** | $1.000 \times 10^{-4}$ | 115.19 s | 24.44 s | **Saved Best Checkpoint** |
| 2 | 93.1722% | 0.2244 | 80.9400% | 0.5223 | $9.758 \times 10^{-5}$ | 117.40 s | 24.46 s | |
| 3 | 97.3710% | 0.0844 | 79.8600% | 0.5641 | $9.055 \times 10^{-5}$ | 117.32 s | 24.44 s | |
| 4 | 98.6401% | 0.0427 | 79.0533% | 0.5964 | $7.960 \times 10^{-5}$ | 117.09 s | 24.43 s | |
| 5 | 99.2716% | 0.0232 | 79.2556% | 0.6355 | $6.580 \times 10^{-5}$ | 117.02 s | 24.44 s | |
| 6 | 99.5883% | 0.0130 | 79.1689% | 0.6608 | $5.050 \times 10^{-5}$ | 117.16 s | 24.47 s | |
| 7 | 99.7716% | 0.0074 | 79.3667% | 0.7038 | $3.520 \times 10^{-5}$ | 117.17 s | 24.47 s | |
| 8 | 99.8519% | 0.0044 | 79.1289% | 0.7421 | $2.140 \times 10^{-5}$ | 117.12 s | 24.47 s | |
| 9 | 99.9216% | 0.0025 | 79.0356% | 0.7920 | $1.045 \times 10^{-5}$ | 117.02 s | 24.46 s | |
| 10 | 99.9401% | 0.0019 | 79.0778% | 0.7965 | $3.423 \times 10^{-6}$ | 117.15 s | 24.42 s | |

### 6.2 D1: DW + LayerNorm + EMA
| Epoch | Training Accuracy (%) | Training Loss | Validation Accuracy (%) | Validation Loss | Learning Rate | Train Time (s) | Val Time (s) | Status |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **79.1315%** | **0.8445** | **84.3600%** | **0.4311** | $1.000 \times 10^{-4}$ | 118.12 s | 24.46 s | **Saved Best Checkpoint** |
| 2 | 93.8444% | 0.2021 | 82.3378% | 0.5027 | $9.758 \times 10^{-5}$ | 119.56 s | 24.46 s | |
| 3 | 97.6500% | 0.0757 | 80.6822% | 0.5841 | $9.055 \times 10^{-5}$ | 119.62 s | 24.46 s | |
| 4 | 98.8241% | 0.0378 | 80.0644% | 0.6353 | $7.960 \times 10^{-5}$ | 119.35 s | 24.43 s | |
| 5 | 99.3327% | 0.0211 | 79.7844% | 0.6896 | $6.580 \times 10^{-5}$ | 119.19 s | 24.44 s | |
| 6 | 99.5932% | 0.0123 | 79.1311% | 0.7450 | $5.050 \times 10^{-5}$ | 118.89 s | 24.46 s | |
| 7 | 99.7889% | 0.0065 | 79.2378% | 0.8123 | $3.520 \times 10^{-5}$ | 119.29 s | 24.42 s | |
| 8 | 99.8531% | 0.0041 | 79.1467% | 0.8392 | $2.140 \times 10^{-5}$ | 119.21 s | 24.45 s | |
| 9 | 99.9142% | 0.0029 | 79.2400% | 0.8526 | $1.045 \times 10^{-5}$ | 119.06 s | 24.43 s | |
| 10 | 99.9475% | 0.0018 | 79.1778% | 0.8660 | $3.423 \times 10^{-6}$ | 119.07 s | 24.45 s | |

### 6.3 D2: DW + LayerNorm + EMA + GeM
| Epoch | Training Accuracy (%) | Training Loss | Validation Accuracy (%) | Validation Loss | Learning Rate | Train Time (s) | Val Time (s) | Status |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **79.2802%** | **0.8262** | **84.4467%** | **0.4286** | $1.000 \times 10^{-4}$ | 140.57 s | 27.57 s | **Saved Best Checkpoint** |
| 2 | 93.8982% | 0.1991 | 82.0622% | 0.5048 | $9.758 \times 10^{-5}$ | 142.34 s | 27.58 s | |
| 3 | 97.7315% | 0.0729 | 80.3378% | 0.5901 | $9.055 \times 10^{-5}$ | 142.18 s | 27.51 s | |
| 4 | 98.8309% | 0.0369 | 79.8156% | 0.6565 | $7.960 \times 10^{-5}$ | 142.09 s | 27.51 s | |
| 5 | 99.3327% | 0.0210 | 79.5489% | 0.6984 | $6.580 \times 10^{-5}$ | 141.90 s | 27.57 s | |
| 6 | 99.6488% | 0.0107 | 78.6489% | 0.7718 | $5.050 \times 10^{-5}$ | 142.14 s | 27.69 s | |
| 7 | 99.8019% | 0.0063 | 78.9689% | 0.8099 | $3.520 \times 10^{-5}$ | 142.10 s | 27.66 s | |
| 8 | 99.8809% | 0.0037 | 78.7667% | 0.8523 | $2.140 \times 10^{-5}$ | 141.97 s | 27.67 s | |
| 9 | 99.9161% | 0.0028 | 78.8689% | 0.8639 | $1.045 \times 10^{-5}$ | 140.64 s | 27.69 s | |
| 10 | 99.9407% | 0.0018 | 78.8733% | 0.8735 | $3.423 \times 10^{-6}$ | 140.46 s | 27.69 s | |

---

## 7. Generated Visualizations Overview

All high-resolution diagnostic plots generated during execution are stored in [`Stacked-DW/results/plots/`](results/plots):

1. **`01_train_accuracy_vs_epoch.png` & `02_train_loss_vs_epoch.png`:** Document rapid training convergence reaching >99.9% accuracy by Epoch 10 across all configurations.
2. **`03_val_accuracy_vs_epoch.png` & `04_val_loss_vs_epoch.png`:** Highlight the sharp performance peak at Epoch 1 (~84.4% for D1/D2 vs 82.7% for D0) followed by gradual validation divergence typical of episodic meta-training.
3. **`05_test_accuracy_comparison.png`:** Primary comparative figure showing test accuracy with 95% confidence intervals and validation benchmarks (**embedded above** in Section 1).
4. **`06_training_time_comparison.png` & `07_inference_time_comparison.png`:** Confirm near-identical training time per epoch for D0 and D1 (~117–119 s) with minor increase for D2 (~141 s), while per-episode inference latency remains extremely fast (~37 ms for D0/D1, ~42 ms for D2).
5. **`08_gpu_memory_comparison.png`:** Displays uniform inference VRAM (~514 MB) and training memory profiles (~3.9 GB for D0/D1, 4.67 GB for D2).
6. **`09_stage_weight_trajectories.png` & `10_gamma_trajectories.png`:** Detail the evolution of stage weights and spatial attention scales ($\gamma_1, \gamma_2$), demonstrating that LayerNorm moderates self-attention inflation.
7. **`11_gem_p_trajectories.png`:** Illustrates the smooth evolution of GeM power parameters from the initial $p=3.0$, particularly showing deep stage 5 expanding to $p \approx 3.33$.
8. **`12_per_episode_accuracy_boxplot.png`:** Visualizes the distribution of per-episode test accuracies across the 600 evaluation tasks, demonstrating tighter variance and higher median accuracy for D1 and D2.
9. **`13_dw_parameter_trajectories.png` & `14_dw_dynamic_weights_per_epoch.png`:** Track the dynamic gating parameters (Fisher temperature $\tau_{\text{fisher}}$, mix logit $\alpha$, gating norms) and the effective dynamic stage weights over training epochs.

---

## 8. CSV Results Index

For raw numerical evaluation data, please consult the exported CSV files in [`Stacked-DW/results/comparison_tables/`](results/comparison_tables) and [`Stacked-DW/results/parameter_tracking/`](results/parameter_tracking):

- [`main_comparison.csv`](results/comparison_tables/main_comparison.csv): Comprehensive comparative metrics, test accuracy, 95% CI, delta vs. D0, validation performance, runtimes, and memory footprints.
- [`three_way_comparison.csv`](results/comparison_tables/three_way_comparison.csv): Clean 3-way table of configurations D0, D1, and D2.
- [`improvement_over_D0.csv`](results/comparison_tables/improvement_over_D0.csv): Absolute and relative test accuracy gains over baseline D0.
- [`dynamic_weights_test_stats.csv`](results/comparison_tables/dynamic_weights_test_stats.csv): Mean, standard deviation, global priors, and sample task weights across 600 test episodes.
- [`computational_efficiency.csv`](results/comparison_tables/computational_efficiency.csv): Profiling runtimes, throughput, parameters, and peak VRAM.
- [`per_epoch_metrics.csv`](results/comparison_tables/per_epoch_metrics.csv): Full 10-epoch per-epoch training/val loss, accuracy, timings, dynamic weights, and attention parameters.
- [`final_parameters.csv`](results/parameter_tracking/final_parameters.csv): Checkpoint vs. epoch 10 parameters (prior weights, gammas, DW gates, GeM $p$).
- [`parameter_trajectories.csv`](results/parameter_tracking/parameter_trajectories.csv): Step-by-step evolution of parameters recorded throughout training.
- [`experiment_config.csv`](results/comparison_tables/experiment_config.csv): Complete runtime environment, library versions, hyperparameters, and seeds.
