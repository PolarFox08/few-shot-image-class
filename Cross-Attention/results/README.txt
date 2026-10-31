MSENet vs Adjacent vs Bidirectional Cross-Attention - MiniImageNet
Config: 10 epochs x 360 episodes, 30-way 5-shot 15-query training, euclidean distance, seed 42.
Test: 600 fixed 5-way episodes per shot (5-way 5-shot), identical for all models, using the best-validation checkpoint.
Files
  table1_accuracy_comparison.csv            test accuracy, 95% CI, paired difference vs baseline
  table2_efficiency_comparison.csv          params, training time, inference time, FLOPs, GPU memory
  table3_parameter_breakdown.csv            parameter counts per component
  table4_training_history.csv               per-epoch loss / val accuracy / time / learned weights
  table5_learned_stage_weights_and_gammas.csv
  table6_cross_attention_diagnostics.csv    per-link gate, attention entropy / max weight, update ratio, projection weight norms
  table7_per_episode_test_accuracy.csv
  learned_params_and_stats_<model>.json     compact weight/statistics reports
  raw_results_<model>.json, run_config.json
  fig1..fig7 *.png (accuracy, training/validation curves, efficiency, accuracy-vs-cost, learned weights, per-episode distribution, cross-attention diagnostics)
No model checkpoints are included.
