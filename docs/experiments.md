# Experiments

Run all: `make experiments`. Each script: `--seeds 0 1 2 --n-flows N [--config path]`.
Outputs: `experiments/<name>/results/*.{csv,md,json}`.

| ID | Output | What to look at |
|---|---|---|
| 01 baseline | `baseline_comparison` | `auroc_unseen`, `recall_unseen@5%FPR` of supervised vs AE/VAE vs HYBRID; known-class performance must not degrade |
| 02 GNN | `tabular_vs_graph` | hybrid with vs without the graph component; per-unseen-class recall |
| 03 unseen | `unseen_attack_detection` | leave-one-class-out; `labelled_unknown` and fusion-rule comparison |
| 04 XAI | `attribution_fidelity`, `counterfactuals` | risk drop of top-k features vs `random`; counterfactual validity and sparsity |
| 05 continual | `continual_learning`, `drift_detection` | static vs naive vs replay vs full: `mean_forgetting_final`, newest-class recall |

Suggested extra ablations: `features.use_novelty=false`; fusion weights; `target_fpr` sweep;
VAE vs AE (needs torch); temporal split; transfer (train on CICIDS2017, test on CSE-CIC-IDS2018).
