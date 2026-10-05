# Contributing

1. `pip install -e ".[dev,boost,xai,api]"` (add `dl` for the PyTorch paths).
2. `ruff check src api tests experiments` and `pytest -q` must pass.
3. Any new experiment goes in `experiments/experiment_NN_<name>/run.py`, takes `--seeds`, reports
   mean ± std over seeds, and writes into its own `results/` folder. Do not hand-edit result files.
4. Never evaluate on data used for threshold selection; never let unseen-class labels reach training.
