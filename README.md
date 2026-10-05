# ARGUS-XAI

**An explainable, continually learning network-anomaly intelligence platform** - hybrid supervised /
unsupervised / graph detection of *previously unseen* attacks, with analyst-readable explanations,
an investigation graph, and feedback-driven model updates.

```
NETWORK FLOWS -> feature engine (flow + host-window aggregates + novelty)
                      |
        +-------------+--------------+
        v             v              v
   XGBoost/HGB    Autoencoder/VAE   Graph model (GraphSAGE edge-reconstruction)
   (known attacks) (benign-only)     (benign-only, neighbourhood context)
        +-------------+--------------+
                      v
           Hybrid decision engine  (calibrated scores -> fused risk; threshold at target benign FPR)
                      |
     Normal | Known attack: <class> | Anomalous - previously unseen behavior
                      |
   Explanations (TreeSHAP / LIME / occlusion / AE-error share / graph-error share + counterfactuals)
                      |
   FastAPI + PostgreSQL + Redis + MLflow  ->  Streamlit SOC dashboard  ->  analyst feedback
                      |                                                        |
                      +------------- continual update (replay) <---------------+
```

## Research questions (one experiment each)

| # | Question | Script |
|---|----------|--------|
| 1 | Does a supervised + unsupervised hybrid detect unseen attacks better than any single model? | `experiments/experiment_01_baseline` |
| 2 | Does network topology add signal over per-flow tabular models? | `experiment_02_gnn` |
| 3 | Leave-one-attack-out: are unseen attacks flagged *and* reported as "unseen" rather than forced into a known class? | `experiment_03_unknown_attacks` |
| 4 | Are explanations faithful (deletion test vs random) and are counterfactuals valid/sparse? | `experiment_04_xai` |
| 5 | Can the model learn analyst-labelled new classes without catastrophic forgetting? | `experiment_05_continual_learning` |

Every experiment: multi-seed, mean ± std, writes CSV/JSON/Markdown into its own `results/` folder.
**No numbers are pre-filled in this repository** - run them yourself (see below), and run them on
real datasets before quoting anything.

## Quickstart

```bash
pip install -e ".[all]"            # or: pip install -e .  (core only; sklearn fallbacks are used)
python -m argus.cli train          # trains on synthetic flows, saves artifacts/model
pytest -q
uvicorn api.main:app               # http://localhost:8000/docs
streamlit run dashboard/app.py     # http://localhost:8501
make experiments                   # runs all five experiments (3 seeds)
docker compose -f docker/docker-compose.yml up --build   # api + dashboard + postgres + redis + mlflow
```

Dependency tiers (everything degrades gracefully):

| Missing | Fallback |
|---|---|
| `torch` | autoencoder -> sklearn MLP autoencoder (no VAE); graph model -> neighbourhood-context autoencoder (no message passing) |
| `xgboost` | sklearn `HistGradientBoostingClassifier` (TreeSHAP unavailable -> occlusion attributions) |
| `shap` / `lime` | occlusion attributions only |
| `mlflow` | tracking silently disabled |

## Real data

See `data/README.md`. Set `data.source: cicids | unsw` and point `data.path` to your CSVs.

## How the open-set protocol works

Training sees only BENIGN + `known_attacks`. `unknown_attacks` appear **only** in the test set. The alert
threshold and score calibrators are fit on *benign validation traffic only* (target FPR, default 5%), so
no attack label of any kind influences operating-point selection. A flagged flow is reported as
*Known attack: X* only if the supervised head is confident (`known_class_confidence`); otherwise it is
*Anomalous - previously unseen behavior*.

## Limitations (read before citing)

* The built-in **synthetic data is easy** and exists for tests/demos; conclusions need CICIDS2017 /
  CSE-CIC-IDS2018 / UNSW-NB15 (note the documented label/flow-construction problems in CICIDS2017).
* Graph is built **transductively** over the evaluated timeline (label-free). A streaming deployment sees
  smaller windows; degree-based features shift accordingly.
* "Risk" is a benign-calibrated anomaly score, **not** a probability of attack.
* Counterfactuals are behavioural (feature-space) diagnostics, not packet-realisable edits.
* Explanation *fidelity* is measured; analyst *usefulness* would need a user study.
* Model artifacts are pickled with joblib: only load files you produced.
* Continual learning updates the supervised head; the unsupervised detectors are frozen between retrains.

See `docs/` for architecture, methodology, experiment protocol and the threat model.
