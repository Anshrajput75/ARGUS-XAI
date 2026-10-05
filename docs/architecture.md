# Architecture

| Layer | Module | Responsibility |
|---|---|---|
| Data | `argus.data`, `argus.preprocessing` | synthetic generator; CICIDS/UNSW loaders; label taxonomy; open-set split |
| Features | `argus.features` | 18 flow features + 5 host-window aggregates (+ `dst_unseen` novelty) |
| Graph | `argus.graph` | nodes = (host, 60 s window), edges = flows; node/edge context features |
| Models | `argus.models.*` | supervised (`xgboost/`), autoencoder/VAE (`autoencoder/`), graph (`gnn/`), `hybrid.py` |
| Explainability | `argus.explainability` | TreeSHAP, LIME, occlusion, AE/graph error shares, greedy counterfactuals, narratives |
| Continual learning | `argus.continual_learning` | class-balanced replay, PSI/KS drift, strategies, forgetting metrics |
| Serving | `argus.inference`, `api/` | `ArgusEngine` (train/analyze/explain/update/save/load); FastAPI |
| Storage/MLOps | `argus.storage`, `argus.tracking`, `docker/` | SQLAlchemy (SQLite/PostgreSQL), Redis cache, MLflow, compose stack |
| UI | `dashboard/` | KPIs, investigation graph, host drill-down, explanations, feedback |

## Hybrid decision engine

1. Supervised score `s_sup = 1 - P(BENIGN)`.
2. Autoencoder / graph reconstruction errors are calibrated on benign validation data:
   `s = sigmoid((log err - mu) / sd - z_offset)`.
3. Fusion (`max` | `noisy_or` | `mean`, weights configurable) -> `risk`.
4. Alert iff `risk >= threshold`, threshold = benign-validation `(1 - target_fpr)` quantile of fused risk.
5. Verdict: not flagged -> Normal; flagged and top attack class prob >= `known_class_confidence` ->
   Known attack; else -> Anomalous - previously unseen behavior.

## Graph model

GraphSAGE-style 2-layer mean aggregation (undirected) over node features
`[mean out-edge features, mean in-edge features, log degrees, log unique neighbours]`; an edge decoder
predicts the flow's standardised features from its two endpoint embeddings. Trained on benign training
edges only; anomaly score = reconstruction error. Pure PyTorch, no PyG dependency.

## API

`POST /analyze` `POST /explain` `GET /alerts` `GET /hosts/risk` `GET /graph` `POST /feedback`
`POST /update` `GET /model/info` `GET /health` - interactive docs at `/docs`.
