"""Optional MLflow tracking: silently no-ops when mlflow is not installed."""
from __future__ import annotations

from contextlib import contextmanager


def _flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "."))
        else:
            out[key] = v
    return out


class _Null:
    def log_metrics(self, m: dict, step: int | None = None): ...
    def log_artifacts(self, path: str): ...


class _Run:
    def __init__(self, mlflow):
        self.m = mlflow

    def log_metrics(self, metrics: dict, step: int | None = None):
        flat = {k: float(v) for k, v in _flat(metrics).items() if isinstance(v, (int, float)) and v == v}
        self.m.log_metrics(flat, step=step)

    def log_artifacts(self, path: str):
        self.m.log_artifacts(path)


@contextmanager
def track(run_name: str, params: dict | None = None, experiment: str = "argus-xai"):
    try:
        import mlflow
    except Exception:
        yield _Null()
        return
    try:
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=run_name):
            if params:
                mlflow.log_params({k: str(v)[:250] for k, v in _flat(params).items()})
            yield _Run(mlflow)
    except Exception as e:  # tracking server down must never break training
        print(f"[tracking] disabled: {e}")
        yield _Null()
