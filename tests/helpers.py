from functools import lru_cache

from argus.config import load_config
from argus.inference import ArgusEngine


@lru_cache(maxsize=1)
def small_engine():
    cfg = load_config(overrides={"data": {"n_flows": 9000},
                                 "models": {"autoencoder": {"epochs": 10}, "graph": {"epochs": 10}}})
    return ArgusEngine.train(cfg)
