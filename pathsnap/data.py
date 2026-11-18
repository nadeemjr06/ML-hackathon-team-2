"""Loading the cached candidate/observation tables produced by scripts/02_build_dataset.py."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def load_part(part: str):
    cands = pd.read_parquet(C.CACHE_DIR / f"{part}_cands.parquet")
    obs = pd.read_parquet(C.CACHE_DIR / f"{part}_obs.parquet")
    return cands, obs


def training_mask(cands: pd.DataFrame, obs: pd.DataFrame) -> np.ndarray:
    """Rows used for fitting: reliable labels and the true arc present among candidates."""
    key = obs.set_index(["trip_id", "obs_idx"])["gt_in_cands"]
    cov = key.reindex(pd.MultiIndex.from_arrays([cands.trip_id, cands.obs_idx])).to_numpy()
    return cands["reliable"].to_numpy() & cov


def subsample_obs(cands: pd.DataFrame, max_rows: int | None, seed: int) -> np.ndarray:
    """Indices of a random subset of observations (whole candidate groups) with <= max_rows rows."""
    if max_rows is None or len(cands) <= max_rows:
        return np.arange(len(cands))
    keys = cands["trip_id"].astype(str) + ":" + cands["obs_idx"].astype(str)
    uniq = keys.unique()
    rng = np.random.default_rng(seed)
    frac = max_rows / len(cands)
    chosen = set(rng.choice(uniq, size=int(len(uniq) * frac), replace=False))
    return np.flatnonzero(keys.isin(chosen).to_numpy())
