"""Shared evaluation helpers for scripts 03-05."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import classification_metrics, top1_from_probs


def eval_probs(cands: pd.DataFrame, obs: pd.DataFrame, p: np.ndarray) -> dict:
    """Metrics on observations with a reliable reference label (see config.LABEL_MAX_DIST_M)."""
    m = cands["reliable"].to_numpy()
    o = obs[obs["reliable"]]
    out = classification_metrics(cands["label"].to_numpy()[m], p[m])
    out.update(top1_from_probs(cands[m], p[m], o))
    return out


def group_lists(trip_cands: pd.DataFrame, p: np.ndarray, n_obs: int):
    """Per-observation candidate lists (arcs, projection t, probability) for the HMM."""
    arcs = [np.zeros(0, int)] * n_obs
    ts = [np.zeros(0)] * n_obs
    ps = [np.zeros(0)] * n_obs
    o = trip_cands["obs_idx"].to_numpy()
    a = trip_cands["arc_id"].to_numpy()
    t = trip_cands["proj_t"].to_numpy()
    if len(o) == 0:
        return arcs, ts, ps
    bounds = np.flatnonzero(np.r_[True, o[1:] != o[:-1], True])
    for s, e in zip(bounds[:-1], bounds[1:]):
        i = o[s]
        arcs[i], ts[i], ps[i] = a[s:e], t[s:e], p[s:e]
    return arcs, ts, ps
