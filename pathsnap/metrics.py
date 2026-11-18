"""Evaluation metrics: candidate-level classification and route-level matching."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score)

from .network import RoadNetwork


def classification_metrics(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> dict:
    """Row-level (candidate) metrics. Macro averages over the two classes."""
    yhat = (p >= thr).astype(int)
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    return {
        "accuracy": accuracy_score(y, yhat),
        "precision_macro": precision_score(y, yhat, average="macro", zero_division=0),
        "recall_macro": recall_score(y, yhat, average="macro", zero_division=0),
        "f1_macro": f1_score(y, yhat, average="macro", zero_division=0),
        "log_loss": log_loss(y, pc, labels=[0, 1]),
        "confusion_matrix": confusion_matrix(y, yhat, labels=[0, 1]).tolist(),
    }


def top1_from_probs(df: pd.DataFrame, p: np.ndarray, obs: pd.DataFrame) -> dict:
    """Observation-level Top-1 accuracy.

    df : candidate rows with trip_id, obs_idx, arc_id, label
    obs: one row per observation with trip_id, obs_idx, gt_in_cands
    Observations whose reference arc is not among the candidates count as errors.
    """
    d = df[["trip_id", "obs_idx", "label"]].copy()
    d["p"] = p
    idx = d.groupby(["trip_id", "obs_idx"], sort=False)["p"].idxmax()
    correct = d.loc[idx, "label"].sum()
    n_obs = len(obs)
    n_cover = int(obs["gt_in_cands"].sum())
    return {
        "top1_acc": correct / n_obs,
        "top1_acc_given_candidate": correct / max(n_cover, 1),
        "candidate_miss_rate": 1 - n_cover / n_obs,
    }


def argmax_arcs(trip_df: pd.DataFrame, p: np.ndarray, n_obs: int) -> np.ndarray:
    """Classifier-only prediction: highest-probability arc per observation (-1 if none)."""
    out = np.full(n_obs, -1, dtype=int)
    d = trip_df[["obs_idx", "arc_id"]].copy()
    d["p"] = p
    best = d.loc[d.groupby("obs_idx", sort=False)["p"].idxmax()]
    out[best["obs_idx"].to_numpy()] = best["arc_id"].to_numpy()
    return out


def compress(seq) -> list[int]:
    out: list[int] = []
    for a in seq:
        if a < 0:
            continue
        if not out or out[-1] != a:
            out.append(int(a))
    return out


def route_continuity(net: RoadNetwork, arcs_per_obs: np.ndarray) -> float:
    """Fraction of consecutive *distinct* predicted arcs that are directly connected
    (to-node of the first == from-node of the second). 1.0 = a drivable route."""
    seq = compress(arcs_per_obs)
    if len(seq) < 2:
        return 1.0
    s = np.array(seq)
    return float(np.mean(net.arcs[s[:-1], 1] == net.arcs[s[1:], 0]))


def _levenshtein(a: list, b: list) -> int:
    if not a:
        return len(b)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def sequence_metrics(net: RoadNetwork, pred_arcs_per_obs: np.ndarray, route: np.ndarray) -> dict:
    """Route-level comparison between the compressed predicted arc sequence and the
    reference route: length-weighted precision/recall/F1 over arcs and an
    edit-distance similarity of the two sequences."""
    pred = compress(pred_arcs_per_obs)
    ref = compress(route)
    ps, rs = set(pred), set(ref)
    inter = ps & rs
    L = net.length
    lp = L[list(ps)].sum() if ps else 0.0
    lr = L[list(rs)].sum() if rs else 0.0
    li = L[list(inter)].sum() if inter else 0.0
    prec = li / lp if lp > 0 else 0.0
    rec = li / lr if lr > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    ed = _levenshtein(pred, ref) if len(pred) * len(ref) < 4e6 else None
    sim = 1 - ed / max(len(pred), len(ref)) if ed is not None else float("nan")
    return {"route_precision": prec, "route_recall": rec, "route_f1": f1, "route_edit_similarity": sim}


def observation_accuracy(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(pred == gt))
