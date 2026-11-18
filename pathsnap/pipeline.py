"""End-to-end inference: raw trip (track + road network) -> matched arc per GPS observation.

``MapMatcher.predict`` never sees the reference route. Evaluation against the
reference route lives in ``evaluate_against_reference`` and is called
separately (server-side only).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np

from . import config as C
from .evaluate import group_lists
from .features import build_features, prepare
from .hmm import HMMConfig, viterbi
from .io import Trip
from .labels import align_track_to_route
from .metrics import argmax_arcs, compress, route_continuity, sequence_metrics
from .network import RoadNetwork


@dataclass
class MatchResult:
    trip_id: str
    net: RoadNetwork
    px: np.ndarray
    py: np.ndarray
    track: np.ndarray
    clf_arcs: np.ndarray
    hmm_arcs: np.ndarray
    has_candidate: np.ndarray
    timings: dict = field(default_factory=dict)


class MapMatcher:
    def __init__(self, model_dir: Path = C.MODELS_DIR):
        model_dir = Path(model_dir)
        self.model = joblib.load(model_dir / "final_model.joblib")
        self.meta = json.loads((model_dir / "model_meta.json").read_text())
        self.features = self.meta["features"]
        hp = model_dir / "hmm_config.json"
        self.hmm_cfg = HMMConfig(**json.loads(hp.read_text())) if hp.exists() else HMMConfig()

    def predict(self, trip: Trip) -> MatchResult:
        tm = {}
        t0 = time.perf_counter()
        proj, net, px, py = prepare(trip)
        tm["prepare_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        df = build_features(trip, net, px, py)
        tm["features_s"] = time.perf_counter() - t0
        n = len(px)
        t0 = time.perf_counter()
        p = self.model.predict_proba(df[self.features].to_numpy(np.float32))[:, 1] if len(df) else np.zeros(0)
        tm["classifier_s"] = time.perf_counter() - t0
        clf = argmax_arcs(df, p, n)
        t0 = time.perf_counter()
        arcs, ts, ps = group_lists(df, p, n)
        hmm = viterbi(net, px, py, trip.track[:, 2], arcs, ts, ps, self.hmm_cfg)
        tm["viterbi_s"] = time.perf_counter() - t0
        has_c = np.array([len(a) > 0 for a in arcs])
        return MatchResult(trip.trip_id, net, px, py, trip.track, clf, hmm, has_c, tm)


def route_segments(net: RoadNetwork, arcs_per_obs, fill: bool = False) -> list[list[list[float]]]:
    """Split the compressed arc sequence into connected polylines ([lat, lon]).
    With ``fill`` the shortest-path arcs between non-adjacent decoded arcs are inserted."""
    seq = compress(arcs_per_obs)
    if fill:
        seq = net.fill_route(seq)
    segs, cur = [], []
    for a in seq:
        if cur and net.arcs[cur[-1], 1] != net.arcs[a, 0]:
            segs.append(cur)
            cur = []
        cur.append(a)
    if cur:
        segs.append(cur)
    return [net.route_polyline(s) for s in segs]


def evaluate_against_reference(res: MatchResult, route: np.ndarray) -> dict:
    """Server-side scoring of a prediction against the reference route."""
    gt, gd, _ = align_track_to_route(res.px, res.py, res.net, route)
    rel = gd <= C.LABEL_MAX_DIST_M
    filled = res.net.fill_route(compress(res.hmm_arcs))
    out = {
        "n_observations": int(len(gt)),
        "n_reliable_observations": int(rel.sum()),
        "accuracy_hmm": float(np.mean(res.hmm_arcs[rel] == gt[rel])) if rel.any() else None,
        "accuracy_classifier_only": float(np.mean(res.clf_arcs[rel] == gt[rel])) if rel.any() else None,
        "accuracy_hmm_all_obs": float(np.mean(res.hmm_arcs == gt)),
        "accuracy_classifier_only_all_obs": float(np.mean(res.clf_arcs == gt)),
        "route_continuity_hmm": route_continuity(res.net, np.array(filled)),
        "route_continuity_hmm_unfilled": route_continuity(res.net, res.hmm_arcs),
        "route_continuity_classifier_only": route_continuity(res.net, res.clf_arcs),
    }
    for k, v in sequence_metrics(res.net, np.array(filled), route).items():
        out[f"{k}_hmm"] = float(v)
    for k, v in sequence_metrics(res.net, res.clf_arcs, route).items():
        out[f"{k}_classifier_only"] = float(v)
    return out
