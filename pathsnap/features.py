"""Candidate generation + feature engineering for one trip.

Each row of the output describes one (GPS observation, candidate arc) pair.
The classifier is a *binary* model: P(candidate arc is the true arc | features).
This formulation works for any road network (arc ids are trip-local, so a
multi-class model over arc ids would not generalise).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .geo import LocalProjection, point_segment
from .io import Trip
from .network import RoadNetwork


def prepare(trip: Trip):
    """Project the trip and build its road network index."""
    proj = LocalProjection.from_points(trip.track[:, 0], trip.track[:, 1])
    px, py = proj.to_xy(trip.track[:, 0], trip.track[:, 1])
    net = RoadNetwork(trip.nodes, trip.arcs, proj, focus_xy=np.c_[px, py])
    return proj, net, px, py


def _window_index(n: int, w: int):
    i = np.arange(n)
    return np.clip(i - w, 0, n - 1), np.clip(i + w, 0, n - 1)


def build_features(trip: Trip, net: RoadNetwork, px: np.ndarray, py: np.ndarray) -> pd.DataFrame:
    t = trip.track[:, 2]
    n = len(px)
    cands = net.candidates(px, py)

    # ---- per-observation kinematics
    lo, hi = _window_index(n, C.HEADING_WINDOW)
    hx, hy = px[hi] - px[lo], py[hi] - py[lo]
    hdt = np.maximum(t[hi] - t[lo], 1e-6)
    disp = np.hypot(hx, hy)
    speed = disp / hdt
    heading = np.arctan2(hx, hy)
    heading_valid = (speed >= C.MIN_SPEED_FOR_HEADING).astype(float)
    dt_prev = np.r_[0.0, np.diff(t)]
    dt_next = np.r_[np.diff(t), 0.0]

    # ---- flatten candidates
    lens = np.array([len(c[0]) for c in cands])
    obs = np.repeat(np.arange(n), lens)
    if len(obs) == 0:
        return pd.DataFrame(columns=["obs_idx", "arc_id", *C.FEATURES])
    arc = np.concatenate([c[0] for c in cands]).astype(int)
    dist = np.concatenate([c[1] for c in cands])
    proj_t = np.concatenate([c[2] for c in cands])
    rank = np.concatenate([np.arange(k) for k in lens]).astype(float)
    n_c = np.repeat(lens, lens).astype(float)
    dmin = np.repeat(np.array([c[1][0] if len(c[1]) else 0.0 for c in cands]), lens)

    # ---- window distance: how well the arc's line fits neighbouring points
    w = C.DIST_WINDOW
    offs = np.arange(-w, w + 1)
    nb = np.clip(obs[:, None] + offs[None, :], 0, n - 1)
    wd, _ = point_segment(px[nb], py[nb], net.ax[arc][:, None], net.ay[arc][:, None],
                          net.bx[arc][:, None], net.by[arc][:, None])
    win_mean = wd.mean(axis=1)
    win_max = wd.max(axis=1)

    df = pd.DataFrame({"obs_idx": obs, "arc_id": arc, "dist": dist, "dist_rank": rank,
                       "n_cands": n_c, "proj_t": proj_t, "win_dist_mean": win_mean,
                       "win_dist_max": win_max})
    df["dist_minus_min"] = dist - dmin
    df["dist_ratio"] = dist / (dmin + 1.0)
    g = df.groupby("obs_idx", sort=False)
    df["win_dist_minus_min"] = win_mean - g["win_dist_mean"].transform("min").to_numpy()

    s2 = 2 * C.GAUSS_SIGMA_M ** 2
    for src, dst in (("dist", "gauss_prob"), ("win_dist_mean", "win_gauss_prob")):
        x = df[src].to_numpy()
        xmin = g[src].transform("min").to_numpy()
        e = np.exp(-(x ** 2 - xmin ** 2) / s2)  # shift by the group max for stability
        df[dst] = e / pd.Series(e).groupby(obs).transform("sum").to_numpy()

    # ---- direction agreement between GPS motion and the directed arc
    dtheta = heading[obs] - net.bearing[arc]
    hv = heading_valid[obs]
    df["heading_cos"] = np.cos(dtheta) * hv
    df["heading_sin_abs"] = np.abs(np.sin(dtheta)) * hv
    df["heading_valid"] = hv
    df["along_disp"] = hx[obs] * net.ux[arc] + hy[obs] * net.uy[arc]
    df["speed"] = speed[obs]
    df["endpoint_clip"] = ((proj_t <= 1e-6) | (proj_t >= 1 - 1e-6)).astype(float)
    df["log_arc_len"] = np.log1p(net.length[arc])
    df["from_in_deg"] = net.in_deg[net.arcs[arc, 0]].astype(float)
    df["to_out_deg"] = net.out_deg[net.arcs[arc, 1]].astype(float)
    df["has_reverse"] = (net.reverse_arc[arc] >= 0).astype(float)
    df["dt_prev"] = dt_prev[obs]
    df["dt_next"] = dt_next[obs]
    return df
