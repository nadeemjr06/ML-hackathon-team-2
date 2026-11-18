"""Deriving per-observation reference labels from the reference route.

The ``.route`` file gives the chronological sequence of arcs, *not* one arc
per GPS row (route length != track length). To obtain an observation-level
reference arc we align the GPS sequence to the route sequence with a monotonic
dynamic programme:

    cost[i, j] = d(p_i, arc_j) + min_{j' <= j} cost[i-1, j']

i.e. each GPS point is assigned to one route position, positions never go
backwards in time (which correctly handles routes that visit the same arc
twice), and arcs may be skipped (short arcs can be traversed between two 1 Hz
fixes). The total point-to-arc distance is minimised.

This is only used for (a) building training labels and (b) server-side
evaluation. It is never called by the prediction pipeline.
"""
from __future__ import annotations

import numpy as np

from .geo import point_segment
from .network import RoadNetwork


def align_track_to_route(px: np.ndarray, py: np.ndarray, net: RoadNetwork, route: np.ndarray):
    """Return (gt_arc per point, distance to that arc in m, route position per point)."""
    route = np.asarray(route, dtype=int)
    n, m = len(px), len(route)
    ax, ay, bx, by = net.ax[route], net.ay[route], net.bx[route], net.by[route]
    back = np.empty((n, m), dtype=np.int16 if m < 32000 else np.int32)
    prev = None
    for i in range(n):
        d, _ = point_segment(px[i], py[i], ax, ay, bx, by)
        if prev is None:
            cost = d
            back[i] = np.arange(m)
        else:
            # running prefix minimum of prev and its argmin
            pm = np.minimum.accumulate(prev)
            is_new = np.r_[True, prev[1:] < pm[:-1]]
            arg = np.maximum.accumulate(np.where(is_new, np.arange(m), 0))
            cost = d + pm
            back[i] = arg
        prev = cost
    pos = np.empty(n, dtype=int)
    pos[-1] = int(np.argmin(prev))
    for i in range(n - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]
    gt = route[pos]
    dist, _ = point_segment(px, py, net.ax[gt], net.ay[gt], net.bx[gt], net.by[gt])
    return gt, dist, pos
