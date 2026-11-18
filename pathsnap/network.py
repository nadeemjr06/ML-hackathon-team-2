"""Road-network representation for one trip.

Arcs are *directed* (both directions of a two-way street appear as two arcs),
so the hidden state of the HMM is a directed arc and travel direction matters.
"""
from __future__ import annotations

import heapq
import math

import numpy as np
from scipy.spatial import cKDTree

from . import config as C
from .geo import LocalProjection, point_segment


class RoadNetwork:
    def __init__(self, nodes_lonlat: np.ndarray, arcs: np.ndarray, proj: LocalProjection,
                 focus_xy: np.ndarray | None = None, focus_margin_m: float = 1500.0):
        self.proj = proj
        self.nodes_lonlat = nodes_lonlat
        self.arcs = arcs.astype(np.int64)
        self.n_nodes = len(nodes_lonlat)
        self.n_arcs = len(arcs)
        x, y = proj.to_xy(nodes_lonlat[:, 0], nodes_lonlat[:, 1])
        self.nx, self.ny = x, y
        u, v = self.arcs[:, 0], self.arcs[:, 1]
        self.ax, self.ay, self.bx, self.by = x[u], y[u], x[v], y[v]
        dx, dy = self.bx - self.ax, self.by - self.ay
        self.length = np.hypot(dx, dy)
        self.bearing = np.arctan2(dx, dy)  # compass-style angle (0 = north)
        self.ux = np.where(self.length > 0, dx / np.maximum(self.length, 1e-9), 0.0)
        self.uy = np.where(self.length > 0, dy / np.maximum(self.length, 1e-9), 0.0)

        self.out_deg = np.bincount(u, minlength=self.n_nodes)
        self.in_deg = np.bincount(v, minlength=self.n_nodes)

        key = u * self.n_nodes + v
        rev_key = v * self.n_nodes + u
        order = np.argsort(key)
        pos = np.searchsorted(key[order], rev_key)
        pos = np.clip(pos, 0, len(key) - 1)
        found = key[order][pos] == rev_key
        self.reverse_arc = np.where(found, order[pos], -1)

        # CSR adjacency (node -> outgoing arcs) for Dijkstra
        o = np.argsort(u, kind="stable")
        self._out_arcs = o
        self._out_ptr = np.r_[0, np.cumsum(self.out_deg)]
        self._adj: list | None = None
        self._sp_cache: dict[int, tuple] = {}

        self._build_index(focus_xy, focus_margin_m)

    # ------------------------------------------------------------------ index
    def _build_index(self, focus_xy, margin):
        if focus_xy is not None and len(focus_xy):
            xmin, ymin = focus_xy.min(axis=0) - margin
            xmax, ymax = focus_xy.max(axis=0) + margin
            m = ~((np.maximum(self.ax, self.bx) < xmin) | (np.minimum(self.ax, self.bx) > xmax) |
                  (np.maximum(self.ay, self.by) < ymin) | (np.minimum(self.ay, self.by) > ymax))
            idx_arcs = np.flatnonzero(m)
        else:
            idx_arcs = np.arange(self.n_arcs)
        step = C.KD_SAMPLE_STEP_M
        n_s = np.maximum(1, np.ceil(self.length[idx_arcs] / step).astype(int)) + 1
        rep = np.repeat(idx_arcs, n_s)
        frac = np.concatenate([np.linspace(0, 1, k) for k in n_s]) if len(n_s) else np.zeros(0)
        sx = self.ax[rep] + frac * (self.bx[rep] - self.ax[rep])
        sy = self.ay[rep] + frac * (self.by[rep] - self.ay[rep])
        self._sample_arc = rep
        self._tree = cKDTree(np.c_[sx, sy]) if len(rep) else None
        self.indexed_arcs = idx_arcs

    # ------------------------------------------------------------ candidates
    def candidates(self, px: np.ndarray, py: np.ndarray, radius: float = C.CAND_RADIUS_M,
                   max_radius: float = C.CAND_MAX_RADIUS_M, k: int = C.CAND_MAX_K):
        """Return list (per point) of (arc_ids, dists, ts) sorted by distance.

        The radius is doubled for points with no candidate, up to ``max_radius``.
        """
        n = len(px)
        out: list = [None] * n
        todo = np.arange(n)
        r = radius
        while len(todo) and self._tree is not None:
            res = self._query(px[todo], py[todo], r, k)
            nxt = []
            for j, i in enumerate(todo):
                if len(res[j][0]) or r >= max_radius:
                    out[i] = res[j]
                else:
                    nxt.append(i)
            todo = np.array(nxt, dtype=int)
            r = min(r * 2, max_radius) if r < max_radius else r * 2
        empty = (np.zeros(0, int), np.zeros(0), np.zeros(0))
        return [o if o is not None else empty for o in out]

    def _query(self, px, py, r, k):
        step = C.KD_SAMPLE_STEP_M
        lists = self._tree.query_ball_point(np.c_[px, py], r + step / 2 + 1e-6)
        lens = np.fromiter((len(l) for l in lists), int, len(lists))
        pid = np.repeat(np.arange(len(lists)), lens)
        if len(pid) == 0:
            return [(np.zeros(0, int), np.zeros(0), np.zeros(0)) for _ in lists]
        sidx = np.fromiter((s for l in lists for s in l), int, lens.sum())
        arc = self._sample_arc[sidx]
        pair = np.unique(pid * self.n_arcs + arc)
        pid, arc = pair // self.n_arcs, pair % self.n_arcs
        d, t = point_segment(px[pid], py[pid], self.ax[arc], self.ay[arc], self.bx[arc], self.by[arc])
        keep = d <= r
        pid, arc, d, t = pid[keep], arc[keep], d[keep], t[keep]
        o = np.lexsort((d, pid))
        pid, arc, d, t = pid[o], arc[o], d[o], t[o]
        bounds = np.searchsorted(pid, np.arange(len(lists) + 1))
        res = []
        for j in range(len(lists)):
            s, e = bounds[j], min(bounds[j + 1], bounds[j] + k)
            res.append((arc[s:e], d[s:e], t[s:e]))
        return res

    # --------------------------------------------------------- shortest paths
    def _ensure_adj(self):
        if self._adj is None:
            v = self.arcs[:, 1]
            L = self.length
            adj = [None] * self.n_nodes
            o, p = self._out_arcs, self._out_ptr
            for n in range(self.n_nodes):
                a = o[p[n]:p[n + 1]]
                adj[n] = list(zip(v[a].tolist(), L[a].tolist(), a.tolist()))
            self._adj = adj

    def dists_from(self, src: int, cutoff: float) -> dict:
        """Dijkstra network distances (m) from node ``src`` up to ``cutoff`` (cached)."""
        return self._dijkstra(src, cutoff)[0]

    def _dijkstra(self, src: int, cutoff: float):
        c = self._sp_cache.get(src)
        if c is not None and c[0] >= cutoff:
            return c[1], c[2]
        self._ensure_adj()
        adj = self._adj
        dist = {src: 0.0}
        pred: dict[int, int] = {}          # node -> arc used to reach it
        heap = [(0.0, src)]
        while heap:
            d, n = heapq.heappop(heap)
            if d > dist.get(n, math.inf):
                continue
            for m, w, a in adj[n]:
                nd = d + w
                if nd <= cutoff and nd < dist.get(m, math.inf):
                    dist[m] = nd
                    pred[m] = a
                    heapq.heappush(heap, (nd, m))
        self._sp_cache[src] = (cutoff, dist, pred)
        return dist, pred

    def path_arcs(self, src: int, dst: int, cutoff: float = 5000.0) -> list[int] | None:
        """Arcs of the shortest path from node src to node dst (None if unreachable)."""
        if src == dst:
            return []
        dist, pred = self._dijkstra(src, cutoff)
        if dst not in dist:
            return None
        out = []
        n = dst
        while n != src:
            a = pred[n]
            out.append(int(a))
            n = int(self.arcs[a, 0])
        return out[::-1]

    def fill_route(self, arc_seq: list[int], cutoff: float = 5000.0) -> list[int]:
        """Insert shortest-path arcs between consecutive arcs that are not directly connected."""
        if not arc_seq:
            return []
        out = [int(arc_seq[0])]
        for b in arc_seq[1:]:
            a = out[-1]
            u, v = int(self.arcs[a, 1]), int(self.arcs[b, 0])
            if u != v:
                mid = self.path_arcs(u, v, cutoff)
                if mid:
                    out.extend(mid)
            out.append(int(b))
        return out

    def arc_lonlat(self, arc_ids) -> list[list[list[float]]]:
        """[[lat, lon], [lat, lon]] per arc (Leaflet/Folium order)."""
        a = np.asarray(arc_ids, dtype=int)
        u, v = self.arcs[a, 0], self.arcs[a, 1]
        return [[[float(self.nodes_lonlat[i, 1]), float(self.nodes_lonlat[i, 0])],
                 [float(self.nodes_lonlat[j, 1]), float(self.nodes_lonlat[j, 0])]] for i, j in zip(u, v)]

    def route_polyline(self, arc_seq) -> list[list[float]]:
        """Connected [lat, lon] polyline for a sequence of arcs (breaks are kept as jumps)."""
        pts: list[list[float]] = []
        for a in arc_seq:
            u, v = self.arcs[a]
            pu = [float(self.nodes_lonlat[u, 1]), float(self.nodes_lonlat[u, 0])]
            pv = [float(self.nodes_lonlat[v, 1]), float(self.nodes_lonlat[v, 0])]
            if not pts or pts[-1] != pu:
                pts.append(pu)
            pts.append(pv)
        return pts
