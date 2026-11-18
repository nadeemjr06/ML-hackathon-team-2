"""HMM / Viterbi route reconstruction.

Hidden state at observation i : one of the candidate (directed) arcs of i,
                                together with the projection point on that arc.
Emission  log b_i(s)            : w_e * log P_clf(s | features), where P_clf is the
                                  classifier probability renormalised over the
                                  candidates of observation i (floored at p_min).
Transition log a(s_{i-1}->s_i)  : -|d_route - d_gc| / beta_eff  (Newson & Krumm 2009)
                                  d_route = network distance between the two
                                  projection points along directed arcs (Dijkstra),
                                  d_gc    = straight-line distance between the GPS fixes.
                                  beta_eff = beta * sqrt(max(dt, 1)) widens the
                                  distribution for GPS gaps.
                                  A U-turn onto the reverse arc pays ``uturn_penalty``.
                                  Unreachable pairs (no path within the Dijkstra
                                  cutoff) get ``unreachable_logp`` instead of -inf,
                                  so the decoder never breaks and can recover.
Observations without any candidate are skipped by the decoder (they inherit
the arc of the previous decoded observation in the output).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .network import RoadNetwork


@dataclass
class HMMConfig:
    beta: float = 5.0
    emission_weight: float = 1.0
    p_min: float = 1e-4
    uturn_penalty: float = 3.0
    unreachable_logp: float = -30.0
    cutoff_factor: float = 2.0
    cutoff_min_m: float = 150.0
    cutoff_max_m: float = 5000.0

    def to_dict(self):
        return asdict(self)


def _route_distance(net: RoadNetwork, a: int, ta: float, b: int, tb: float, cutoff: float):
    La, Lb = net.length[a], net.length[b]
    if a == b:
        delta = (tb - ta) * La
        if delta >= -5.0:          # forward (or GPS jitter backwards up to 5 m)
            return abs(delta)
    rem_a = (1.0 - ta) * La
    dists = net.dists_from(int(net.arcs[a, 1]), cutoff)
    d = dists.get(int(net.arcs[b, 0]))
    if d is None:
        return None
    return rem_a + d + tb * Lb


def viterbi(net: RoadNetwork, px, py, t, cand_arcs: list, cand_t: list, cand_p: list,
            cfg: HMMConfig) -> np.ndarray:
    """Return decoded arc id per observation (-1 if no candidate anywhere before)."""
    n = len(px)
    valid = [i for i in range(n) if len(cand_arcs[i])]
    out = np.full(n, -1, dtype=int)
    if not valid:
        return out

    def emis(i):
        p = np.asarray(cand_p[i], dtype=float)
        p = p / max(p.sum(), 1e-12)
        return cfg.emission_weight * np.log(np.maximum(p, cfg.p_min))

    deltas = [emis(valid[0])]
    backs = [None]
    for k in range(1, len(valid)):
        i_prev, i = valid[k - 1], valid[k]
        dgc = math.hypot(px[i] - px[i_prev], py[i] - py[i_prev])
        dt = max(t[i] - t[i_prev], 1.0)
        beta_eff = cfg.beta * math.sqrt(dt)
        cutoff = min(max(cfg.cutoff_factor * dgc + cfg.cutoff_min_m, cfg.cutoff_min_m), cfg.cutoff_max_m)
        A_prev, T_prev = cand_arcs[i_prev], cand_t[i_prev]
        A_cur, T_cur = cand_arcs[i], cand_t[i]
        trans = np.empty((len(A_prev), len(A_cur)))
        for a_i, (a, ta) in enumerate(zip(A_prev, T_prev)):
            rev = net.reverse_arc[a]
            for b_i, (b, tb) in enumerate(zip(A_cur, T_cur)):
                dr = _route_distance(net, int(a), float(ta), int(b), float(tb), cutoff)
                if dr is None:
                    trans[a_i, b_i] = cfg.unreachable_logp
                else:
                    lp = -abs(dr - dgc) / beta_eff
                    if b == rev:
                        lp -= cfg.uturn_penalty
                    trans[a_i, b_i] = lp
        score = deltas[-1][:, None] + trans
        bp = np.argmax(score, axis=0)
        deltas.append(score[bp, np.arange(len(A_cur))] + emis(i))
        backs.append(bp)

    # backtrack
    s = int(np.argmax(deltas[-1]))
    states = [0] * len(valid)
    for k in range(len(valid) - 1, -1, -1):
        states[k] = s
        if k > 0:
            s = int(backs[k][s])
    for k, i in enumerate(valid):
        out[i] = int(cand_arcs[i][states[k]])
    # fill observations without candidates with the previous decoded arc
    last = -1
    for i in range(n):
        if out[i] >= 0:
            last = out[i]
        else:
            out[i] = last
    if out[0] < 0:
        first = out[out >= 0][0]
        out[out < 0] = first
    return out
