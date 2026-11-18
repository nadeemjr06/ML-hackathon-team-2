"""Reading raw trip folders.

Layout (verified during EDA):
    <partition>/<trip_id>/<trip_id>.track   lon lat t        (whitespace, no header)
    <partition>/<trip_id>/<trip_id>.nodes   lon lat          (row index = node_id)
    <partition>/<trip_id>/<trip_id>.arcs    from to          (row index = arc_id)
    <partition>/<trip_id>/<trip_id>.route   arc_id           (chronological reference route)

The prediction pipeline only ever calls ``load_trip`` (no route). The
reference route is read separately via ``load_route`` by evaluation code.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray          # (N,3) lon, lat, t  -- cleaned
    nodes: np.ndarray          # (M,2) lon, lat
    arcs: np.ndarray           # (A,2) int from, to
    n_raw_points: int = 0
    n_dropped: int = 0


def list_trips(part_dir: Path) -> list[str]:
    part_dir = Path(part_dir)
    return sorted(p.name for p in part_dir.iterdir() if p.is_dir() and (p / f"{p.name}.track").exists())


def _read(path: Path, dtype=float) -> np.ndarray:
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="c")
    return df.to_numpy(dtype=dtype)


def clean_track(track: np.ndarray) -> tuple[np.ndarray, int]:
    """Drop non-finite / implausible coordinates, sort by time, drop duplicate timestamps."""
    n0 = len(track)
    ok = np.isfinite(track).all(axis=1)
    ok &= (np.abs(track[:, 0]) <= 180) & (np.abs(track[:, 1]) <= 90)
    ok &= ~((track[:, 0] == 0) & (track[:, 1] == 0))
    track = track[ok]
    order = np.argsort(track[:, 2], kind="stable")
    track = track[order]
    keep = np.r_[True, np.diff(track[:, 2]) > 0]
    track = track[keep]
    return track, n0 - len(track)


def load_trip(part_dir: Path, trip_id: str) -> Trip:
    d = Path(part_dir) / trip_id
    raw = _read(d / f"{trip_id}.track")
    track, dropped = clean_track(raw)
    nodes = _read(d / f"{trip_id}.nodes")
    arcs = _read(d / f"{trip_id}.arcs", dtype=np.int64)
    return Trip(trip_id, track, nodes, arcs, n_raw_points=len(raw), n_dropped=dropped)


def load_route(part_dir: Path, trip_id: str) -> np.ndarray | None:
    p = Path(part_dir) / trip_id / f"{trip_id}.route"
    if not p.exists():
        return None
    return _read(p, dtype=np.int64).ravel()
