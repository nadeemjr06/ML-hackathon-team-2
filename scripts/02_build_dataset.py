"""Candidate generation + feature engineering + label derivation for train and student_test.

Outputs (data_cache/):
    <part>_cands.parquet   one row per (observation, candidate arc): features + label
    <part>_obs.parquet     one row per observation: gt arc, gt distance, reliability, coverage

Usage:  python scripts/02_build_dataset.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from pathsnap import config as C
from pathsnap.features import build_features, prepare
from pathsnap.io import list_trips, load_route, load_trip
from pathsnap.labels import align_track_to_route


def process(part_dir: Path, tid: str):
    t0 = time.time()
    trip = load_trip(part_dir, tid)
    proj, net, px, py = prepare(trip)
    df = build_features(trip, net, px, py)
    route = load_route(part_dir, tid)
    gt, gd, _ = align_track_to_route(px, py, net, route)
    df["label"] = (df["arc_id"].to_numpy() == gt[df["obs_idx"].to_numpy()]).astype(np.int8)
    df.insert(0, "trip_id", tid)
    cover = np.zeros(len(px), dtype=bool)
    cover[df.loc[df.label == 1, "obs_idx"].to_numpy()] = True
    obs = pd.DataFrame({"trip_id": tid, "obs_idx": np.arange(len(px)), "t": trip.track[:, 2],
                        "gt_arc": gt, "gt_dist": gd, "reliable": gd <= C.LABEL_MAX_DIST_M,
                        "gt_in_cands": cover})
    df["reliable"] = obs["reliable"].to_numpy()[df["obs_idx"].to_numpy()]
    return df, obs, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for part, d in (("train", C.TRAIN_DIR), ("student_test", C.STUDENT_TEST_DIR)):
        trips = list_trips(d)
        res = Parallel(n_jobs=args.jobs, verbose=2)(delayed(process)(d, t) for t in trips)
        cands = pd.concat([r[0] for r in res], ignore_index=True)
        obs = pd.concat([r[1] for r in res], ignore_index=True)
        float_cols = [c for c in C.FEATURES if cands[c].dtype == np.float64]
        cands[float_cols] = cands[float_cols].astype(np.float32)
        cands.to_parquet(C.CACHE_DIR / f"{part}_cands.parquet", index=False)
        obs.to_parquet(C.CACHE_DIR / f"{part}_obs.parquet", index=False)
        rel = obs[obs.reliable]
        print(f"[{part}] trips={len(trips)} obs={len(obs)} cand_rows={len(cands)} "
              f"pos_rate={cands.label.mean():.3f} cands/obs={len(cands)/len(obs):.2f} "
              f"reliable={obs.reliable.mean():.3%} miss_rate(all)={1-obs.gt_in_cands.mean():.3%} "
              f"miss_rate(reliable)={1-rel.gt_in_cands.mean():.3%} time={sum(r[2] for r in res):.0f}s")


if __name__ == "__main__":
    main()
