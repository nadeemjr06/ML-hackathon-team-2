"""Exploratory data analysis of the raw dataset (train + student_test partitions).

Outputs (reports/eda/):
    inventory.csv            one row per trip: counts, sizes, gaps, extent, connectivity, label quality
    data_quality.md          short record of issues / assumptions
    *.png                    distributions and trajectory plots
    map_<trip>.html          interactive Folium maps (GPS + road network + reference route)

Usage:  python scripts/01_eda.py [--maps 00000001 00000052]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from pathsnap import config as C
from pathsnap.features import prepare
from pathsnap.geo import point_segment
from pathsnap.io import _read, list_trips, load_route, load_trip
from pathsnap.labels import align_track_to_route

OUT = C.REPORTS_DIR / "eda"


def trip_stats(part: str, part_dir: Path, tid: str) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    d = part_dir / tid
    raw = _read(d / f"{tid}.track")
    trip = load_trip(part_dir, tid)
    route = load_route(part_dir, tid)
    proj, net, px, py = prepare(trip)
    tr = trip.track
    dt = np.diff(raw[:, 2])
    step = np.hypot(np.diff(px), np.diff(py))
    speed = step / np.maximum(np.diff(tr[:, 2]), 1e-6)

    n_nodes, n_arcs = len(trip.nodes), len(trip.arcs)
    g = coo_matrix((np.ones(n_arcs), (trip.arcs[:, 0], trip.arcs[:, 1])), shape=(n_nodes, n_nodes))
    n_wcc, lab_w = connected_components(g, directed=True, connection="weak")
    n_scc, lab_s = connected_components(g, directed=True, connection="strong")
    deg = net.in_deg + net.out_deg
    giant_scc = np.bincount(lab_s).max() / n_nodes

    row = {
        "partition": part, "trip_id": tid,
        "track_kb": (d / f"{tid}.track").stat().st_size / 1024,
        "nodes_kb": (d / f"{tid}.nodes").stat().st_size / 1024,
        "arcs_kb": (d / f"{tid}.arcs").stat().st_size / 1024,
        "has_route": route is not None,
        "n_points_raw": len(raw), "n_points_clean": len(tr), "n_dropped": trip.n_dropped,
        "n_nan": int((~np.isfinite(raw)).sum()),
        "non_monotonic_ts": int((dt <= 0).sum()),
        "dt_eq_1_frac": float(np.mean(dt == 1)) if len(dt) else np.nan,
        "dt_max_s": float(dt.max()) if len(dt) else np.nan,
        "n_gaps_gt_10s": int((dt > 10).sum()),
        "duration_s": float(tr[-1, 2] - tr[0, 2]),
        "track_length_km": float(step.sum() / 1000),
        "speed_p95_ms": float(np.percentile(speed, 95)) if len(speed) else np.nan,
        "n_speed_gt_70ms": int((speed > 70).sum()),
        "lon_min": tr[:, 0].min(), "lon_max": tr[:, 0].max(),
        "lat_min": tr[:, 1].min(), "lat_max": tr[:, 1].max(),
        "extent_km_x": float((px.max() - px.min()) / 1000), "extent_km_y": float((py.max() - py.min()) / 1000),
        "n_nodes": n_nodes, "n_arcs": n_arcs,
        "isolated_nodes": int((deg == 0).sum()),
        "n_weak_components": int(n_wcc), "n_strong_components": int(n_scc),
        "giant_scc_frac": float(giant_scc),
        "two_way_arc_frac": float(np.mean(net.reverse_arc >= 0)),
        "arc_len_median_m": float(np.median(net.length)),
    }
    gd = np.zeros(0)
    nearest_d = np.zeros(0)
    if route is not None:
        row["route_len"] = len(route)
        row["route_repeated_arcs"] = int(len(route) - len(np.unique(route)))
        row["route_continuity"] = float(np.mean(trip.arcs[route[:-1], 1] == trip.arcs[route[1:], 0])) if len(route) > 1 else 1.0
        row["route_length_km"] = float(net.length[route].sum() / 1000)
        gt, gd, _ = align_track_to_route(px, py, net, route)
        row["gt_dist_median_m"] = float(np.median(gd))
        row["gt_dist_p95_m"] = float(np.percentile(gd, 95))
        row["unreliable_label_frac"] = float(np.mean(gd > C.LABEL_MAX_DIST_M))
    # distance from each GPS fix to the nearest arc of the whole network
    cands = net.candidates(px, py, radius=C.CAND_MAX_RADIUS_M, max_radius=C.CAND_MAX_RADIUS_M, k=1)
    nearest_d = np.array([c[1][0] if len(c[1]) else np.inf for c in cands])
    row["nearest_arc_dist_median_m"] = float(np.median(nearest_d))
    return row, dt, gd, nearest_d


def folium_map(part_dir: Path, tid: str, path: Path):
    trip = load_trip(part_dir, tid)
    route = load_route(part_dir, tid)
    proj, net, px, py = prepare(trip)
    tr = trip.track
    m = folium.Map(location=[tr[:, 1].mean(), tr[:, 0].mean()], zoom_start=14, tiles="OpenStreetMap")
    # road network near the track (arcs within 150 m)
    near = net.candidates(px, py, radius=150, max_radius=150, k=50)
    arcs = np.unique(np.concatenate([c[0] for c in near])) if near else np.zeros(0, int)
    fg_net = folium.FeatureGroup("Road network (arcs < 150 m from GPS)")
    for seg in net.arc_lonlat(arcs):
        folium.PolyLine(seg, color="#888888", weight=2, opacity=0.6).add_to(fg_net)
    fg_net.add_to(m)
    if route is not None:
        fg_r = folium.FeatureGroup("Reference route (.route)")
        folium.PolyLine(net.route_polyline(route), color="green", weight=5, opacity=0.8).add_to(fg_r)
        fg_r.add_to(m)
    fg_g = folium.FeatureGroup("Raw GPS (.track)")
    for i in range(0, len(tr), max(1, len(tr) // 2000)):
        folium.CircleMarker([tr[i, 1], tr[i, 0]], radius=2, color="blue", fill=True,
                            tooltip=f"t={tr[i, 2]:.0f}s").add_to(fg_g)
    fg_g.add_to(m)
    folium.Marker([tr[0, 1], tr[0, 0]], tooltip="start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker([tr[-1, 1], tr[-1, 0]], tooltip="end", icon=folium.Icon(color="red")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[tr[:, 1].min(), tr[:, 0].min()], [tr[:, 1].max(), tr[:, 0].max()]])
    m.save(str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", nargs="*", default=["00000001", "00000054"],
                    help="train trip ids to render as Folium maps")
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    jobs = [("train", C.TRAIN_DIR, t) for t in list_trips(C.TRAIN_DIR)] + \
           [("student_test", C.STUDENT_TEST_DIR, t) for t in list_trips(C.STUDENT_TEST_DIR)]
    print(f"{len(jobs)} trips: train={sum(j[0]=='train' for j in jobs)} student_test={sum(j[0]=='student_test' for j in jobs)}")
    res = Parallel(n_jobs=args.jobs, verbose=5)(delayed(trip_stats)(*j) for j in jobs)
    inv = pd.DataFrame([r[0] for r in res])
    inv.to_csv(OUT / "inventory.csv", index=False)
    dts = np.concatenate([r[1] for r in res])
    gds = np.concatenate([r[2] for r in res if r[0]["partition"] == "train"])
    nds = np.concatenate([r[3] for r in res if r[0]["partition"] == "train"])

    summ = inv.groupby("partition").agg(
        trips=("trip_id", "count"), points=("n_points_clean", "sum"),
        points_median=("n_points_clean", "median"), nan_values=("n_nan", "sum"),
        dropped=("n_dropped", "sum"), gaps_gt_10s=("n_gaps_gt_10s", "sum"),
        nodes_median=("n_nodes", "median"), arcs_median=("n_arcs", "median"),
        isolated_nodes=("isolated_nodes", "sum"), route_cont_mean=("route_continuity", "mean"),
        unreliable_label_frac_mean=("unreliable_label_frac", "mean"))
    summ.to_csv(OUT / "partition_summary.csv")
    print(summ.T)

    # ---- plots (train partition unless stated)
    tr = inv[inv.partition == "train"]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    ax[0, 0].hist(tr.n_points_clean, bins=30); ax[0, 0].set_title("GPS points per trip (train)")
    ax[0, 1].hist(np.clip(dts, 0, 20), bins=40, log=True); ax[0, 1].set_title("Sampling interval s (all, clipped 20)")
    ax[0, 2].hist(np.clip(gds, 0, 100), bins=50, log=True); ax[0, 2].axvline(C.LABEL_MAX_DIST_M, c="r")
    ax[0, 2].set_title("GPS -> aligned reference arc distance m (train)")
    ax[1, 0].hist(np.clip(nds, 0, 100), bins=50, log=True); ax[1, 0].axvline(C.CAND_RADIUS_M, c="r")
    ax[1, 0].set_title("GPS -> nearest arc distance m (train)")
    ax[1, 1].scatter(tr.n_nodes, tr.n_arcs, s=10); ax[1, 1].set_xscale("log"); ax[1, 1].set_yscale("log")
    ax[1, 1].set_title("Network size: nodes vs arcs (train)")
    ax[1, 2].hist(tr.unreliable_label_frac, bins=20); ax[1, 2].set_title("Fraction of obs > 50 m from reference")
    plt.tight_layout(); plt.savefig(OUT / "distributions.png", dpi=120); plt.close()

    plt.figure(figsize=(10, 5))
    plt.scatter(inv.lon_min, inv.lat_min, c=(inv.partition == "train"), cmap="coolwarm", s=15)
    plt.xlabel("lon"); plt.ylabel("lat"); plt.title("Trip locations (red=train, blue=student_test)")
    plt.savefig(OUT / "trip_locations.png", dpi=120); plt.close()

    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    for a, tid in zip(axs.ravel(), tr.sort_values("n_points_clean").trip_id.iloc[::10][:8]):
        trip = load_trip(C.TRAIN_DIR, tid)
        proj, net, px, py = prepare(trip)
        a.plot(px / 1000, py / 1000, ".", ms=1); a.set_title(tid); a.set_aspect("equal")
    plt.suptitle("GPS trajectories in local projected km (train)"); plt.tight_layout()
    plt.savefig(OUT / "trajectories.png", dpi=120); plt.close()

    for tid in args.maps:
        folium_map(C.TRAIN_DIR, tid, OUT / f"map_{tid}.html")
        print("map ->", OUT / f"map_{tid}.html")

    with open(OUT / "data_quality.md", "w") as f:
        f.write("# Data-quality record (generated by scripts/01_eda.py)\n\n")
        f.write(summ.T.to_markdown() + "\n\n")
        f.write(f"- Sampling intervals equal to 1 s: {np.mean(dts == 1):.3%}; max gap {dts.max():.0f} s.\n")
        f.write(f"- Train GPS->reference distance: median {np.median(gds):.1f} m, p95 {np.percentile(gds,95):.1f} m.\n")
        f.write(f"- Train GPS->nearest arc: median {np.median(nds):.1f} m, p99 {np.percentile(nds[np.isfinite(nds)],99):.1f} m, "
                f"> {C.CAND_RADIUS_M:.0f} m: {np.mean(nds > C.CAND_RADIUS_M):.3%}.\n")
        bad = tr.sort_values("unreliable_label_frac", ascending=False).head(10)
        f.write("\nTrips with the largest share of observations > 50 m from their aligned reference arc "
                "(reference route truncated / GPS off-network):\n\n")
        f.write(bad[["trip_id", "n_points_clean", "route_length_km", "track_length_km", "unreliable_label_frac"]].to_markdown(index=False))
        f.write("\n")
    print("EDA written to", OUT)


if __name__ == "__main__":
    main()
