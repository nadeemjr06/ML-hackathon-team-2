"""HMM / Viterbi decoder: parameter tuning and classifier-only vs classifier+HMM comparison.

Tuning   : grid search over (beta, emission_weight, uturn_penalty) on a seeded subset
           of *training* trips, using the final model's out-of-fold probabilities
           (so emissions are not over-confident on trips the model was fitted on).
           Objective = 0.5 * observation accuracy (reliable obs, pooled)
                     + 0.5 * mean route F1 of the output (filled) route,
           so that both the pointwise and the sequence-level goal are optimised.
Evaluation: student-test trips, classifier-only argmax vs Viterbi decoding, for the
           final model and every model from scripts/03 (decoder compatibility),
           plus a non-learned distance-only emission baseline.

Outputs: models/hmm_config.json, results/hmm_grid.csv, results/hmm_student_test_per_trip.csv,
         results/hmm_all_models.csv, results/plots/hmm_vs_classifier.png

Usage:  python scripts/05_hmm_eval.py [--tune-trips 30]
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from pathsnap import config as C
from pathsnap.data import load_part
from pathsnap.evaluate import group_lists
from pathsnap.features import prepare
from pathsnap.hmm import HMMConfig, viterbi
from pathsnap.io import load_route, load_trip
from pathsnap.metrics import argmax_arcs, compress, route_continuity, sequence_metrics

GRID = {"beta": [5.0, 10.0, 20.0, 40.0, 80.0], "emission_weight": [1.0, 2.0, 4.0, 8.0], "uturn_penalty": [0.0, 5.0]}


def tune_trip(tid, trip_c, p, obs, configs):
    trip = load_trip(C.TRAIN_DIR, tid)
    route = load_route(C.TRAIN_DIR, tid)
    _, net, px, py = prepare(trip)
    n = len(px)
    arcs, ts, ps = group_lists(trip_c, p, n)
    gt = obs["gt_arc"].to_numpy()
    rel = obs["reliable"].to_numpy()
    clf = argmax_arcs(trip_c, p, n)
    out = [("classifier_only", int((clf[rel] == gt[rel]).sum()), int(rel.sum()),
            sequence_metrics(net, clf, route)["route_f1"])]
    for cfg in configs:
        path = viterbi(net, px, py, trip.track[:, 2], arcs, ts, ps, cfg)
        filled = np.array(net.fill_route(compress(path)))
        out.append((json.dumps(cfg.to_dict()), int((path[rel] == gt[rel]).sum()), int(rel.sum()),
                    sequence_metrics(net, filled, route)["route_f1"]))
    return out


def eval_trip(tid, trip_c, preds: dict, obs, cfg):
    trip = load_trip(C.STUDENT_TEST_DIR, tid)
    route = load_route(C.STUDENT_TEST_DIR, tid)
    _, net, px, py = prepare(trip)
    n = len(px)
    gt = obs["gt_arc"].to_numpy()
    rel = obs["reliable"].to_numpy()
    rows = []
    for name, p in preds.items():
        arcs, ts, ps = group_lists(trip_c, p, n)
        clf = argmax_arcs(trip_c, p, n)
        hmm = viterbi(net, px, py, trip.track[:, 2], arcs, ts, ps, cfg)
        filled = np.array(net.fill_route(compress(hmm)))
        for dec, pred, route_seq in (("classifier_only", clf, clf), ("classifier+hmm", hmm, filled)):
            # per-observation accuracy uses the per-observation arcs; route metrics use the output
            # route (for the HMM: decoded arcs + shortest-path fill between non-adjacent arcs)
            r = {"model": name, "decoder": dec, "trip_id": tid, "n_obs": n,
                 "obs_accuracy": float(np.mean(pred[rel] == gt[rel])),
                 "obs_accuracy_all": float(np.mean(pred == gt)),
                 "route_continuity_unfilled": route_continuity(net, pred),
                 "route_continuity": route_continuity(net, route_seq)}
            r.update(sequence_metrics(net, route_seq, route))
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune-trips", type=int, default=30)
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()
    RES, PLOTS = C.RESULTS_DIR, C.RESULTS_DIR / "plots"
    PLOTS.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ tuning
    tr_c, tr_o = load_part("train")
    oof = np.load(C.CACHE_DIR / "train_oof_final.npy")
    rng = np.random.default_rng(C.SEED)
    trips = sorted(tr_c.trip_id.unique())
    tune = sorted(rng.choice(trips, size=min(args.tune_trips, len(trips)), replace=False))
    configs = [HMMConfig(beta=b, emission_weight=w, uturn_penalty=u)
               for b, w, u in itertools.product(GRID["beta"], GRID["emission_weight"], GRID["uturn_penalty"])]
    jobs = []
    for tid in tune:
        m = (tr_c.trip_id == tid).to_numpy()
        jobs.append(delayed(tune_trip)(tid, tr_c[m].reset_index(drop=True), oof[m],
                                       tr_o[tr_o.trip_id == tid].reset_index(drop=True), configs))
    res = Parallel(n_jobs=args.jobs, verbose=5)(jobs)
    agg: dict = {}
    for trip_res in res:
        for key, c, t, f1 in trip_res:
            a = agg.setdefault(key, [0, 0, []])
            a[0] += c; a[1] += t; a[2].append(f1)
    grid = pd.DataFrame([{"config": k, "accuracy": v[0] / v[1], "route_f1": float(np.mean(v[2]))}
                         for k, v in agg.items()])
    grid["objective"] = 0.5 * grid.accuracy + 0.5 * grid.route_f1
    grid = grid.sort_values("objective", ascending=False)
    grid.to_csv(RES / "hmm_grid.csv", index=False)
    best_key = grid[grid.config != "classifier_only"].iloc[0].config
    best = HMMConfig(**json.loads(best_key))
    print(grid.head(10).to_string(index=False))
    print("classifier-only on tuning trips:", grid[grid.config == "classifier_only"].to_dict("records"))
    (C.MODELS_DIR / "hmm_config.json").write_text(json.dumps(best.to_dict(), indent=2))
    print("best HMM config:", best)

    # ----------------------------------------------- student-test comparison
    st_c, st_o = load_part("student_test")
    preds = {"final_tuned": np.load(RES / "preds" / "final_student_test.npy"),
             "distance_only_baseline": st_c["gauss_prob"].to_numpy()}
    for f in sorted((RES / "preds").glob("*_student_test.npy")):
        name = f.name.replace("_student_test.npy", "")
        if name != "final":
            preds[name] = np.load(f)
    jobs = []
    for tid in sorted(st_c.trip_id.unique()):
        m = (st_c.trip_id == tid).to_numpy()
        jobs.append(delayed(eval_trip)(tid, st_c[m].reset_index(drop=True), {k: v[m] for k, v in preds.items()},
                                       st_o[st_o.trip_id == tid].reset_index(drop=True), best))
    rows = [r for trip_rows in Parallel(n_jobs=args.jobs)(jobs) for r in trip_rows]
    per = pd.DataFrame(rows)
    per[per.model == "final_tuned"].to_csv(RES / "hmm_student_test_per_trip.csv", index=False)
    # observation-weighted accuracy, trip-averaged route metrics
    def summarise(g):
        w = g.n_obs
        return pd.Series({"obs_accuracy": np.average(g.obs_accuracy, weights=w),
                          "obs_accuracy_all": np.average(g.obs_accuracy_all, weights=w),
                          "route_continuity": g.route_continuity.mean(),
                          "route_continuity_unfilled": g.route_continuity_unfilled.mean(),
                          "route_precision": g.route_precision.mean(), "route_recall": g.route_recall.mean(),
                          "route_f1": g.route_f1.mean(), "route_edit_similarity": g.route_edit_similarity.mean()})
    summ = per.groupby(["model", "decoder"]).apply(summarise, include_groups=False).reset_index()
    summ.to_csv(RES / "hmm_all_models.csv", index=False)
    print(summ.to_string(index=False))

    fin = per[per.model == "final_tuned"].pivot(index="trip_id", columns="decoder", values=["obs_accuracy", "route_continuity"])
    fig, axs = plt.subplots(1, 2, figsize=(14, 4))
    fin["obs_accuracy"].plot.bar(ax=axs[0]); axs[0].set_title("Student-test obs accuracy per trip"); axs[0].set_ylim(0.5, 1)
    fin["route_continuity"].plot.bar(ax=axs[1]); axs[1].set_title("Route continuity per trip")
    plt.tight_layout(); plt.savefig(PLOTS / "hmm_vs_classifier.png", dpi=120); plt.close()


if __name__ == "__main__":
    main()
