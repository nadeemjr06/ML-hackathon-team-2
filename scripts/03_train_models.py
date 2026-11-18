"""Train and compare >= 5 candidate classifiers.

Protocol
  * 5-fold GroupKFold on the 80 training trips (groups = trip_id, no trip is split
    across folds) -> CV Macro-F1 / Top-1 / Log-loss (mean and std across folds).
  * Refit on all 80 training trips -> evaluate on the 10 student-testing trips.
  * Record configuration, seed, training time, inference time.

Outputs (results/):
  model_comparison.csv, cv_folds.csv, experiment_log.jsonl,
  preds/<model>_student_test.npy, plots/*.png

Usage:  python scripts/03_train_models.py [--models logreg lightgbm ...] [--folds 5]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from pathsnap import config as C
from pathsnap.data import load_part, subsample_obs, training_mask
from pathsnap.evaluate import eval_probs
from pathsnap.features import prepare
from pathsnap.io import load_trip
from pathsnap.metrics import argmax_arcs, route_continuity
from pathsnap.models import REGISTRY, build

RES = C.RESULTS_DIR
PLOTS = RES / "plots"
PREDS = RES / "preds"


def fit_model(name, X, y, cands_fit, seed=C.SEED, params=None):
    model, family, p, cap = build(name, params)
    idx = subsample_obs(cands_fit, cap, seed)
    t0 = time.time()
    model.fit(X[idx], y[idx])
    return model, family, p, cap, time.time() - t0


def student_continuity(st_cands, p, st_obs):
    vals = []
    for tid, g in st_cands.groupby("trip_id", sort=False):
        trip = load_trip(C.STUDENT_TEST_DIR, tid)
        _, net, px, _ = prepare(trip)
        pred = argmax_arcs(g, p[g.index.to_numpy()], len(px))
        vals.append(route_continuity(net, pred))
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(REGISTRY))
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    for d in (RES, PLOTS, PREDS):
        d.mkdir(parents=True, exist_ok=True)

    tr_c, tr_o = load_part("train")
    st_c, st_o = load_part("student_test")
    X = tr_c[C.FEATURES].to_numpy(np.float32)
    y = tr_c["label"].to_numpy()
    fit_mask = training_mask(tr_c, tr_o)
    groups = tr_c["trip_id"].to_numpy()
    Xst = st_c[C.FEATURES].to_numpy(np.float32)
    print(f"train rows={len(tr_c)} (fit rows={fit_mask.sum()}), student-test rows={len(st_c)}")

    rows, fold_rows = [], []
    log = open(RES / "experiment_log.jsonl", "a")
    gkf = GroupKFold(n_splits=args.folds)
    for name in args.models:
        print(f"\n=== {name}")
        cv = []
        for k, (tri, vai) in enumerate(gkf.split(X, y, groups)):
            tri = tri[fit_mask[tri]]
            model, family, params, cap, ttrain = fit_model(name, X[tri], y[tri], tr_c.iloc[tri])
            vc = tr_c.iloc[vai]
            vo = tr_o[tr_o.trip_id.isin(vc.trip_id.unique())]
            pv = model.predict_proba(X[vai])[:, 1]
            m = eval_probs(vc, vo, pv)
            cv.append(m)
            fold_rows.append({"model": name, "fold": k, "f1_macro": m["f1_macro"], "top1_acc": m["top1_acc"],
                              "log_loss": m["log_loss"], "train_time_s": ttrain})
            print(f"  fold {k}: F1={m['f1_macro']:.4f} top1={m['top1_acc']:.4f} logloss={m['log_loss']:.4f} ({ttrain:.1f}s)")

        idx_all = np.flatnonzero(fit_mask)
        model, family, params, cap, ttrain = fit_model(name, X[idx_all], y[idx_all], tr_c.iloc[idx_all])
        t0 = time.time()
        pst = model.predict_proba(Xst)[:, 1]
        tinf = time.time() - t0
        np.save(PREDS / f"{name}_student_test.npy", pst)
        ms = eval_probs(st_c, st_o, pst)
        cont = student_continuity(st_c, pst, st_o)
        if hasattr(model, "steps") and hasattr(model.steps[-1][1], "loss_curve_"):
            mlp = model.steps[-1][1]
            fig, ax1 = plt.subplots(figsize=(6, 4))
            ax1.plot(mlp.loss_curve_, label="training loss (log-loss)")
            ax1.set_xlabel("epoch"); ax1.set_ylabel("training loss")
            ax2 = ax1.twinx()
            ax2.plot(mlp.validation_scores_, "r", label="validation accuracy (10% hold-out)")
            ax2.set_ylabel("validation accuracy")
            fig.legend(loc="upper right"); plt.title("MLP training history (full-train refit)")
            plt.tight_layout(); plt.savefig(PLOTS / "mlp_training_history.png", dpi=120); plt.close()

        cm = np.array(ms["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(4, 3.5))
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v}", ha="center", va="center")
        ax.set_xticks([0, 1], ["pred 0", "pred 1"]); ax.set_yticks([0, 1], ["true 0", "true 1"])
        ax.set_title(f"{name} (student-test, thr=0.5)")
        plt.tight_layout(); plt.savefig(PLOTS / f"cm_{name}.png", dpi=120); plt.close()

        cvd = pd.DataFrame(cv)
        row = {
            "model": name, "family": family,
            "config": json.dumps({k: v for k, v in params.items() if k not in ("n_jobs", "verbose")}),
            "max_train_rows": cap,
            "cv_f1_macro": cvd.f1_macro.mean(), "cv_f1_macro_std": cvd.f1_macro.std(),
            "cv_top1_acc": cvd.top1_acc.mean(), "cv_top1_acc_std": cvd.top1_acc.std(),
            "cv_log_loss": cvd.log_loss.mean(),
            "st_accuracy": ms["accuracy"], "st_precision_macro": ms["precision_macro"],
            "st_recall_macro": ms["recall_macro"], "st_f1_macro": ms["f1_macro"],
            "st_log_loss": ms["log_loss"], "st_top1_acc": ms["top1_acc"],
            "st_candidate_miss_rate": ms["candidate_miss_rate"],
            "st_route_continuity_clf_only": cont,
            "train_time_s": ttrain, "inference_time_s": tinf,
            "inference_ms_per_1k_rows": 1000 * tinf / len(Xst) * 1000,
        }
        rows.append(row)
        print(f"  STUDENT-TEST: F1={ms['f1_macro']:.4f} top1={ms['top1_acc']:.4f} logloss={ms['log_loss']:.4f} "
              f"continuity={cont:.3f} train={ttrain:.1f}s infer={tinf:.2f}s")
        log.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "script": "03_train_models",
                              "seed": C.SEED, "folds": args.folds, "features": C.FEATURES,
                              **{k: (v if not isinstance(v, (np.floating,)) else float(v)) for k, v in row.items()},
                              "st_confusion_matrix": ms["confusion_matrix"]}) + "\n")
        log.flush()

    comp = pd.DataFrame(rows)
    out = RES / "model_comparison.csv"
    if out.exists() and set(args.models) != set(REGISTRY):
        old = pd.read_csv(out)
        comp = pd.concat([old[~old.model.isin(comp.model)], comp], ignore_index=True)
    comp.to_csv(out, index=False)
    folds = pd.DataFrame(fold_rows)
    fout = RES / "cv_folds.csv"
    if fout.exists() and set(args.models) != set(REGISTRY):
        old = pd.read_csv(fout)
        folds = pd.concat([old[~old.model.isin(folds.model)], folds], ignore_index=True)
    folds.to_csv(fout, index=False)

    # ---- comparison plots (all from the CSVs just written)
    comp = comp.sort_values("st_top1_acc", ascending=False)
    fig, axs = plt.subplots(1, 4, figsize=(20, 4.5))
    axs[0].bar(comp.model, comp.cv_f1_macro, yerr=comp.cv_f1_macro_std, capsize=3)
    axs[0].set_title("CV Macro-F1 (5-fold GroupKFold, mean±std)")
    xx = np.arange(len(comp))
    axs[1].bar(xx - 0.2, comp.st_top1_acc, 0.4, label="Top-1 acc")
    axs[1].bar(xx + 0.2, comp.st_f1_macro, 0.4, label="Macro-F1")
    axs[1].set_xticks(xx, comp.model); axs[1].legend(); axs[1].set_title("Student-test")
    axs[2].bar(comp.model, comp.st_log_loss); axs[2].set_title("Student-test log-loss (lower better)")
    axs[3].bar(comp.model, comp.inference_ms_per_1k_rows); axs[3].set_yscale("log")
    axs[3].set_title("Inference ms / 1000 candidate rows")
    for a in axs:
        a.tick_params(axis="x", rotation=45)
        lo = 0 if a in (axs[2], axs[3]) else max(0, min(p.get_height() for p in a.patches) - 0.05)
        if a in (axs[0], axs[1]):
            a.set_ylim(lo, 1.0)
    plt.tight_layout(); plt.savefig(PLOTS / "model_comparison.png", dpi=120); plt.close()

    plt.figure(figsize=(8, 4))
    folds.boxplot(column="top1_acc", by="model", rot=45)
    plt.suptitle(""); plt.title("CV Top-1 accuracy per fold (stability)")
    plt.tight_layout(); plt.savefig(PLOTS / "cv_top1_folds.png", dpi=120); plt.close()
    print("\n", comp[["model", "cv_f1_macro", "cv_top1_acc", "st_f1_macro", "st_top1_acc", "st_log_loss",
                      "st_route_continuity_clf_only", "inference_ms_per_1k_rows"]].to_string(index=False))


if __name__ == "__main__":
    main()
