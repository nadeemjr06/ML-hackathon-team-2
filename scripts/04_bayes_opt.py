"""Bayesian hyperparameter optimisation (Optuna TPE) of the selected classifier.

Objective : maximise mean observation-level Top-1 accuracy over GroupKFold folds
            (groups = trip_id) on the 80 training trips (reliable observations).
            Top-1 is the primary metric because the decoder ultimately has to pick
            one arc per observation; log-loss is recorded per trial as well since
            the probabilities are used as HMM emissions.
Sampler   : TPE (Tree-structured Parzen Estimator), seed fixed; MedianPruner on folds.
After the search: refit with best params on all training rows, evaluate on the
student-test trips, compare against the untuned (registry default) model with the
*same* protocol, export the final model + metadata and out-of-fold train
probabilities (used to tune the HMM without leakage).

Usage:  python scripts/04_bayes_opt.py --model lightgbm --n-trials 40
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold

from pathsnap import config as C
from pathsnap.data import load_part, subsample_obs, training_mask
from pathsnap.evaluate import eval_probs
from pathsnap.models import build

RES = C.RESULTS_DIR
PLOTS = RES / "plots"


def space(trial: optuna.Trial, model: str) -> dict:
    if model == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 300, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        }
    if model == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=25),
            "max_depth": trial.suggest_int("max_depth", 6, 40),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50, log=True),
            "max_features": trial.suggest_float("max_features", 0.2, 0.9),
        }
    if model == "mlp":
        n1 = trial.suggest_int("n_units_1", 16, 256, log=True)
        n2 = trial.suggest_int("n_units_2", 8, 128, log=True)
        return {
            "hidden_layer_sizes": (n1, n2),
            "alpha": trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
        }
    raise ValueError(f"no search space defined for {model}")


def cv_eval(model_name, params, X, y, tr_c, tr_o, fit_mask, folds, max_rows=None, trial=None, return_oof=False):
    gkf = GroupKFold(n_splits=folds)
    scores, oof = [], np.full(len(X), np.nan)
    for k, (tri, vai) in enumerate(gkf.split(X, y, tr_c["trip_id"].to_numpy())):
        tri = tri[fit_mask[tri]]
        model, _, _, cap = build(model_name, params)
        cap = min(x for x in (cap, max_rows) if x is not None) if (cap or max_rows) else None
        sub = tri[subsample_obs(tr_c.iloc[tri], cap, C.SEED + k)]
        model.fit(X[sub], y[sub])
        p = model.predict_proba(X[vai])[:, 1]
        oof[vai] = p
        vc = tr_c.iloc[vai]
        m = eval_probs(vc, tr_o[tr_o.trip_id.isin(vc.trip_id.unique())], p)
        scores.append(m)
        if trial is not None:
            trial.report(np.mean([s["top1_acc"] for s in scores]), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
    df = pd.DataFrame(scores)
    return (df, oof) if return_oof else df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lightgbm")
    ap.add_argument("--n-trials", type=int, default=40)
    ap.add_argument("--search-folds", type=int, default=3)
    ap.add_argument("--trial-rows", type=int, default=400_000,
                    help="max training rows per fold during the search (observation subsample) for speed")
    args = ap.parse_args()
    PLOTS.mkdir(parents=True, exist_ok=True)
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    tr_c, tr_o = load_part("train")
    st_c, st_o = load_part("student_test")
    X = tr_c[C.FEATURES].to_numpy(np.float32)
    y = tr_c["label"].to_numpy()
    fit_mask = training_mask(tr_c, tr_o)
    Xst = st_c[C.FEATURES].to_numpy(np.float32)

    # ------------------------------------------------------------- search
    def objective(trial):
        params = space(trial, args.model)
        t0 = time.time()
        df = cv_eval(args.model, params, X, y, tr_c, tr_o, fit_mask, args.search_folds, args.trial_rows, trial)
        trial.set_user_attr("cv_log_loss", float(df.log_loss.mean()))
        trial.set_user_attr("cv_f1_macro", float(df.f1_macro.mean()))
        trial.set_user_attr("cv_top1_std", float(df.top1_acc.std()))
        trial.set_user_attr("seconds", time.time() - t0)
        return float(df.top1_acc.mean())

    study = optuna.create_study(direction="maximize", study_name=f"pathsnap_{args.model}",
                                sampler=optuna.samplers.TPESampler(seed=C.SEED, n_startup_trials=8),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0))
    t_search = time.time()
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)
    t_search = time.time() - t_search
    trials = study.trials_dataframe()
    trials.to_csv(RES / f"optuna_trials_{args.model}.csv", index=False)
    best = study.best_params
    if args.model == "mlp":
        best = {"hidden_layer_sizes": (best.pop("n_units_1"), best.pop("n_units_2")), **best}
    print("best CV top-1 (search protocol):", study.best_value, "params:", best)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    done = trials[trials.state == "COMPLETE"]
    axs[0].plot(done.number, done.value, "o", label="trial")
    axs[0].plot(done.number, done.value.cummax(), "r-", label="best so far")
    axs[0].set_xlabel("trial"); axs[0].set_ylabel("CV top-1 acc"); axs[0].legend()
    axs[0].set_title(f"Optuna TPE history ({args.model})")
    try:
        imp = optuna.importance.get_param_importances(study)
        axs[1].barh(list(imp.keys()), list(imp.values())); axs[1].set_title("Hyperparameter importance (fANOVA)")
    except Exception as e:  # noqa: BLE001
        axs[1].text(0.1, 0.5, f"importance unavailable: {e}")
    plt.tight_layout(); plt.savefig(PLOTS / f"optuna_{args.model}.png", dpi=120); plt.close()

    # --------------------------------------- tuned vs untuned, same protocol
    compare = []
    final = None
    for label, params in (("untuned", None), ("tuned", best)):
        cvdf, oof = cv_eval(args.model, params, X, y, tr_c, tr_o, fit_mask, 5, return_oof=True)
        model, family, p_used, cap = build(args.model, params)
        idx = np.flatnonzero(fit_mask)
        idx = idx[subsample_obs(tr_c.iloc[idx], cap, C.SEED)]
        t0 = time.time(); model.fit(X[idx], y[idx]); ttrain = time.time() - t0
        t0 = time.time(); pst = model.predict_proba(Xst)[:, 1]; tinf = time.time() - t0
        ms = eval_probs(st_c, st_o, pst)
        compare.append({"variant": label, "params": json.dumps(p_used, default=str),
                        "cv_f1_macro": cvdf.f1_macro.mean(), "cv_top1_acc": cvdf.top1_acc.mean(),
                        "cv_log_loss": cvdf.log_loss.mean(),
                        "st_top1_acc": ms["top1_acc"], "st_f1_macro": ms["f1_macro"],
                        "st_log_loss": ms["log_loss"], "train_time_s": ttrain, "inference_time_s": tinf})
        print(compare[-1])
        if label == "tuned":
            final = (model, p_used, oof, pst, ms, cvdf)
    cmp = pd.DataFrame(compare)
    cmp["search_time_s"] = [np.nan, t_search]
    cmp["n_trials"] = [np.nan, len(trials)]
    cmp.to_csv(RES / f"tuned_vs_untuned_{args.model}.csv", index=False)

    # Final model = tuned variant (the tuned-vs-untuned table documents whether that helped).
    model, p_used, oof, pst, ms, cvdf = final
    joblib.dump(model, C.MODELS_DIR / "final_model.joblib")
    np.save(C.CACHE_DIR / "train_oof_final.npy", oof)
    np.save(RES / "preds" / "final_student_test.npy", pst)
    meta = {
        "model": args.model, "params": p_used, "features": C.FEATURES, "seed": C.SEED,
        "candidate": {"radius_m": C.CAND_RADIUS_M, "max_radius_m": C.CAND_MAX_RADIUS_M, "max_k": C.CAND_MAX_K},
        "label_max_dist_m": C.LABEL_MAX_DIST_M,
        "cv_top1_acc": float(cvdf.top1_acc.mean()), "student_test_top1_acc": float(ms["top1_acc"]),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (C.MODELS_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print("saved", C.MODELS_DIR / "final_model.joblib")


if __name__ == "__main__":
    main()
