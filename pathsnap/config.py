"""Central configuration: paths and pipeline constants.

All paths can be overridden with environment variables so the same code runs
on the training machine, on Laptop 1 (API) and in CI.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.environ.get("PATHSNAP_DATA", ROOT / "data"))
TRAIN_DIR = DATA_DIR / "train"
STUDENT_TEST_DIR = DATA_DIR / "student_test"
# Directory holding the 10 evaluator trips on hackathon day (same layout as
# train/ and student_test/). Defaults to student_test for development.
EVAL_DIR = Path(os.environ.get("PATHSNAP_EVAL_DIR", STUDENT_TEST_DIR))

CACHE_DIR = Path(os.environ.get("PATHSNAP_CACHE", ROOT / "data_cache"))
RESULTS_DIR = ROOT / "results"
MODELS_DIR = Path(os.environ.get("PATHSNAP_MODELS", ROOT / "models"))
REPORTS_DIR = ROOT / "reports"

SEED = 42

# ---- Candidate generation -------------------------------------------------
CAND_RADIUS_M = 50.0         # initial search radius around each GPS point
CAND_MAX_RADIUS_M = 200.0    # radius is doubled up to this if nothing is found
CAND_MAX_K = 8               # keep at most K nearest arcs per observation
KD_SAMPLE_STEP_M = 10.0      # arc densification step for the KD-tree index

# ---- Feature engineering --------------------------------------------------
HEADING_WINDOW = 3           # +-w observations used to estimate GPS heading
DIST_WINDOW = 3              # +-w observations used for window distance features
GAUSS_SIGMA_M = 10.0         # sigma for the distance-softmax feature
MIN_SPEED_FOR_HEADING = 1.0  # m/s; below this heading is considered unreliable

# ---- Labels ---------------------------------------------------------------
# Observations whose aligned reference arc is further than this are treated as
# unreliable labels and are excluded from *training* (never from evaluation).
LABEL_MAX_DIST_M = 50.0

FEATURES = [
    "dist", "dist_rank", "dist_minus_min", "dist_ratio", "n_cands",
    "win_dist_mean", "win_dist_max", "win_dist_minus_min",
    "gauss_prob", "win_gauss_prob",
    "heading_cos", "heading_sin_abs", "heading_valid", "along_disp",
    "speed", "proj_t", "endpoint_clip", "log_arc_len",
    "from_in_deg", "to_out_deg", "has_reverse",
    "dt_prev", "dt_next",
]
