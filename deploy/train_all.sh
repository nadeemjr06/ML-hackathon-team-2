#!/usr/bin/env bash
# Reproduce every result in results/ and the artifacts in models/ from raw data.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
$PY -u scripts/01_eda.py
$PY -u scripts/02_build_dataset.py
$PY -u scripts/03_train_models.py
$PY -u scripts/04_bayes_opt.py --model "${MODEL:-lightgbm}" --n-trials "${TRIALS:-40}"
$PY -u scripts/05_hmm_eval.py
$PY -u scripts/06_make_maps.py
