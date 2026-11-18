# PathSnap: Distributed Map-Matching & Route Reconstruction

**Machine Learning Hackathon · Team 2**

PathSnap takes a noisy GPS trajectory and its local road network and predicts the road segment (directed arc) occupied at each GPS observation. It then reconstructs a connected route.

## How it works

1. **Candidate generation and classification:** nearby directed road arcs are generated for each GPS fix. A binary classifier scores the probability that each candidate is the true arc using geometric and kinematic features.
2. **HMM / Viterbi decoding:** classifier probabilities provide emission scores, while road-network distances provide transition scores. Viterbi selects the most likely connected route.

## System architecture

The application runs across two laptops on the same network:

| Machine | Component | Responsibility |
|---|---|---|
| Laptop 1 | FastAPI backend (`api/main.py`) | Loads the model, runs candidate generation, feature extraction, classification and HMM/Viterbi decoding; computes evaluation metrics |
| Laptop 2 | Web UI (`ui/`) | Lists evaluation trips, sends the selected trip ID to the API, and displays the predicted route and metrics on a Leaflet map |

The prediction pipeline reads `.track`, `.nodes` and `.arcs`. The reference `.route` is loaded only after prediction for server-side evaluation and is not sent to the UI.

## Run the application

### Requirements

- Python 3.10–3.13
- The project dependencies from `requirements-api.txt` on Laptop 1
- A browser and Python on Laptop 2
- Both laptops connected to the same Wi-Fi network or hotspot
- The evaluation trip folder, containing one folder per trip with `.track`, `.nodes`, `.arcs` and `.route` files

### Laptop 1 — start the API

Install the project dependencies and activate the project's virtual environment. Then run:

```bash
# Set the folder containing the evaluator trips
# Windows PowerShell:
$env:PATHSNAP_EVAL_DIR = "C:\path\to\eval_trips"

# Start the API
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Alternatively, use `deploy/run_api.bat` on Windows or `deploy/run_api.sh` on macOS/Linux. Share Laptop 1's LAN IP address with Laptop 2. Keep the API terminal open to view request logs.

Check that the API is reachable at:

```text
http://<LAPTOP1_IP>:8000/health
```

### Laptop 2 — start the UI

From the project directory, run:

```bash
cd ui
python -m http.server 5500
```

Alternatively, use `deploy/run_ui.bat` or `deploy/run_ui.sh` with the API address. Open this URL in the browser:

```text
http://localhost:5500/?api=http://<LAPTOP1_IP>:8000
```

Select an evaluation trip to view its GPS points, predicted route and evaluation metrics. If the laptops cannot communicate over campus Wi-Fi, use a phone hotspot.

## Reproduce the experiments

The trained model and configuration files are included in `models/`, so retraining is not required to run the API. To regenerate the experiments, install the full dependencies and run:

```bash
bash deploy/train_all.sh
```

This runs EDA, dataset construction, model comparison, Bayesian hyperparameter optimisation, HMM evaluation and map generation. It takes approximately 40 minutes and overwrites files in `results/` and `models/`.

Individual stages can also be run with the scripts in `scripts/`:

| Script | Purpose |
|---|---|
| `01_eda.py` | Dataset inspection, quality checks and EDA plots |
| `02_build_dataset.py` | Candidate generation, feature extraction and label construction |
| `03_train_models.py` | Train and compare seven models using 5-fold GroupKFold |
| `04_bayes_opt.py` | Optuna Bayesian hyperparameter optimisation |
| `05_hmm_eval.py` | HMM parameter tuning and evaluation |
| `06_make_maps.py` | Generate maps of predicted and reference routes |

The random seed is fixed at `42`.

## Method summary

| Stage | Approach |
|---|---|
| Projection | Per-trip local equirectangular projection centred on the GPS centroid |
| Cleaning | Remove invalid coordinates and duplicate timestamps; sort observations by time |
| Labels | Monotonic dynamic-programming alignment of GPS observations to the reference arc sequence |
| Candidates | Directed arcs within 50 m, expanding the radius up to 200 m when needed; retain at most 8 candidates |
| Features | 23 geometric, heading, kinematic and road-topology features |
| Classifier | Binary candidate scorer; seven models compared using trip-grouped cross-validation |
| Hyperparameter optimisation | Optuna TPE search with GroupKFold by trip; objective is observation-level Top-1 accuracy |
| Route decoding | HMM with Viterbi decoding, network-distance transitions and a U-turn penalty |

**Leakage controls:** cross-validation folds are grouped by trip. Student-test trips are reserved for final evaluation. HMM parameters are tuned using out-of-fold probabilities from training trips. The prediction path does not read the reference `.route` file.

## Results and outputs

Experiment outputs are stored in `results/`:

- `model_comparison.csv` — model comparison metrics
- `cv_folds.csv` — per-fold cross-validation scores
- `experiment_log.jsonl` — experiment configurations, seeds and timings
- `optuna_trials_<model>.csv` and `tuned_vs_untuned_<model>.csv` — hyperparameter search and comparison results
- `hmm_grid.csv` and related HMM result files — decoder tuning and evaluation
- `plots/` — model comparison, confusion matrices, optimisation history and other experiment plots

Generated EDA reports and interactive route maps are stored in `reports/eda/` and `reports/maps/`.

## Repository structure

```text
pathsnap/       Core map-matching library, models, features, metrics and HMM decoder
scripts/        EDA, dataset construction, training, optimisation and map-generation scripts
api/            FastAPI backend
ui/             Leaflet-based evaluator interface
deploy/         API, UI and full-training launch scripts
models/         Trained model and configuration files
results/        Experiment metrics, logs, trial records and plots
reports/        EDA outputs and generated route maps
```
