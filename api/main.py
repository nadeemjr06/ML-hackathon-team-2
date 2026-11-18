"""PathSnap FastAPI backend (runs on Laptop 1).

Hosts the trained classifier, the HMM/Viterbi decoder, the road-network data and
the server-side evaluation. The UI (Laptop 2) only sends a trip id.

    GET  /health            liveness + loaded model info
    GET  /trips             the evaluation trips available on this server
    POST /predict           {"trip_id": "..."} -> predicted route + server-side metrics

Reference routes (.route) are read only by the evaluation step, after the
prediction has been produced, and are never returned to the client.

Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000
Env:  PATHSNAP_EVAL_DIR  folder with the evaluation trips (default: data/student_test)
      PATHSNAP_MODELS    folder with final_model.joblib, model_meta.json, hmm_config.json
"""
from __future__ import annotations

import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pathsnap import config as C
from pathsnap.io import list_trips, load_route, load_trip
from pathsnap.pipeline import MapMatcher, evaluate_against_reference, route_segments

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                    stream=sys.stdout)
log = logging.getLogger("pathsnap.api")

STATE: dict = {}


class PredictRequest(BaseModel):
    trip_id: str


def startup():
    t0 = time.time()
    STATE["matcher"] = MapMatcher(C.MODELS_DIR)
    STATE["eval_dir"] = C.EVAL_DIR
    STATE["trips"] = list_trips(C.EVAL_DIR)
    m = STATE["matcher"]
    log.info("Loaded model=%s from %s in %.2fs", m.meta["model"], C.MODELS_DIR, time.time() - t0)
    log.info("HMM config: %s", m.hmm_cfg.to_dict())
    log.info("Evaluation trips (%d) from %s: %s", len(STATE["trips"]), C.EVAL_DIR, STATE["trips"])


@asynccontextmanager
async def lifespan(_app: FastAPI):
    startup()
    yield


app = FastAPI(title="PathSnap Map-Matching API", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    m = STATE["matcher"]
    return {"status": "ok", "model": m.meta["model"], "model_created": m.meta.get("created"),
            "n_eval_trips": len(STATE["trips"])}


@app.get("/trips")
def trips(request: Request):
    out = []
    for tid in STATE["trips"]:
        p = Path(STATE["eval_dir"]) / tid / f"{tid}.track"
        with open(p, "rb") as f:
            n = sum(1 for _ in f)
        out.append({"trip_id": tid, "n_gps_points": n})
    log.info("GET /trips from %s -> %d trips", request.client.host if request.client else "?", len(out))
    return {"trips": out}


@app.post("/predict")
def predict(req: PredictRequest, request: Request):
    client = request.client.host if request.client else "?"
    if req.trip_id not in STATE["trips"]:
        log.warning("POST /predict from %s: unknown trip_id=%r", client, req.trip_id)
        raise HTTPException(404, f"unknown trip_id {req.trip_id}")
    log.info("POST /predict from %s: trip_id=%s -- running pipeline", client, req.trip_id)
    t0 = time.perf_counter()
    trip = load_trip(STATE["eval_dir"], req.trip_id)          # track + road network only
    res = STATE["matcher"].predict(trip)
    t_pred = time.perf_counter() - t0
    log.info("  trip %s: %d GPS points, %d without candidates, timings %s", req.trip_id, len(res.px),
             int((~res.has_candidate).sum()), {k: round(v, 3) for k, v in res.timings.items()})

    metrics = None
    route = load_route(STATE["eval_dir"], req.trip_id)       # server-side reference, evaluation only
    if route is not None:
        metrics = evaluate_against_reference(res, route)
        log.info("  trip %s: ACCURACY (HMM)=%.4f | classifier-only=%.4f | continuity HMM=%.3f | route F1=%.3f",
                 req.trip_id, metrics["accuracy_hmm"] or float("nan"),
                 metrics["accuracy_classifier_only"] or float("nan"),
                 metrics["route_continuity_hmm"], metrics["route_f1_hmm"])
    else:
        log.info("  trip %s: no reference route on server -> metrics unavailable", req.trip_id)

    tr = trip.track
    step = max(1, len(tr) // 3000)
    payload = {
        "trip_id": req.trip_id,
        "n_gps_points": int(len(tr)),
        "gps": [[float(tr[i, 1]), float(tr[i, 0])] for i in range(0, len(tr), step)],
        "predicted_route": route_segments(res.net, res.hmm_arcs, fill=True),
        "classifier_only_route": route_segments(res.net, res.clf_arcs),
        "predicted_arcs_per_observation": res.hmm_arcs.tolist(),
        "metrics": metrics,
        "timings_s": {**{k: round(v, 3) for k, v in res.timings.items()}, "total_prediction": round(t_pred, 3)},
        "model": STATE["matcher"].meta["model"],
    }
    log.info("  trip %s: done in %.2fs (response: %d route segments)", req.trip_id,
             time.perf_counter() - t0, len(payload["predicted_route"]))
    return payload
