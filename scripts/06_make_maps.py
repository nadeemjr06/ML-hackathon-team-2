"""Folium maps of predictions on student-test trips (raw GPS, road network, reference, predicted).

Usage:  python scripts/06_make_maps.py [--trips 00000052 00000009] (default: all student-test trips)
Output: reports/maps/pred_<trip>.html
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import folium
import numpy as np
from branca.element import MacroElement, Template

from pathsnap import config as C
from pathsnap.io import list_trips, load_route, load_trip
from pathsnap.pipeline import MapMatcher, evaluate_against_reference, route_segments

LEGEND = """{% macro html(this, kwargs) %}
<div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999; background: white; padding: 10px 12px;
     border: 1px solid #999; border-radius: 6px; font: 13px sans-serif;">
<b>PathSnap {{ this.trip }}</b><br>
<span style="color:blue">&#9679;</span> raw GPS observations<br>
<span style="color:#888">&#9473;</span> road network (near track)<br>
<span style="color:green">&#9473;</span> reference route<br>
<span style="color:red">&#9473;</span> predicted (classifier + HMM)<br>
<span style="color:orange">&#9476;</span> classifier-only argmax<br>
<small>{{ this.metrics }}</small>
</div>{% endmacro %}"""


class Legend(MacroElement):
    def __init__(self, trip, metrics):
        super().__init__()
        self._template = Template(LEGEND)
        self.trip, self.metrics = trip, metrics


def make_map(matcher: MapMatcher, part_dir: Path, tid: str, out: Path):
    trip = load_trip(part_dir, tid)
    res = matcher.predict(trip)
    route = load_route(part_dir, tid)
    net, tr = res.net, trip.track
    m = folium.Map(location=[tr[:, 1].mean(), tr[:, 0].mean()], zoom_start=15, tiles="OpenStreetMap")
    near = net.candidates(res.px, res.py, radius=80, max_radius=80, k=30)
    arcs = np.unique(np.concatenate([c[0] for c in near]))
    fg = folium.FeatureGroup("Road network")
    for seg in net.arc_lonlat(arcs):
        folium.PolyLine(seg, color="#888888", weight=2, opacity=0.5).add_to(fg)
    fg.add_to(m)
    metrics = ""
    if route is not None:
        ev = evaluate_against_reference(res, route)
        metrics = (f"obs acc HMM {ev['accuracy_hmm']:.3f} vs clf {ev['accuracy_classifier_only']:.3f}; "
                   f"continuity {ev['route_continuity_hmm']:.3f} vs {ev['route_continuity_classifier_only']:.3f}")
        fg = folium.FeatureGroup("Reference route")
        folium.PolyLine(net.route_polyline(route), color="green", weight=7, opacity=0.6).add_to(fg)
        fg.add_to(m)
    fg = folium.FeatureGroup("Classifier-only", show=False)
    for s in route_segments(net, res.clf_arcs):
        folium.PolyLine(s, color="orange", weight=3, dash_array="6", opacity=0.9).add_to(fg)
    fg.add_to(m)
    fg = folium.FeatureGroup("Predicted (classifier + HMM)")
    for s in route_segments(net, res.hmm_arcs, fill=True):
        folium.PolyLine(s, color="red", weight=3, opacity=0.9).add_to(fg)
    fg.add_to(m)
    fg = folium.FeatureGroup("Raw GPS")
    for i in range(0, len(tr), max(1, len(tr) // 2500)):
        folium.CircleMarker([tr[i, 1], tr[i, 0]], radius=2, color="blue", fill=True, weight=1,
                            tooltip=f"obs {i} t={tr[i, 2]:.0f}s").add_to(fg)
    fg.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().add_child(Legend(tid, metrics))
    m.fit_bounds([[tr[:, 1].min(), tr[:, 0].min()], [tr[:, 1].max(), tr[:, 0].max()]])
    m.save(str(out))
    print(tid, metrics, "->", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trips", nargs="*")
    args = ap.parse_args()
    out = C.REPORTS_DIR / "maps"
    out.mkdir(parents=True, exist_ok=True)
    matcher = MapMatcher()
    for tid in args.trips or list_trips(C.STUDENT_TEST_DIR):
        make_map(matcher, C.STUDENT_TEST_DIR, tid, out / f"pred_{tid}.html")


if __name__ == "__main__":
    main()
