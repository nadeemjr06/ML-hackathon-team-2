// PathSnap evaluator UI (Laptop 2). All predictions and accuracy numbers come
// from the API on Laptop 1; this page never sees reference routes.
const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
let API = params.get("api") || safeGet("pathsnap_api") || window.PATHSNAP_API;
$("api").value = API;

function safeGet(k) { try { return localStorage.getItem(k); } catch { return null; } }
function safeSet(k, v) { try { localStorage.setItem(k, v); } catch {} }

const map = L.map("map", { preferCanvas: true }).setView([20, 0], 2);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19, attribution: "&copy; OpenStreetMap contributors",
}).addTo(map);
const layers = { gps: L.layerGroup(), pred: L.layerGroup(), clf: L.layerGroup() };
layers.gps.addTo(map); layers.pred.addTo(map);
L.control.layers(null, {
  "Raw GPS observations": layers.gps,
  "Predicted route (classifier + HMM)": layers.pred,
  "Classifier-only argmax": layers.clf,
}, { collapsed: false }).addTo(map);
const legend = L.control({ position: "bottomleft" });
legend.onAdd = () => {
  const d = L.DomUtil.create("div", "legend");
  d.innerHTML = '<div><span class="sw" style="background:#1f6feb;height:8px;width:8px;border-radius:50%"></span>GPS observation</div>' +
    '<div><span class="sw" style="background:#d62828"></span>Predicted route (HMM)</div>' +
    '<div><span class="sw" style="background:orange"></span>Classifier-only</div>';
  return d;
};
legend.addTo(map);

async function connect() {
  API = $("api").value.trim().replace(/\/$/, "");
  safeSet("pathsnap_api", API);
  const st = $("api-status");
  st.textContent = "connecting..."; st.className = "pill";
  try {
    const h = await (await fetch(`${API}/health`)).json();
    st.textContent = `connected · ${h.model}`; st.className = "pill ok";
    const t = await (await fetch(`${API}/trips`)).json();
    renderTrips(t.trips);
  } catch (e) {
    st.textContent = "cannot reach API"; st.className = "pill err";
    $("trips").innerHTML = `<p class="muted">Could not reach ${API}. Check the IP, port and that the API is running.</p>`;
  }
}

function renderTrips(trips) {
  const box = $("trips");
  box.innerHTML = "";
  trips.forEach((t) => {
    const b = document.createElement("button");
    b.className = "trip";
    b.innerHTML = `<b>${t.trip_id}</b><small>${t.n_gps_points} GPS points</small>`;
    b.onclick = () => runTrip(t.trip_id, b);
    box.appendChild(b);
  });
}

const pct = (v) => (v === null || v === undefined || Number.isNaN(v) ? "n/a" : (100 * v).toFixed(2) + " %");
const num = (v, d = 3) => (v === null || v === undefined ? "n/a" : Number(v).toFixed(d));

async function runTrip(tripId, btn) {
  document.querySelectorAll(".trip").forEach((x) => x.classList.remove("active"));
  btn.classList.add("active");
  $("status").textContent = `Sending trip ${tripId} to the API...`;
  $("metrics").innerHTML = "";
  const t0 = performance.now();
  let r;
  try {
    const resp = await fetch(`${API}/predict`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ trip_id: tripId }),
    });
    if (!resp.ok) throw new Error(`${resp.status} ${await resp.text()}`);
    r = await resp.json();
  } catch (e) {
    $("status").textContent = `Error: ${e.message}`;
    return;
  }
  const rtt = (performance.now() - t0) / 1000;
  drawResult(r);
  $("status").innerHTML = `Trip <b>${r.trip_id}</b> · ${r.n_gps_points} GPS points · model <b>${r.model}</b> · round-trip ${rtt.toFixed(2)} s`;
  renderMetrics(r);
}

function drawResult(r) {
  Object.values(layers).forEach((l) => l.clearLayers());
  r.gps.forEach((p) => L.circleMarker(p, { radius: 2.5, color: "#1f6feb", weight: 1, fillOpacity: 0.8 }).addTo(layers.gps));
  r.classifier_only_route.forEach((s) => L.polyline(s, { color: "orange", weight: 3, dashArray: "6 6" }).addTo(layers.clf));
  r.predicted_route.forEach((s) => L.polyline(s, { color: "#d62828", weight: 4, opacity: 0.9 }).addTo(layers.pred));
  const b = L.latLngBounds(r.gps);
  if (b.isValid()) map.fitBounds(b, { padding: [20, 20] });
}

function renderMetrics(r) {
  const m = r.metrics;
  const rows = [];
  const head = (t) => rows.push(`<tr class="head"><td colspan="2">${t}</td></tr>`);
  const row = (k, v) => rows.push(`<tr><td>${k}</td><td>${v}</td></tr>`);
  if (m) {
    head("Accuracy (computed on the API server)");
    row("<b>Observation accuracy (HMM)</b>", `<b>${pct(m.accuracy_hmm)}</b>`);
    row("Observation accuracy (classifier only)", pct(m.accuracy_classifier_only));
    row("Accuracy, all observations (HMM)", pct(m.accuracy_hmm_all_obs));
    row("Reliable / total observations", `${m.n_reliable_observations} / ${m.n_observations}`);
    head("Route quality");
    row("Route continuity (HMM)", num(m.route_continuity_hmm));
    row("Route continuity (classifier only)", num(m.route_continuity_classifier_only));
    row("Route precision / recall (HMM)", `${num(m.route_precision_hmm)} / ${num(m.route_recall_hmm)}`);
    row("Route F1 (HMM)", num(m.route_f1_hmm));
    row("Route edit similarity (HMM)", num(m.route_edit_similarity_hmm));
  } else {
    head("No reference route on the server for this trip");
  }
  head("Timing (server)");
  Object.entries(r.timings_s).forEach(([k, v]) => row(k, `${v} s`));
  $("metrics").innerHTML = rows.join("");
}

$("connect").onclick = connect;
$("api").addEventListener("keydown", (e) => { if (e.key === "Enter") connect(); });
connect();
