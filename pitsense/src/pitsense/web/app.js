"use strict";
// PitSense pit wall: renders /api/stream (server-sent events); falls back to polling /api/snapshot.
const $ = (id) => document.getElementById(id);
const ESC = {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"};
const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ESC[c]);
const TRACK = {"1": ["GREEN", "green"], "2": ["YELLOW", "yellow"], "3": ["", "green"], "4": ["SAFETY CAR", "sc"],
  "5": ["RED FLAG", "red"], "6": ["VSC", "vsc"], "7": ["VSC ENDING", "vsc"]};
const SHORT = {SOFT: "S", MEDIUM: "M", HARD: "H", INTERMEDIATE: "I", WET: "W"};
let pinned = new Set();
try { pinned = new Set(JSON.parse(localStorage.getItem("pinned") || "[]")); } catch (e) { /* no storage */ }
let snap = null;

const fmtTime = (x) => {
  if (x == null) return "--";
  const m = Math.floor(x / 60), s = x - m * 60;
  return m ? m + ":" + s.toFixed(3).padStart(6, "0") : s.toFixed(3);
};
const fmtGap = (g, down) => (down ? "+" + down + "L" : g == null ? "--" : g === 0 ? "LEADER" : "+" + g.toFixed(3));
const fmtClock = (t) => {
  const s = Math.floor(t), h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return h + ":" + String(m).padStart(2, "0") + ":" + String(s % 60).padStart(2, "0");
};
const pct = (x) => (x == null ? "--" : Math.round(x * 100) + "%");
const num = (x, d) => (x == null ? "--" : Number(x).toFixed(d == null ? 1 : d));
const first = (a, b) => (a != null ? a : b);

function tyre(c, age) {
  const k = SHORT[c] || "u";
  return '<span class="tyre ' + k + '">' + (k === "u" ? "?" : k) + "</span>" + (age == null ? "--" : age);
}
function action(c, big) {
  return '<span class="' + (big ? "big-act " : "") + "act " + esc(c) + '">' + esc(c.replace(/_/g, " ")) + "</span>";
}
const focusSet = () => new Set((snap.focus || []).concat([...pinned]));

function renderHeader(s) {
  const x = s.extra;
  $("title").textContent = x.title || "";
  $("lap").textContent = "LAP " + s.lap + (s.total_laps ? " / " + s.total_laps : "");
  $("clock").textContent = fmtClock(s.t);
  const tr = TRACK[s.track_status] || [s.track_status, "dim"];
  const st = $("status");
  const secs = s.track_status !== "1" && x.track_status_since != null ? " " + Math.max(0, Math.round(s.t - x.track_status_since)) + "s" : "";
  st.textContent = tr[0] + secs;
  st.className = "status " + tr[1];
  if (s.session_status === "Finished" || s.session_status === "Finalised") { st.textContent = "FINISHED"; st.className = "status dim"; }
  const w = x.weather || {};
  $("wx").textContent = w.AirTemp != null
    ? "air " + num(w.AirTemp) + "  track " + num(w.TrackTemp) + "  hum " + num(w.Humidity, 0) + "%  " + (w.Rainfall ? "RAIN" : "dry")
    : "weather --";
  const loss = (s.race || {}).pitstop__loss_now;
  $("loss").textContent = loss != null ? "pit loss " + num(loss) + " s" : "pit loss --";
  const speed = x.speed === 0 ? "max" : (x.speed || 1) + "x";
  $("mode").textContent = String(x.mode || "").toUpperCase() + (x.mode === "replay" ? " " + speed : "") +
    (x.model && x.model.loaded ? " | models" : " | no models");
  const msgs = [];
  if (x.inferred_order) msgs.push("Order INFERRED from laps and gaps: this feed has no car positions (no-auth). Positions can lag or swap in the pits.");
  if (s.track_status === "5") msgs.push("Red flag: pit loss estimates are not valid.");
  $("warn").hidden = !msgs.length;
  $("warn").textContent = msgs.join("  ");
}

function renderTower(s) {
  const mine = focusSet(), cars = s.cars || {}, calls = {}, rows = [], out = [];
  for (const c of s.calls || []) calls[c.car] = c;
  for (const r of s.tower) {
    if (!r.running) { out.push(r.tla || r.car); continue; }
    const v = cars[r.car] || {};
    const col = (s.extra.colours || {})[r.car];
    const rj = v.pitstop__rejoin_if_box_now;
    const rjc = rj == null || r.position == null ? "" : rj > r.position ? "worse" : "better";
    const p1 = first(v.models__pit_prob_1, v.rivals__pit_prob_1), p3 = first(v.models__pit_prob_3, v.rivals__pit_prob_3);
    const hot = p3 != null && p3 > 0.5 ? "hotter" : p3 != null && p3 > 0.25 ? "hot" : "";
    const call = calls[r.car];
    rows.push('<tr class="car ' + (mine.has(r.car) ? "mine " : "") + (r.in_pit ? "pit" : "") + '" data-car="' + esc(r.car) + '">' +
      '<td class="p">' + (r.position == null ? "-" : r.position) + "</td>" +
      '<td><span class="team" style="background:' + (col ? "#" + esc(col) : "#444") + '"></span><span class="tla">' + esc(r.tla || r.car) +
      '</span><span class="cn">' + esc(r.car) + "</span></td>" +
      "<td>" + fmtGap(r.gap_to_leader, r.laps_down) + "</td>" +
      "<td>" + (r.position === 1 ? "" : fmtGap(r.interval, 0)) + "</td>" +
      "<td>" + tyre(r.compound, r.tyre_age) + "</td><td>" + r.pit_stops + "</td>" +
      "<td>" + fmtTime(r.last_lap_time) + "</td>" +
      '<td class="' + rjc + '">' + (r.in_pit ? "in pit" : rj == null ? "--" : "P" + Math.round(rj)) + "</td>" +
      '<td class="' + hot + '">' + (p1 == null && p3 == null ? "--" : pct(p1) + " / " + pct(p3)) + "</td>" +
      "<td>" + (call && call.action !== "NO_CALL" ? action(call.action) : "") + "</td></tr>");
  }
  document.querySelector("#tower tbody").innerHTML = rows.join("");
  $("out").textContent = out.length ? "Out: " + out.join(", ") : "";
}

function planHtml(p, label) {
  if (!p) return "";
  const stops = (p.stops || []).map((s) => "L" + s.lap + " " + esc(s.compound)).join(", ") || "no stop";
  const ex = p.expected_position != null ? " &rarr; P" + num(p.expected_position) : "";
  const tr = p.trigger ? " <i>(" + esc(p.trigger) + ")</i>" : "";
  return "<div><b>Plan " + esc(p.name || label) + "</b> " + stops + ex + tr + "</div>";
}

// ---- radio: the voice engineer's message for each car's latest call (spoken by the browser if enabled)
let radioOn = false;
try { radioOn = localStorage.getItem("pitsense.radio") === "1"; } catch (e) {}
const spoken = {};
function radioHtml(s, n) {
  const r = ((s.extra || {}).radio || {})[n];
  return r ? '<div class="radio">&#128251; <i>"' + esc(r.text) + '"</i> <span class="kv">lap ' + esc(r.lap) + "</span></div>" : "";
}
function speakRadio(s, cars) {
  const all = (s.extra || {}).radio || {};
  for (const n of cars) {
    const r = all[n];
    if (!r || spoken[n] === r.text) continue;
    const first = spoken[n] === undefined;
    spoken[n] = r.text;
    if (first || !radioOn) continue;  // don't read out the backlog on connect
    radioQueue.push({ car: n, text: r.text });
  }
  playNext();
}
// play radio clips one after another: Piper audio from the server, else the browser's own voice
const radioQueue = [];
let radioBusy = false;
function playNext() {
  if (radioBusy || !radioQueue.length) return;
  const m = radioQueue.shift();
  radioBusy = true;
  const done = () => { radioBusy = false; playNext(); };
  const browserVoice = () => {
    if (!window.speechSynthesis) return done();
    const u = new SpeechSynthesisUtterance(m.text);
    u.rate = 1.05; u.onend = done; u.onerror = done;
    window.speechSynthesis.speak(u);
  };
  const a = new Audio("/api/radio.wav?car=" + encodeURIComponent(m.car) + "&v=" + encodeURIComponent(m.text.length + m.text.slice(0, 20)));
  a.onended = done;
  a.onerror = browserVoice;
  a.play().catch(browserVoice);
}
function flipRadio(btn) {
  radioOn = !radioOn;
  try { localStorage.setItem("pitsense.radio", radioOn ? "1" : "0"); } catch (e) {}
  btn.textContent = radioOn ? "radio on" : "radio off";
}
function radioToggle() {
  return '<button id="radiobtn" onclick="flipRadio(this)">' + (radioOn ? "radio on" : "radio off") + "</button>";
}

function renderFocus(s) {
  const mine = [...focusSet()];
  speakRadio(s, mine);
  $("focusnote").innerHTML = esc((s.focus || []).length ? (s.extra.team || "") : "no team set: pin cars from the tower") + " " + radioToggle();
  if (!mine.length) { $("focus").innerHTML = '<div class="empty">Start with --team, or click a car in the tower.</div>'; return; }
  const row = {}, calls = {};
  for (const r of s.tower) row[r.car] = r;
  for (const c of s.calls || []) calls[c.car] = c;
  $("focus").innerHTML = mine.filter((n) => row[n]).map((n) => {
    const r = row[n], v = (s.cars || {})[n] || {}, c = calls[n];
    const reasons = c && c.reasons && c.reasons.length ? "<ul>" + c.reasons.map((x) => "<li>" + esc(x.text) + "</li>").join("") + "</ul>" : "";
    const conf = c && c.confidence != null ? '<span class="kv">confidence <span>' + pct(c.confidence) + "</span></span>" : "";
    const comp = c && c.compound ? '<span class="kv">fit <span>' + esc(c.compound) + "</span></span>" : "";
    const plans = c ? planHtml(c.plan_a, "A") + planHtml(c.plan_b, "B") : "";
    return '<div class="card"><div class="top"><b>' + esc(r.tla || n) + '</b><span class="kv">#' + esc(n) + "</span>" +
      '<span class="kv">P<span>' + (r.position == null ? "-" : r.position) + "</span></span>" + tyre(r.compound, r.tyre_age) +
      action(c ? c.action : "NO_CALL", true) + comp + conf + "</div>" + radioHtml(s, n) + reasons +
      '<div class="plan">' + plans + "</div>" +
      '<div class="kv"><span>box now &rarr; ' + (v.pitstop__rejoin_if_box_now != null ? "P" + Math.round(v.pitstop__rejoin_if_box_now) : "--") + "</span>" +
      "<span>deg " + num(v.tyre__deg_s_per_lap, 3) + " s/lap</span><span>cliff " + pct(v.tyre__cliff_risk) + "</span>" +
      "<span>undercut threat " + pct(v.rivals__undercut_threat) + "</span><span>chance " + pct(v.rivals__undercut_chance) + "</span></div></div>";
  }).join("");
}

function renderAlerts(s) {
  const al = s.alerts || [], rc = s.extra.rc || [], order = {critical: 0, warn: 1, info: 2};
  $("alerts").innerHTML = al.length
    ? al.slice().sort((a, b) => order[a.severity] - order[b.severity]).map((a) =>
      '<div class="alert ' + esc(a.severity) + '">' + esc(a.message) + " <small>" + esc(a.engineer) + (a.car ? " #" + esc(a.car) : "") + "</small></div>").join("")
    : '<div class="empty">No active alerts.</div>';
  $("rc").innerHTML = rc.length
    ? rc.slice().reverse().map((m) => '<div class="rc"><b>' + fmtClock(m.t) + "</b> " + esc(m.message) + "</div>").join("")
    : '<div class="empty">None.</div>';
}

function render(s) {
  if (!s || s.waiting) return;
  s.extra = s.extra || {};
  snap = s;
  renderHeader(s); renderTower(s); renderFocus(s); renderAlerts(s);
}

document.querySelector("#tower tbody").addEventListener("click", (e) => {
  const tr = e.target.closest("tr[data-car]");
  if (!tr) return;
  const n = tr.dataset.car;
  if (pinned.has(n)) pinned.delete(n); else pinned.add(n);
  try { localStorage.setItem("pinned", JSON.stringify([...pinned])); } catch (e2) { /* no storage */ }
  if (snap) { renderTower(snap); renderFocus(snap); }
});

function conn(ok, text) { const c = $("conn"); c.textContent = text; c.className = "chip " + (ok ? "on" : "off"); }
function poll() {
  setInterval(() => fetch("/api/snapshot").then((r) => r.json()).then((s) => { conn(true, "polling"); render(s); })
    .catch(() => conn(false, "offline")), 2000);
}
if (window.EventSource) {
  const es = new EventSource("/api/stream");
  es.onopen = () => conn(true, "live");
  es.onerror = () => conn(false, "reconnecting");
  es.addEventListener("snapshot", (e) => render(JSON.parse(e.data)));
  es.addEventListener("status", (e) => { if (JSON.parse(e.data).status === "finished") conn(true, "feed ended"); });
} else poll();
