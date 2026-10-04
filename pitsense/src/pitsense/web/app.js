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

// ---- radio: the conversation's clips are played one after another, never over each other.
// Driver clips are the real team-radio mp3s; our pit wall's are the Piper rendering of the voice text
// (else the browser's own voice). Nothing is ever synthesised as driver speech.
let radioOn = false;
try { radioOn = localStorage.getItem("pitsense.radio") === "1"; } catch (e) {}
function radioHtml(s, n) {
  const r = ((s.extra || {}).radio || {})[n];
  return r ? '<div class="radio">&#128251; <i>"' + esc(r.text) + '"</i> <span class="kv">lap ' + esc(r.lap) + "</span></div>" : "";
}
const radioQueue = [];
let radioBusy = false, playingId = null, curAudio = null;
function markPlaying(id) {
  playingId = id;
  document.querySelectorAll("#convo .msg.playing").forEach((n) => n.classList.remove("playing"));
  if (id != null) { const n = document.querySelector('#convo .msg[data-id="' + id + '"]'); if (n) n.classList.add("playing"); }
}
function enqueue(m) {
  if (radioQueue.some((x) => x.id === m.id) || playingId === m.id) return;
  radioQueue.push(m);
  while (radioQueue.length > 6) radioQueue.shift();  // at high replay speed, drop the oldest
  playNext();
}
function playNext() {
  if (radioBusy || !radioQueue.length) return;
  const m = radioQueue.shift();
  radioBusy = true;
  markPlaying(m.id);
  const done = () => { radioBusy = false; curAudio = null; markPlaying(null); playNext(); };
  const browserVoice = () => {
    if (m.kind !== "wall" || !window.speechSynthesis) return done();  // never voice a driver
    const u = new SpeechSynthesisUtterance(m.text);
    u.rate = 1.05; u.onend = done; u.onerror = done;
    window.speechSynthesis.speak(u);
  };
  const a = new Audio(m.url);
  curAudio = a;
  a.onended = done;
  a.onerror = m.kind === "wall" ? browserVoice : done;
  a.play().catch(m.kind === "wall" ? browserVoice : done);
}
function flipRadio(btn) {
  radioOn = !radioOn;
  try { localStorage.setItem("pitsense.radio", radioOn ? "1" : "0"); } catch (e) {}
  btn.textContent = radioOn ? "radio on" : "radio off";
  if (!radioOn) { radioQueue.length = 0; if (curAudio) { curAudio.pause(); curAudio = null; } radioBusy = false; markPlaying(null); }
}

// ---- conversation: real driver radio + our pit wall (voice calls, engineer alerts), oldest first
let radioList = [];
const seenMsgs = new Set();
let convoSig = "", convoItems = [];
function convoEntries(s) {
  const mine = focusSet(), x = s.extra, out = [], tla = {};
  for (const r of s.tower || []) tla[r.car] = r.tla;
  for (const m of radioList) {
    if (!mine.has(m.car)) continue;
    out.push({id: "d" + m.id, kind: "driver", car: m.car, tla: m.tla || tla[m.car] || m.car, t: m.t, lap: m.lap,
      text: m.text, url: m.audio, n: 0});
  }
  for (const m of x.wall_msgs || []) {
    if (m.kind === "alert") {
      if (m.car ? !mine.has(m.car) : m.severity !== "critical") continue;
      out.push({id: "a" + m.id, kind: "alert", car: m.car, tla: m.car ? tla[m.car] || m.car : "", t: m.t, lap: m.lap,
        text: m.text, who: m.engineer, sev: m.severity, n: m.id});
    } else {
      if (!mine.has(m.car)) continue;
      out.push({id: "w" + m.id, kind: "wall", car: m.car, tla: tla[m.car] || m.car, t: m.t, lap: m.lap, text: m.text,
        action: m.action, url: "/api/radio.wav?car=" + encodeURIComponent(m.car) + "&i=" + m.id, n: m.id});
    }
  }
  return out.sort((a, b) => a.t - b.t || a.n - b.n);
}
function msgHtml(m, colours) {
  const col = colours[m.car], when = "L" + esc(m.lap) + " " + fmtClock(m.t);
  const play = m.url ? '<button class="play" data-id="' + esc(m.id) + '" title="play">&#9654;</button>' : "";
  if (m.kind === "driver") {
    const txt = m.text != null ? esc(m.text) : '<i class="dim">radio message, transcript pending</i>';
    return '<div class="msg driver" data-id="' + esc(m.id) + '"><div class="who"><span class="team" style="background:' +
      (col ? "#" + esc(col) : "#444") + '"></span><b>' + esc(m.tla) + '</b> driver <span class="when">' + when + "</span></div>" +
      '<div class="bub">' + play + '<span class="txt">' + txt + "</span></div></div>";
  }
  if (m.kind === "alert")
    return '<div class="msg alert-msg ' + esc(m.sev) + '" data-id="' + esc(m.id) + '"><div class="who"><b>' + esc(m.who || "engineer") +
      "</b>" + (m.tla ? " &rarr; " + esc(m.tla) : "") + ' <span class="when">' + when + '</span></div><div class="bub">' + esc(m.text) + "</div></div>";
  return '<div class="msg wallmsg" data-id="' + esc(m.id) + '"><div class="who"><b>PIT WALL</b> &rarr; ' + esc(m.tla) +
    (m.action ? ' <span class="act ' + esc(m.action) + '">' + esc(String(m.action).replace(/_/g, " ")) + "</span>" : "") +
    ' <span class="when">' + when + '</span></div><div class="bub">' + play + '<span class="txt">' + esc(m.text) + "</span></div></div>";
}
function renderConvo(s) {
  const items = convoEntries(s);
  // new clips for our cars are queued when radio is on (the backlog on connect is never read out)
  const first = seenMsgs.size === 0;
  const fresh = [];
  for (const m of radioList) if (!seenMsgs.has("d" + m.id)) { seenMsgs.add("d" + m.id); fresh.push("d" + m.id); }
  for (const m of s.extra.wall_msgs || []) {
    const k = (m.kind === "alert" ? "a" : "w") + m.id;
    if (!seenMsgs.has(k)) { seenMsgs.add(k); fresh.push(k); }
  }
  if (!first && radioOn) for (const m of items) if (fresh.includes(m.id) && m.url) enqueue(m);
  const sig = items.map((m) => m.id + (m.text != null ? "+" : "-")).join(",") + "|" + [...focusSet()].join(",");
  if (sig === convoSig) return;
  convoSig = sig;
  convoItems = items;
  const box = $("convo"), near = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
  box.innerHTML = items.length ? items.map((m) => msgHtml(m, s.extra.colours || {})).join("")
    : '<div class="empty">No messages yet for ' + ([...focusSet()].length ? "our cars" : "any car: set --team or pin cars") + ".</div>";
  if (near || first) box.scrollTop = box.scrollHeight;
  markPlaying(playingId);
}
$("convo").addEventListener("click", (e) => {
  const b = e.target.closest("button.play");
  if (!b) return;
  const m = convoItems.find((x) => x.id === b.dataset.id);
  if (m && m.url) enqueue(m);
});
$("radiobtn").textContent = radioOn ? "radio on" : "radio off";

// ---- track map: outline, pit lane, start line and one dot per car, animated between position updates.
// Without an outline the map draws the trails the cars leave. The view is rotated so the circuit's long
// axis is horizontal, and fitted to the panel.
const SVGNS = "http://www.w3.org/2000/svg";
const MAP = {key: null, rot: null, box: null, cars: {}, last: null, anim: false, k: 1, trail: {}, trailAt: 0, hover: null, vb: null};
const el = (tag, attrs, parent) => {
  const n = document.createElementNS(SVGNS, tag);
  for (const k in attrs || {}) n.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(n);
  return n;
};
function pcaAngle(xs, ys) {
  let mx = 0, my = 0;
  const n = xs.length;
  for (let i = 0; i < n; i++) { mx += xs[i]; my += ys[i]; }
  mx /= n; my /= n;
  let sxx = 0, syy = 0, sxy = 0;
  for (let i = 0; i < n; i++) { const dx = xs[i] - mx, dy = ys[i] - my; sxx += dx * dx; syy += dy * dy; sxy += dx * dy; }
  return 0.5 * Math.atan2(2 * sxy, sxx - syy);
}
const mapXY = (x, y) => {
  const c = Math.cos(MAP.rot), sn = Math.sin(MAP.rot);
  return [x * c + y * sn, -(-x * sn + y * c)];  // along the long axis; Y flipped (the feed's Y points up)
};
const pathOf = (xs, ys) => xs.map((x, i) => { const p = mapXY(x, ys[i]); return (i ? "L" : "M") + p[0].toFixed(0) + " " + p[1].toFixed(0); }).join("");
function setBox(xs, ys) {
  let x0 = 1e18, x1 = -1e18, y0 = 1e18, y1 = -1e18;
  for (let i = 0; i < xs.length; i++) {
    const p = mapXY(xs[i], ys[i]);
    if (p[0] < x0) x0 = p[0]; if (p[0] > x1) x1 = p[0]; if (p[1] < y0) y0 = p[1]; if (p[1] > y1) y1 = p[1];
  }
  const pad = Math.max(x1 - x0, y1 - y0) * 0.06 + 1;
  MAP.box = [x0 - pad, y0 - pad, x1 - x0 + 2 * pad, y1 - y0 + 2 * pad];
  $("map").setAttribute("viewBox", MAP.box.join(" "));
  scaleMap();
}
function scaleMap() {  // dots and labels keep their pixel size whatever the circuit's size
  const svg = $("map"), b = MAP.box;
  if (!b || !svg.clientWidth) return;
  MAP.k = Math.max(b[2] / svg.clientWidth, b[3] / svg.clientHeight);
  for (const n in MAP.cars) sizeCar(MAP.cars[n]);
  const sf = svg.querySelector(".sf");
  if (sf) sf.setAttribute("stroke-width", 2.5 * MAP.k);
}
function sizeCar(c) {
  const k = MAP.k, r = (c.mine ? 6.5 : 3.6) * k;
  c.dot.setAttribute("r", r);
  c.dot.setAttribute("stroke-width", (c.mine ? 2 : 0.8) * k);
  c.txt.setAttribute("font-size", 11 * k);
  c.txt.setAttribute("x", r + 3 * k);
  c.txt.setAttribute("y", 4 * k);
}
function buildMap(s) {
  const tr = (s.extra || {}).track, svg = $("map");
  const key = tr ? tr.key || "t" : "trail";
  if (MAP.key === key) return;
  MAP.key = key;
  svg.innerHTML = "";
  MAP.cars = {}; MAP.last = null;
  MAP.layers = {track: el("g", {}, svg), cars: el("g", {}, svg)};
  if (!tr || !tr.x || tr.x.length < 10) {
    MAP.rot = null; MAP.box = null; MAP.trail = {};
    $("mapnote").textContent = "no outline for this circuit yet: drawing the cars' trails";
    return;
  }
  $("mapnote").textContent = "";
  MAP.rot = pcaAngle(tr.x, tr.y);
  const g = MAP.layers.track;
  el("path", {d: pathOf(tr.x, tr.y) + "Z", class: "outline-bed"}, g);
  el("path", {d: pathOf(tr.x, tr.y) + "Z", class: "outline"}, g);
  if (tr.pit && tr.pit.x && tr.pit.x.length > 2) el("path", {d: pathOf(tr.pit.x, tr.pit.y), class: "pitlane"}, g);
  // start/finish: a short bar across the track at the first outline point
  const a = mapXY(tr.x[0], tr.y[0]), b = mapXY(tr.x[2], tr.y[2]);
  const dx = b[0] - a[0], dy = b[1] - a[1], L = Math.hypot(dx, dy) || 1;
  const bar = el("line", {class: "sf", x1: 0, y1: 0, x2: 0, y2: 0}, g);
  MAP.sf = {a, nx: -dy / L, ny: dx / L};
  setBox(tr.x, tr.y);
  const bw = MAP.box[2] * 0.018;
  bar.setAttribute("x1", a[0] - MAP.sf.nx * bw); bar.setAttribute("y1", a[1] - MAP.sf.ny * bw);
  bar.setAttribute("x2", a[0] + MAP.sf.nx * bw); bar.setAttribute("y2", a[1] + MAP.sf.ny * bw);
  scaleMap();
}
function carNode(n, s) {
  let c = MAP.cars[n];
  if (c) return c;
  const g = el("g", {class: "car"}, MAP.layers.cars);
  const col = ((s.extra || {}).colours || {})[n];
  const dot = el("circle", {class: "dot", fill: col ? "#" + col : "#888"}, g);
  const txt = el("text", {class: "tla"}, g);
  c = MAP.cars[n] = {g, dot, txt, x: null, y: null, fx: 0, fy: 0, tx: 0, ty: 0, t0: 0, dur: 1, on: 1, mine: false, tla: n};
  g.addEventListener("pointerenter", (e) => { MAP.hover = n; showTip(e); });
  g.addEventListener("pointermove", showTip);
  g.addEventListener("pointerleave", () => { MAP.hover = null; $("maptip").hidden = true; });
  return c;
}
function applyFocus() {
  if (!snap) return;
  const mine = focusSet(), tla = {};
  for (const r of snap.tower || []) tla[r.car] = r;
  for (const n in MAP.cars) {
    const c = MAP.cars[n];
    c.mine = mine.has(n);
    c.tla = (tla[n] && tla[n].tla) || n;
    c.txt.textContent = c.mine ? c.tla : "";
    c.g.classList.toggle("mine", c.mine);
    if (c.mine) MAP.layers.cars.appendChild(c.g);  // focus cars on top
    sizeCar(c);
  }
}
function onPositions(m) {
  if (!m || !m.cars || !snap) return;
  buildMap(snap);
  if (MAP.rot == null) trailUpdate(m.cars);
  const now = performance.now(), gap = MAP.last == null ? 600 : Math.min(Math.max(now - MAP.last, 120), 1500);
  MAP.last = now;
  let created = false;
  for (const n in m.cars) {
    const p = m.cars[n];
    if (MAP.rot == null && !MAP.box) continue;
    const c = MAP.cars[n] || (created = true, carNode(n, snap));
    const q = mapXY(p[0], p[1]);
    if (c.x == null) { c.x = q[0]; c.y = q[1]; } else { c.x = c.fx + (c.tx - c.fx) * Math.min((now - c.t0) / c.dur, 1); c.y = c.fy + (c.ty - c.fy) * Math.min((now - c.t0) / c.dur, 1); }
    c.fx = c.x; c.fy = c.y; c.tx = q[0]; c.ty = q[1]; c.t0 = now; c.dur = gap;
    c.on = p[2];
    c.g.classList.toggle("off", !p[2]);
  }
  if (created) applyFocus();
  if (!MAP.anim) { MAP.anim = true; requestAnimationFrame(frame); }
}
function frame(now) {
  let moving = false;
  for (const n in MAP.cars) {
    const c = MAP.cars[n];
    if (c.x == null) continue;
    const f = Math.min((now - c.t0) / c.dur, 1);
    c.x = c.fx + (c.tx - c.fx) * f; c.y = c.fy + (c.ty - c.fy) * f;
    c.g.setAttribute("transform", "translate(" + c.x.toFixed(0) + " " + c.y.toFixed(0) + ")");
    if (f < 1) moving = true;
  }
  if (moving && !document.hidden) requestAnimationFrame(frame); else MAP.anim = false;
}
// fallback when there is no outline: the trail every car leaves
function trailUpdate(cars) {
  for (const n in cars) {
    const p = cars[n], t = MAP.trail[n] || (MAP.trail[n] = []), l = t[t.length - 1];
    if (!p[2]) continue;
    if (!l || Math.hypot(p[0] - l[0], p[1] - l[1]) > 150) { t.push([p[0], p[1]]); if (t.length > 500) t.shift(); }
  }
  const now = performance.now();
  if (now - MAP.trailAt < 2000 && MAP.box) return;
  MAP.trailAt = now;
  const xs = [], ys = [];
  for (const n in MAP.trail) for (const q of MAP.trail[n]) { xs.push(q[0]); ys.push(q[1]); }
  if (xs.length < 30) return;
  if (MAP.rot == null || MAP.rotLocked !== true) { MAP.rot = pcaAngle(xs, ys); MAP.rotLocked = xs.length > 300; }
  const g = MAP.layers.track;
  g.innerHTML = "";
  for (const n in MAP.trail) {
    const t = MAP.trail[n];
    if (t.length > 1) el("path", {d: pathOf(t.map((q) => q[0]), t.map((q) => q[1])), class: "trail"}, g);
  }
  setBox(xs, ys);
}
function showTip(e) {
  const n = MAP.hover, tip = $("maptip");
  if (!n || !snap) return;
  const rows = snap.tower || [], me = rows.find((r) => r.car === n);
  if (!me) return;
  const ahead = rows.find((r) => r.position === me.position - 1 && r.running);
  const behind = rows.find((r) => r.position === me.position + 1 && r.running);
  const line = (lab, r, gap) => r ? "<div>" + lab + " <b>" + esc(r.tla || r.car) + "</b> " + fmtGap(gap, 0) + "</div>" : "";
  tip.innerHTML = "<b>" + esc(me.tla || n) + "</b> P" + (me.position == null ? "-" : me.position) + " " + tyre(me.compound, me.tyre_age) +
    (me.position === 1 ? "<div>leader</div>" : line("ahead", ahead, me.interval)) + line("behind", behind, behind && behind.interval);
  tip.hidden = false;
  const box = $("mapwrap").getBoundingClientRect();
  tip.style.left = Math.min(e.clientX - box.left + 12, box.width - 150) + "px";
  tip.style.top = Math.max(e.clientY - box.top - 50, 4) + "px";
}
function renderMap(s) {
  buildMap(s);
  applyFocus();
  if (MAP.hover) { /* keep the tooltip's numbers current */ const c = MAP.cars[MAP.hover]; if (c) { const r = c.g.getBoundingClientRect(); showTip({clientX: r.left, clientY: r.top}); } }
}
window.addEventListener("resize", scaleMap);

function renderFocus(s) {
  const mine = [...focusSet()];
  $("focusnote").textContent = (s.focus || []).length ? (s.extra.team || "") : "no team set: pin cars from the tower";
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
  if (s.extra.team_radio) radioList = s.extra.team_radio;
  if (s.extra.positions) onPositions(s.extra.positions);
  renderMap(s); renderConvo(s);
}

document.querySelector("#tower tbody").addEventListener("click", (e) => {
  const tr = e.target.closest("tr[data-car]");
  if (!tr) return;
  const n = tr.dataset.car;
  if (pinned.has(n)) pinned.delete(n); else pinned.add(n);
  try { localStorage.setItem("pinned", JSON.stringify([...pinned])); } catch (e2) { /* no storage */ }
  if (snap) { renderTower(snap); renderFocus(snap); applyFocus(); renderConvo(snap); }
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
  es.addEventListener("pos", (e) => {
    const m = JSON.parse(e.data);
    if (m.team_radio) { radioList = m.team_radio; if (snap) renderConvo(snap); }
    onPositions(m);
  });
  es.addEventListener("status", (e) => { if (JSON.parse(e.data).status === "finished") conn(true, "feed ended"); });
} else poll();
