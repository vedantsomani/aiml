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
// Each viewer picks its own team (localStorage); "" = whatever the server was started with. Nothing is sent to the server.
let myTeam = "";
try { myTeam = localStorage.getItem("pitsense.team") || ""; } catch (e) { /* no storage */ }
const baseFocus = () => {
  const t = snap && snap.extra && snap.extra.teams;
  return myTeam === "-" ? [] : myTeam && t && t[myTeam] ? t[myTeam] : (snap && snap.focus) || [];
};
const focusName = () => (myTeam && myTeam !== "-" ? myTeam : myTeam === "-" ? "" : (snap && snap.extra && snap.extra.team) || "");
const focusSet = () => new Set(baseFocus().concat([...pinned]));
function syncTeamSel(s) {
  const sel = $("teamsel"), teams = Object.keys((s.extra || {}).teams || {});
  const sig = teams.join("|") + "#" + (s.extra.team || "") + "#" + myTeam;
  if (sel.dataset.sig === sig) return;
  sel.dataset.sig = sig;
  sel.innerHTML = '<option value="">' + esc(s.extra.team ? "team: " + s.extra.team + " (server)" : "team: server default") + '</option><option value="-">no team (pins only)</option>' +
    teams.map((t) => '<option value="' + esc(t) + '">' + esc(t) + "</option>").join("");
  sel.value = myTeam;
  if (sel.value !== myTeam) { myTeam = ""; sel.value = ""; }
}
// risk: the server's setting, shared by every screen
function renderRisk(s) {
  const r = (s.extra || {}).risk, sel = $("risksel");
  if (r && document.activeElement !== sel) sel.value = r;
}
$("risksel").addEventListener("change", (e) => {
  fetch("/api/risk", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({risk: e.target.value})});
});
$("teamsel").addEventListener("change", (e) => {
  myTeam = e.target.value;
  try { localStorage.setItem("pitsense.team", myTeam); } catch (e2) { /* no storage */ }
  convoSig = ""; seenMsgs.clear();
  if (snap) { renderTower(snap); renderFocus(snap); renderStrategy(snap); renderCallbar(snap); applyFocus(); renderConvo(snap); renderQuali(snap); }
});

// ---- phone tabs (CSS shows one section at a time under 700 px; on larger screens everything is visible)
let tab = "tower";
try { tab = localStorage.getItem("pitsense.tab") || "tower"; } catch (e) { /* no storage */ }
function setTab(t) {
  if (!document.querySelector('#tabs [data-tab="' + t + '"]')) t = "tower";
  tab = t;
  document.body.className = "tab-" + t;
  for (const b of document.querySelectorAll("#tabs button")) b.classList.toggle("on", b.dataset.tab === t);
  try { localStorage.setItem("pitsense.tab", t); } catch (e) { /* no storage */ }
  if (t === "map" && typeof scaleMap === "function") setTimeout(scaleMap, 0);
  if (t === "strategy") setTimeout(() => { if (snap) renderCharts(snap); }, 0);
}
$("tabs").addEventListener("click", (e) => { const b = e.target.closest("button[data-tab]"); if (b) setTab(b.dataset.tab); });
setTab(tab);

// sticky call banner: the current call for each focus car, visible on every tab
function renderCallbar(s) {
  const mine = [...focusSet()], row = {}, calls = {};
  for (const r of s.tower || []) row[r.car] = r;
  for (const c of s.calls || []) calls[c.car] = c;
  const cars = mine.filter((n) => row[n]);
  $("callbar").hidden = !cars.length;
  const acks = s.extra.acks || {}, changes = s.extra.changes || {};
  $("callbar").innerHTML = cars.map((n) => {
    const r = row[n], c = calls[n], act = c ? c.action : "NO_CALL", k = acks[n], ch = changes[n];
    // the operator's answer stands for this call while the call (action and tyre) is the same
    const same = k && c && k.action === c.action && (k.compound || null) === (c.compound || null);
    const ans = same ? '<span class="ack ' + esc(k.decision) + '" title="' + esc(k.reason || "") + '">' + (k.decision === "accept" ? "accepted" : "rejected") + "</span>"
      : act !== "NO_CALL" ? '<button class="ackb" data-car="' + esc(n) + '" data-d="accept" title="accept this call">&#10003;</button>' +
        '<button class="ackb" data-car="' + esc(n) + '" data-d="reject" title="reject this call (you can give a reason)">&#10007;</button>' : "";
    const why = ch ? '<small class="chg" title="' + esc(ch.why.join("; ")) + '">was ' + esc(ch.from.toLowerCase()) + ": " + esc(ch.why[0]) + "</small>" : "";
    return '<span class="cb"><b>' + esc(r.tla || n) + "</b> P" + (r.position == null ? "-" : r.position) + " " + action(act) + " " + ans + why + "</span>";
  }).join("");
}
// the operator's accept / reject: logged for the post-race review (pitsense shadow-score), never fed back to the engineers
$("callbar").addEventListener("click", (e) => {
  const b = e.target.closest(".ackb");
  if (!b) return;
  const d = b.dataset.d, reason = d === "reject" ? (window.prompt("Why reject this call? (optional)") || "") : "";
  b.disabled = true;
  fetch("/api/ack", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({car: b.dataset.car, decision: d, reason})})
    .then((r) => r.json()).then((r) => { if (!r.ok) b.disabled = false; }).catch(() => { b.disabled = false; });
});

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

// ---- qualifying panel: shown when the quali engineer is on the wall (race values quali__*)
const SEND = {SEND_NOW: ["SEND NOW", "critical"], TOO_LATE: ["too late", "dim"], WAIT: ["wait", ""], ON_TRACK: ["on track", "dim"], SAFE: ["", ""]};
function renderQuali(s) {
  const r = s.race || {}, on = r.quali__part != null;
  $("qpanel").hidden = !on;
  if (!on) return;
  const left = r.quali__time_left_s;
  const clk = left == null ? "--" : Math.floor(left / 60) + ":" + String(Math.floor(left % 60)).padStart(2, "0");
  $("lap").textContent = "Q" + r.quali__part + (r.quali__sprint ? " (sprint)" : "") + "  " + clk;
  $("qnote").textContent = r.quali__cut_pos ? "cut after P" + r.quali__cut_pos : "final part";
  const bits = [];
  if (r.quali__cut_time_s != null) bits.push("cut " + fmtTime(r.quali__cut_time_s));
  if (r.quali__pred_cut_s != null) bits.push("predicted " + fmtTime(r.quali__pred_cut_s));
  if (r.quali__bubble_s != null) bits.push("bubble " + num(r.quali__bubble_s, 3) + " s");
  bits.push(r.quali__n_on_track + " on track" + (r.quali__traffic_ahead != null ? ", " + r.quali__traffic_ahead + " just ahead of pit exit" : ""));
  $("qsum").textContent = bits.join("  |  ");
  const mine = focusSet(), tla = {}, rows = [];
  for (const t of s.tower || []) tla[t.car] = t.tla;
  const list = Object.keys(s.cars || {}).filter((n) => s.cars[n].quali__eligible).sort((a, b) => s.cars[a].quali__rank - s.cars[b].quali__rank);
  for (const n of list) {
    const v = s.cars[n], sd = SEND[v.quali__send] || ["", ""];
    rows.push('<tr class="' + (mine.has(n) ? "mine " : "") + (v.quali__in_zone ? "zone" : "") + '"><td class="p">' + v.quali__rank + "</td><td>" + esc(tla[n] || n) +
      "</td><td>" + fmtTime(v.quali__best_s) + "</td><td>" + (v.quali__gap_to_cut_s == null ? "--" : (v.quali__gap_to_cut_s > 0 ? "+" : "") + num(v.quali__gap_to_cut_s, 3)) +
      "</td><td>" + (v.quali__p_ko == null ? "--" : pct(v.quali__p_ko)) + '</td><td class="' + sd[1] + '">' + sd[0] +
      (v.quali__laps_needed && v.quali__send !== "SAFE" ? " (" + v.quali__laps_needed + " lap" + (v.quali__laps_needed > 1 ? "s" : "") + ")" : "") + "</td></tr>");
  }
  document.querySelector("#qtab tbody").innerHTML = rows.join("");
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
const asked = [];  // our questions and the answers from /api/ask (the snapshot catches up later)
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
  const wm = (x.wall_msgs || []).slice();
  for (const p of asked) if (!wm.some((w) => w.id === p.id)) wm.push(p);  // answers newer than the snapshot
  for (const m of wm) {
    if (m.kind === "you") {
      out.push({id: "y" + m.id, kind: "you", car: m.car, tla: tla[m.car] || m.car, t: m.t, lap: m.lap, text: m.text, voice: m.source === "voice", n: m.id});
    } else if (m.kind === "alert") {
      if (m.car ? !mine.has(m.car) : m.severity !== "critical") continue;
      out.push({id: "a" + m.id, kind: "alert", car: m.car, tla: m.car ? tla[m.car] || m.car : "", t: m.t, lap: m.lap,
        text: m.text, who: m.engineer, sev: m.severity, n: m.id});
    } else {
      if (!mine.has(m.car) && !m.ask) continue;
      out.push({id: "w" + m.id, kind: "wall", car: m.car, tla: tla[m.car] || m.car, t: m.t, lap: m.lap, text: m.text,
        action: m.action, src: m.ask ? m.source : null, url: "/api/radio.wav?car=" + encodeURIComponent(m.car) + "&i=" + m.id, n: m.id});
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
  if (m.kind === "you")
    return '<div class="msg you" data-id="' + esc(m.id) + '"><div class="who"><b>YOU</b>' + (m.voice ? " (voice)" : "") + " &rarr; " + esc(m.tla) +
      ' <span class="when">' + when + '</span></div><div class="bub">' + esc(m.text) + "</div></div>";
  if (m.kind === "alert")
    return '<div class="msg alert-msg ' + esc(m.sev) + '" data-id="' + esc(m.id) + '"><div class="who"><b>' + esc(m.who || "engineer") +
      "</b>" + (m.tla ? " &rarr; " + esc(m.tla) : "") + ' <span class="when">' + when + '</span></div><div class="bub">' + esc(m.text) + "</div></div>";
  return '<div class="msg wallmsg" data-id="' + esc(m.id) + '"><div class="who"><b>PIT WALL</b> &rarr; ' + esc(m.tla) +
    (m.action ? ' <span class="act ' + esc(m.action) + '">' + esc(String(m.action).replace(/_/g, " ")) + "</span>" : "") +
    (m.src ? ' <span class="src">' + esc(m.src) + "</span>" : "") +
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
  syncAskCars(s);
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

// ---- ask the pit wall: a text box and a push-to-talk microphone. Both end in the same conversation.
function syncAskCars(s) {
  const sel = $("askcar"), tla = {};
  for (const r of s.tower || []) tla[r.car] = r.tla;
  const cars = [...focusSet()].filter((n) => tla[n]);
  const sig = cars.join(",");
  if (sel.dataset.sig === sig) return;
  sel.dataset.sig = sig;
  const keep = sel.value;
  sel.innerHTML = cars.map((n) => '<option value="' + esc(n) + '">' + esc(tla[n]) + "</option>").join("");
  if (cars.includes(keep)) sel.value = keep;
  sel.hidden = cars.length < 2;
}
const askCar = () => $("askcar").value || baseFocus()[0] || [...pinned][0] || "";
function askStat(t) { $("askstat").textContent = t || ""; }
function handleAsk(r) {
  if (!r || !r.ok) { askStat((r && r.error) || "no answer"); return; }
  for (const m of [r.you, r.reply]) { asked.push(m); seenMsgs.add((m.kind === "you" ? "y" : "w") + m.id); }
  while (asked.length > 60) asked.shift();
  askStat((r.heard ? 'heard: "' + r.heard + '"   ' : "") + r.source + ", " + r.ms + " ms");
  convoSig = "";
  if (snap) renderConvo(snap);
  const box = $("convo");
  box.scrollTop = box.scrollHeight;
  enqueue({id: "w" + r.reply.id, kind: "wall", text: r.reply.text, url: r.audio});  // an answer is always spoken
}
function sendAsk(text) {
  askStat("asking the pit wall...");
  fetch("/api/ask", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({car: askCar(), text})})
    .then((r) => r.json()).then(handleAsk).catch(() => askStat("the pit wall did not answer"));
}
$("askform").addEventListener("submit", (e) => {
  e.preventDefault();
  const t = $("asktext").value.trim();
  if (!t) return;
  $("asktext").value = "";
  sendAsk(t);
});
let rec = null, micStream = null, micChunks = [], micWant = false, micT0 = 0;
async function micStart() {
  if (rec || micWant) return;
  micWant = true;
  if (!navigator.mediaDevices || !window.MediaRecorder) { micWant = false; askStat("the microphone needs a browser with MediaRecorder, on localhost or https"); return; }
  try { micStream = await navigator.mediaDevices.getUserMedia({audio: true}); }
  catch (e) { micWant = false; askStat("microphone blocked: " + e.name); return; }
  if (!micWant) { micStream.getTracks().forEach((t) => t.stop()); return; }  // released before it was ready
  const mime = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"].find((m) => MediaRecorder.isTypeSupported(m));
  rec = new MediaRecorder(micStream, mime ? {mimeType: mime} : {});
  micChunks = [];
  rec.ondataavailable = (e) => { if (e.data && e.data.size) micChunks.push(e.data); };
  rec.onstop = () => {
    micStream.getTracks().forEach((t) => t.stop());
    const blob = new Blob(micChunks, {type: (rec && rec.mimeType) || "audio/webm"});
    rec = null; micWant = false;
    $("micbtn").classList.remove("rec");
    if (Date.now() - micT0 < 400 || blob.size < 1000) { askStat("too short: hold the button while you speak"); return; }
    askStat("listening to the recording...");
    fetch("/api/ask_audio?car=" + encodeURIComponent(askCar()), {method: "POST", headers: {"Content-Type": blob.type}, body: blob})
      .then((r) => r.json()).then(handleAsk).catch(() => askStat("the pit wall did not answer"));
  };
  micT0 = Date.now();
  rec.start();
  $("micbtn").classList.add("rec");
  askStat("recording: release to send");
}
function micStop() {
  micWant = false;
  if (rec && rec.state === "recording") rec.stop();
}
const mic = $("micbtn");
mic.addEventListener("contextmenu", (e) => e.preventDefault());  // no long-press menu on touch
mic.addEventListener("pointerdown", (e) => { e.preventDefault(); try { mic.setPointerCapture(e.pointerId); } catch (e2) { /* ok */ } micStart(); });
for (const ev of ["pointerup", "pointerleave", "pointercancel"]) mic.addEventListener(ev, micStop);
mic.addEventListener("keydown", (e) => { if ((e.key === " " || e.key === "Enter") && !e.repeat) { e.preventDefault(); micStart(); } });
mic.addEventListener("keyup", (e) => { if (e.key === " " || e.key === "Enter") micStop(); });

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
  $("focusnote").textContent = baseFocus().length ? focusName() : "no team set: pick one above or pin cars from the tower";
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

// ---- strategy comparison: the simulator's best plans side by side, with the spread of outcomes
function rangeBar(p10, p90, mean, n) {
  if (mean == null) return "--";
  const x = (v) => (100 * (Math.min(n, Math.max(1, v)) - 1) / Math.max(1, n - 1)).toFixed(1);
  const band = p10 == null ? "" : '<i style="left:' + x(p10) + "%;width:" + Math.max(1, x(p90) - x(p10)) + '%"></i>';
  return '<div class="sc-range" title="P' + num(p10) + " to P" + num(p90) + ' in 80% of futures">' + band + '<b style="left:' + x(mean) + '%"></b></div>';
}
function renderStrategy(s) {
  const mine = [...focusSet()], row = {}, n = s.tower.filter((r) => r.running).length || 20;
  for (const r of s.tower) row[r.car] = r;
  const html = mine.filter((c) => row[c]).map((c) => {
    const v = (s.cars || {})[c] || {}, r = row[c], det = ((s.extra.details || {}).strategy || {})[c] || {}, opts = det.ranked || [];
    const head = '<h3>' + esc(r.tla || c) + ' <span class="kv">P<span>' + (r.position == null ? "-" : r.position) + "</span></span>" + tyre(r.compound, r.tyre_age) + "</h3>";
    if (!opts.length) return '<div class="sc-car">' + head + '<div class="empty">' + esc(v.strategy__why || "no plans yet") + "</div></div>";
    const rows = opts.map((o, i) => '<tr class="' + (i === 0 ? "best" : "") + '"><td>' + o.tags.map((t) => '<span class="sc-tag ' + esc(t.split(" ")[0]) + '">' + esc(t) + "</span>").join("") +
      "</td><td>" + esc(o.plan) + "</td><td>P" + num(o.exp_pos, 1) + "</td><td>" + rangeBar(o.p10, o.p90, o.exp_pos, n) +
      '</td><td class="' + (o.delta > 0.05 ? "worse" : o.delta < -0.05 ? "better" : "") + '">' + (o.delta == null ? "" : (o.delta > 0 ? "+" : "") + num(o.delta, 2)) +
      "</td><td>" + num(o.exp_pts, 1) + "</td></tr>").join("");
    const d = det.stop_dist || [], mx = Math.max(...d, 0.001);
    const dist = d.length ? '<div class="kv">when the simulator expects the first stop (laps from now)</div><div class="sc-dist">' +
      d.map((x, i) => '<div class="' + (i === 0 ? "cur" : "") + '" style="height:' + (100 * x / mx).toFixed(0) + '%" title="+' + i + " laps: " + pct(x) + '"></div>').join("") +
      '</div><div class="sc-axis">' + d.map((x, i) => "<span>" + (i === 0 ? "now" : "+" + i) + "</span>").join("") + "</div>" : "";
    const b = det.if_sc, sc = b ? '<div class="sc-b"><span class="sc-tag B">if SC</span> ' + esc(b.plan) + " &rarr; P" + num(b.exp_pos, 1) +
      (b.sc_gain != null ? ' <span class="kv">stopping under it gains <span>' + num(b.sc_gain, 2) + " places</span></span>" : "") +
      (b.trigger ? ' <span class="kv">' + esc(b.trigger) + "</span>" : "") + "</div>" : "";
    return '<div class="sc-car">' + head + '<table class="sc-tab"><thead><tr><th></th><th>PLAN</th><th>EXP</th><th>OUTCOMES (P1 &rarr; P' + n +
      ')</th><th>VS A</th><th>PTS</th></tr></thead><tbody>' + rows + "</tbody></table>" + sc + dist + "</div>";
  });
  $("strat").innerHTML = html.length ? html.join("") : '<div class="empty">Pick a team above or pin cars from the tower to compare strategies.</div>';
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

// ---- charts (Strategy tab): gap chart, stint chart, what-if, call history. Plain SVG; data from extra.charts and /api/calls.
const svgEl = (tag, attrs, kids) => {
  const e = document.createElementNS(SVGNS, tag);
  for (const k in attrs || {}) e.setAttribute(k, attrs[k]);
  for (const c of kids || []) e.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  return e;
};
const chartSvg = (w, h, kids) => svgEl("svg", {class: "chart", viewBox: "0 0 " + w + " " + h, preserveAspectRatio: "xMidYMid meet", role: "img"}, kids);
function gapChart(g, loss) {
  const W = 320, H = 170, L = 26, R = 34, T = 8, B = 16, mid = T + (H - T - B) / 2, half = (H - T - B) / 2;
  const n = g.laps.length;
  if (n < 2) return null;
  const lo = loss ? Math.min(loss.green, loss.now) : null, hi = loss ? Math.max(loss.green, loss.now) : null;
  const bl = lo == null ? null : hi - lo < 3 ? [(lo + hi) / 2 - 1.5, (lo + hi) / 2 + 1.5] : [lo, hi];
  const vals = [].concat(...g.ahead, ...g.behind).filter((v) => v != null);
  const ymax = Math.max(10, Math.min(60, Math.max(bl ? bl[1] * 1.25 : 0, Math.max(...vals, 0) * 1.05)));
  const x = (i) => L + (W - L - R) * i / (n - 1), y = (v, sign) => mid - sign * half * Math.min(v, ymax) / ymax;
  const kids = [];
  if (bl) for (const sg of [1, -1]) kids.push(svgEl("rect", {class: "band", x: L, width: W - L - R, y: Math.min(y(bl[0], sg), y(bl[1], sg)), height: Math.abs(y(bl[1], sg) - y(bl[0], sg))}));
  for (const v of [0, ymax / 2, ymax]) for (const sg of v ? [1, -1] : [1]) {
    kids.push(svgEl("line", {class: "grid", x1: L, x2: W - R, y1: y(v, sg), y2: y(v, sg)}));
    kids.push(svgEl("text", {x: L - 3, y: y(v, sg) + 3, "text-anchor": "end"}, [Math.round(v) + "s"]));
  }
  kids.push(svgEl("text", {x: L, y: H - 3}, ["lap " + g.laps[0]]));
  kids.push(svgEl("text", {x: W - R, y: H - 3, "text-anchor": "end"}, ["lap " + g.laps[n - 1]]));
  const shade = [1, 0.65, 0.4];
  for (const [rows, tl, sg, dash] of [[g.ahead, g.tla_ahead, 1, ""], [g.behind, g.tla_behind, -1, "4 2"]]) {
    for (let k = 2; k >= 0; k--) {
      let d = "", pen = false, last = null;
      rows.forEach((r, i) => { const v = r[k]; if (v == null) { pen = false; return; } d += (pen ? "L" : "M") + x(i).toFixed(1) + " " + y(v, sg).toFixed(1); pen = true; last = [i, v]; });
      if (!d) continue;
      const c = sg > 0 ? "var(--accent)" : "var(--orange)";
      const t = svgEl("path", {d, fill: "none", stroke: c, "stroke-width": k === 0 ? 2 : 1.2, opacity: shade[k], "stroke-dasharray": dash});
      t.appendChild(svgEl("title", {}, [(sg > 0 ? "ahead +" : "behind -") + (k + 1) + " place: " + (tl[n - 1][k] || "")]));
      kids.push(t);
      if (last && last[0] === n - 1) kids.push(svgEl("text", {x: W - R + 3, y: y(last[1], sg) + 3, fill: c}, [(tl[n - 1][k] || "") + " " + last[1].toFixed(1)]));
    }
  }
  return chartSvg(W, H, kids);
}
function renderGaps(s) {
  const ch = (s.extra || {}).charts || {}, mine = [...focusSet()], row = {};
  for (const r of s.tower || []) row[r.car] = r;
  const race = s.race || {}, loss = race.pitstop__loss_green != null && race.pitstop__loss_now != null ? {green: race.pitstop__loss_green, now: race.pitstop__loss_now} : null;
  const box = $("gapchart");
  box.textContent = "";
  let any = false;
  for (const c of mine) {
    const g = (ch.gaps || {})[c];
    if (!g || !row[c]) continue;
    any = true;
    const d = document.createElement("div");
    d.className = "gc-car";
    d.innerHTML = "<h3>" + esc(row[c].tla || c) + ' <span class="kv">P' + row[c].position + "</span></h3>";
    const svg = gapChart(g, loss);
    if (svg) d.appendChild(svg); else d.insertAdjacentHTML("beforeend", '<div class="empty">not enough laps yet</div>');
    box.appendChild(d);
  }
  if (!any) { box.innerHTML = '<div class="empty">Gaps appear once our cars have two timed laps.</div>'; return; }
  box.insertAdjacentHTML("beforeend", '<div class="ch-leg"><span><i style="background:var(--accent)"></i>ahead (solid)</span><span><i style="background:var(--orange)"></i>behind (dashed)</span>' +
    (loss ? "<span>band: pit loss " + num(Math.min(loss.green, loss.now)) + "-" + num(Math.max(loss.green, loss.now)) + " s. A car inside it is one stop away from swapping places.</span>" : "") + "</div>");
}
function renderStints(s) {
  const ch = (s.extra || {}).charts || {}, st = ch.stints || {}, order = (ch.order || []).filter((c) => st[c] && st[c].length);
  if (!order.length) { $("stintchart").innerHTML = '<div class="empty">No stints yet.</div>'; return; }
  const total = s.total_laps || Math.max(s.lap || 1, ...order.map((c) => st[c][st[c].length - 1][2]));
  const W = 320, L = 28, rh = 12, H = order.length * rh + 18, x = (lap) => L + (W - L - 4) * lap / total, mine = focusSet();
  const tla = {};
  for (const r of s.tower || []) tla[r.car] = r.tla || r.car;
  const kids = [];
  order.forEach((c, i) => {
    const g = svgEl("g", {class: mine.has(c) ? "me" : ""}), y = 2 + i * rh;
    g.appendChild(svgEl("text", {x: 0, y: y + 8}, [tla[c] || c]));
    for (const [k, a, b] of st[c]) {
      const r = svgEl("rect", {class: "bar " + (SHORT_K[k] ? k : "u"), x: x(a - 1), y, width: Math.max(1.5, x(b) - x(a - 1) - 0.6), height: rh - 3, rx: 2});
      r.appendChild(svgEl("title", {}, [(tla[c] || c) + " " + k + " laps " + a + "-" + b]));
      g.appendChild(r);
    }
    kids.push(g);
  });
  for (let l = 0; l <= total; l += 10) kids.push(svgEl("text", {x: x(l), y: H - 3, "text-anchor": "middle"}, [String(l)]));
  if (s.lap) kids.push(svgEl("line", {class: "now", x1: x(s.lap), x2: x(s.lap), y1: 0, y2: H - 12}));
  $("stintchart").textContent = "";
  $("stintchart").appendChild(chartSvg(W, H, kids));
  $("stintchart").insertAdjacentHTML("beforeend", '<div class="ch-leg">' + [["soft"], ["medium"], ["hard"], ["inter"], ["wet"]].map((c) => '<span><i style="background:var(--' + c[0] + ')"></i>' + c[0] + "</span>").join("") + "</div>");
}
const SHORT_K = {S: 1, M: 1, H: 1, I: 1, W: 1};
// what-if: POST /api/whatif {car, stop_lap, compound}
let wiBusy = false, wiLapSet = false;
function syncWiCars(s) {
  const sel = $("wicar"), cars = [...focusSet()].filter((c) => (s.tower || []).some((r) => r.car === c && r.running)), sig = cars.join(",");
  if (sel.dataset.sig === sig) return;
  const cur = sel.value;
  sel.dataset.sig = sig;
  const tla = {};
  for (const r of s.tower || []) tla[r.car] = r.tla || r.car;
  sel.innerHTML = cars.map((c) => '<option value="' + esc(c) + '">' + esc(tla[c]) + "</option>").join("");
  if (cars.includes(cur)) sel.value = cur;
}
function syncWiLap(s) {
  if (!wiLapSet && s.lap) { $("wilap").value = Math.min(s.total_laps || 99, s.lap + 3); wiLapSet = true; }
  if (s.total_laps) $("wilap").max = s.total_laps;
}
function wiResult(r) {
  if (!r || !r.ok) return '<div class="empty">' + esc((r && r.error) || "no answer") + "</div>";
  const sc = r.scenario || {}, v = r.versus || {}, dp = r.delta_pos;
  if (r.wet) {  // the wet simulator: the car's own race time over the next laps, not positions
    const dt = r.delta_time_s, wc = dt < -0.5 ? "better" : dt > 0.5 ? "worse" : "";
    return '<div><span class="kv">' + esc(r.tla) + " " + esc(sc.plan || "") + '</span></div><div><span class="kv">vs plan A <span class="' + wc + '">' +
      (dt == null ? "--" : (dt > 0 ? "+" : "") + num(dt, 1) + " s") + "</span></span>" +
      '<span class="kv">next <span>' + esc(r.horizon_laps) + " laps</span></span>" +
      (r.p_gain != null ? '<span class="kv">better in <span>' + pct(r.p_gain) + "</span></span>" : "") + "</div>" +
      (v.plan ? "<p>plan A: " + esc(v.plan) + "</p>" : "") + (r.notes || []).map((n) => "<p>" + esc(n) + "</p>").join("") +
      "<p class=\"dim\">wet simulator: race time only</p>";
  }
  const cls = dp > 0.05 ? "better" : dp < -0.05 ? "worse" : "";
  return '<div><span class="kv">' + esc(r.tla) + " " + esc(sc.plan || "") + '</span></div><div><span class="kv">expected <span>P' + num(sc.exp_pos, 1) + '</span></span><span class="kv">P10-P90 <span>P' + num(sc.p10, 0) + "-P" + num(sc.p90, 0) + "</span></span>" +
    '<span class="kv">vs plan A <span class="' + cls + '">' + (dp == null ? "--" : (dp > 0 ? "+" : "") + num(dp, 2) + " places") + "</span></span>" +
    (r.p_gain != null ? '<span class="kv">better in <span>' + pct(r.p_gain) + "</span></span>" : "") + "</div>" +
    (v.plan ? "<p>plan A: " + esc(v.plan) + " &rarr; P" + num(v.exp_pos, 1) + " (P" + num(v.p10, 0) + "-P" + num(v.p90, 0) + ")</p>" : "") +
    (r.notes || []).map((n) => "<p>" + esc(n) + "</p>").join("");
}
$("wiform").addEventListener("submit", (e) => {
  e.preventDefault();
  if (wiBusy) return;
  wiBusy = true; $("wibtn").disabled = true;
  $("wiout").innerHTML = '<div class="empty">simulating...</div>';
  fetch("/api/whatif", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({car: $("wicar").value, stop_lap: +$("wilap").value, compound: $("wicomp").value || null})})
    .then((r) => r.json()).then((r) => { $("wiout").innerHTML = wiResult(r); })
    .catch(() => { $("wiout").innerHTML = '<div class="empty">the pit wall did not answer</div>'; })
    .then(() => { wiBusy = false; $("wibtn").disabled = false; });
});
// call history: /api/calls, refetched when the lap or the calls change
let callsLog = [], callsSig = "", callsBusy = false;
const CALL_COL = {BOX: "var(--red)", PREPARE_BOX: "var(--orange)", BOX_IF_SC: "var(--purple)", STAY_OUT: "var(--green)"};
function drawCalls(s) {
  const mine = [...focusSet()].filter((c) => (s.tower || []).some((r) => r.car === c)), total = s.total_laps || Math.max(s.lap || 1, 10);
  if (!mine.length) { $("callhist").innerHTML = '<div class="empty">Pick a team to see its calls.</div>'; return; }
  const tla = {};
  for (const r of s.tower || []) tla[r.car] = r.tla || r.car;
  const W = 320, L = 28, rh = 26, H = mine.length * rh + 18, x = (lap) => L + (W - L - 6) * (lap - 1) / Math.max(1, total - 1), kids = [];
  const real = (r) => r.kind === "call" && r.action !== "NO_CALL";
  mine.forEach((c, i) => {
    const y = 4 + i * rh + rh / 2;
    kids.push(svgEl("text", {x: 0, y: y + 3}, [tla[c] || c]), svgEl("line", {class: "grid", x1: L, x2: W - 6, y1: y, y2: y}));
    callsLog.filter((r) => real(r) && r.car === c).forEach((r, j) => {
      const lap = r.lap || 1, dot = svgEl("circle", {cx: x(lap), cy: y + (j % 2 ? 5 : -5), r: 4.5, fill: CALL_COL[r.action] || "var(--blue)"});
      dot.appendChild(svgEl("title", {}, ["lap " + lap + ": " + r.action.replace(/_/g, " ") + (r.compound ? " " + r.compound : "")]));
      kids.push(dot);
    });
  });
  for (let l = 1; l <= total; l += l === 1 ? 9 : 10) kids.push(svgEl("text", {x: x(l), y: H - 3, "text-anchor": "middle"}, [String(l)]));
  if (s.lap) kids.push(svgEl("line", {class: "now", x1: x(s.lap), x2: x(s.lap), y1: 0, y2: H - 12}));
  const n = callsLog.filter((r) => real(r) && mine.includes(r.car)).length;
  $("callhist").textContent = "";
  $("callhist").appendChild(chartSvg(W, H, kids));
  $("callhist").insertAdjacentHTML("beforeend", '<div class="ch-leg">' + Object.entries(CALL_COL).map((c) => '<span><i style="background:' + c[1] + '"></i>' + c[0].replace(/_/g, " ").toLowerCase() + "</span>").join("") + "<span>" + n + " calls so far</span></div>");
}
function renderCallHistory(s) {
  const sig = s.lap + ":" + (s.calls || []).map((c) => c.car + c.action).join(",");
  if (sig === callsSig || callsBusy) return;
  callsSig = sig;
  callsBusy = true;
  fetch("/api/calls").then((r) => r.json()).then((r) => { callsLog = r.log || []; }).catch(() => {}).then(() => { callsBusy = false; if (snap) drawCalls(snap); });
}
function renderCharts(s) {
  if (tab !== "strategy" && window.innerWidth <= 760) return;  // phones show one tab at a time; redrawn when the tab opens
  renderGaps(s); renderStints(s); syncWiCars(s); syncWiLap(s); renderCallHistory(s);
}

function render(s) {
  if (!s || s.waiting) return;
  s.extra = s.extra || {};
  snap = s;
  syncTeamSel(s); renderHeader(s); renderQuali(s); renderTower(s); renderFocus(s); renderStrategy(s); renderCharts(s); renderCallbar(s); renderRisk(s); renderAlerts(s);
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
  if (snap) { renderTower(snap); renderFocus(snap); renderCallbar(snap); applyFocus(); renderConvo(snap); }
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

// the sticky call banner sits right under the (variable height) sticky header
function fitBars() { document.documentElement.style.setProperty("--hh", document.querySelector("header").offsetHeight + "px"); }
window.addEventListener("resize", fitBars); fitBars();

// ---- health alarms: a red bar listing them, green when clear
// alarms:begin
function alarmView(alarms) {
  const MAP = {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"};
  const e = (x) => String(x == null ? "" : x).replace(/[&<>"']/g, (c) => MAP[c]);
  if (!alarms || !alarms.length) return {cls: "green", html: "<b>HEALTH</b> No alarms"};
  return {cls: "red", html: "<b>" + alarms.length + (alarms.length === 1 ? " ALARM" : " ALARMS") + "</b><ul>" +
    alarms.map((a) => "<li>" + e(a) + "</li>").join("") + "</ul>"};
}
// alarms:end
function pollHealth() {
  const bar = $("alarmbar");
  fetch("/api/health").then((r) => r.json()).then((h) => {
    const v = alarmView(h.alarms);
    bar.className = "alarmbar " + v.cls;
    bar.innerHTML = v.html;
    fitBars();
  }).catch(() => {
    bar.className = "alarmbar red";
    bar.innerHTML = "<b>1 ALARM</b><ul><li>RED: the pit wall server does not answer</li></ul>";
    fitBars();
  });
}
pollHealth(); setInterval(pollHealth, 3000);

// ---- replay lab: pause, speed, lap slider, jump, bookmarks on a timeline
let rmarks = null, rbBusy = false;
const MARK_TOP = {call: 3, pit: 12, mechanic: 21};
function replayPost(path, body) {
  return fetch("/api/replay/" + path, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body || {})})
    .then((r) => r.json()).catch(() => ({ok: false, error: "no answer"}));
}
function drawTimeline(m) {
  const tl = $("timeline"), total = m.total_laps || Math.max(m.lap, 1);
  tl.querySelectorAll(".tl-mark").forEach((n) => n.remove());
  for (const k of m.marks) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "tl-mark " + k.kind + (k.ahead ? " ahead" : "");
    b.title = "lap " + k.lap + "  " + k.label;
    b.dataset.id = k.id;
    b.style.left = Math.min(100, (k.lap / total) * 100) + "%";
    b.style.top = (MARK_TOP[k.kind] != null ? MARK_TOP[k.kind] : 12) + "px";
    tl.appendChild(b);
  }
  $("tl-cur").style.left = Math.min(100, (m.lap / total) * 100) + "%";
}
function renderReplay(m) {
  $("replaybar").hidden = !m.enabled;
  if (!m.enabled) return;
  rmarks = m;
  const pb = $("rb-pause");
  pb.textContent = m.paused ? "Resume" : "Pause";
  pb.classList.toggle("on", m.paused);
  document.querySelectorAll("#rb-speeds button").forEach((b) => b.classList.toggle("on", (b.dataset.speed === "max" ? 0 : +b.dataset.speed) === m.speed));
  const sl = $("rb-lap");
  sl.max = m.total_laps || Math.max(m.lap, 1);
  if (document.activeElement !== sl) sl.value = m.lap;
  $("rb-num").max = sl.max;
  drawTimeline(m);
  if (!rbBusy) $("rb-note").textContent = "lap " + m.lap + " / " + (m.total_laps || "?") + (m.index_done ? "  | every lap ready" : "  | ready to lap " + m.indexed_lap);
  fitBars();
}
function pollReplay() {
  fetch("/api/replay/marks").then((r) => r.json()).then(renderReplay).catch(() => {});
}
function seekTo(body) {
  rbBusy = true;
  $("rb-note").textContent = "jumping...";
  replayPost("seek", body).then((r) => {
    rbBusy = false;
    $("rb-note").textContent = r.ok ? "jumped to lap " + r.lap + " in " + r.ms + " ms (" + r.via + ")" : "jump failed: " + (r.error || "");
    pollReplay();
  });
}
$("rb-pause").addEventListener("click", () => replayPost(rmarks && rmarks.paused ? "resume" : "pause").then(pollReplay));
$("rb-speeds").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) replayPost("speed", {speed: b.dataset.speed}).then(pollReplay); });
$("rb-lap").addEventListener("change", (e) => seekTo({lap: +e.target.value}));
$("rb-go").addEventListener("click", () => seekTo({lap: +$("rb-num").value}));
$("rb-num").addEventListener("keydown", (e) => { if (e.key === "Enter") seekTo({lap: +e.target.value}); });
$("timeline").addEventListener("click", (e) => {
  const b = e.target.closest(".tl-mark");
  if (b) seekTo({mark: b.dataset.id});
  else if (rmarks) {
    const r = $("timeline").getBoundingClientRect();
    seekTo({lap: Math.round(((e.clientX - r.left) / r.width) * (rmarks.total_laps || rmarks.lap))});
  }
});
pollReplay(); setInterval(pollReplay, 4000);
