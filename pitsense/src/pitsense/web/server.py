"""HTTP server for the pit wall: static page, JSON API and a server-sent-events stream.

    GET /                  dashboard
    GET /api/snapshot      latest snapshot (the full Snapshot dict plus an "extra" block)
    GET /api/calls         {"current": [...], "log": [...]}
    GET /api/alerts        {"active": [...], "log": [...]}
    GET /api/health        loop status, throughput, snapshot latency, model bundle
    GET /api/replay/marks  replay controls state and bookmarks (calls, stops, SC/VSC, rain, mechanic alerts); enabled=false when live
    POST /api/replay/pause|resume|speed|seek   {} | {} | {"speed": 1|5|20|"max"} | {"lap": N} or {"mark": id}; 409 in live mode
    GET /api/stream        server-sent events: "snapshot" (the snapshot JSON), "pos" (car positions, ~3 Hz), "status"
    GET /api/radio.wav     ?car=N[&i=ID]: our pit wall's message (latest, or by id), spoken (Piper TTS; 404 if not installed)
    GET /api/teamradio     ?car=N&i=K: the K-th real driver radio mp3 of car N (published clips only, audio/mpeg)
    POST /api/ask          {"car": "16", "text": "what if we box now?"}: answer a what-if or fact question; the Q&A joins the
                           conversation and the answer can be spoken via /api/radio.wav?car=N&i=<reply id>
    POST /api/ask_audio    ?car=N, body = recorded audio (webm/opus, ogg, wav): transcribed on the server, then as /api/ask

The GET endpoints are read-only; the ask POSTs only add a question and its answer to the conversation; the replay
POSTs move a replay (never a live session). Binds to
127.0.0.1 unless told otherwise; a POST from a page on another origin is refused.

Many browsers may watch at once: each SSE client has its own thread and a bounded queue (a slow one loses its oldest
messages, a stalled one is cut after WRITE_TIMEOUT_S), so the publishing loop never waits for a viewer. With a token
set, every request needs it (``?token=``, ``X-PitSense-Token`` or the cookie the first good page load sets).
"""

from __future__ import annotations

import hmac
import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

STATIC = Path(__file__).parent
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
         ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml"}
MAX_BODY = 8_000_000  # bytes: a question is a sentence; a recording a few seconds of opus
WRITE_TIMEOUT_S = 8.0  # a viewer that accepts no data for this long is dropped
MAX_VIEWERS = 64
COOKIE = "pitsense_token"
REPLAY_POSTS = ("/api/replay/pause", "/api/replay/resume", "/api/replay/speed", "/api/replay/seek")
PAGES = {"/": "index.html", "/index.html": "index.html", "/app.js": "app.js", "/style.css": "style.css"}


def _handler(rt, token: str | None = None, max_viewers: int = MAX_VIEWERS):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "PitSense"

        def log_message(self, *args) -> None:  # quiet
            pass

        def _authorised(self) -> bool:
            """True without a token; else the request must carry it (query, header or cookie)."""
            from urllib.parse import parse_qs

            if not token:
                return True
            got = [(parse_qs(self.path.partition("?")[2]).get("token") or [""])[0],
                   self.headers.get("X-PitSense-Token", "")]
            for part in self.headers.get("Cookie", "").split(";"):
                k, _, v = part.strip().partition("=")
                if k == COOKIE:
                    got.append(v)
            ok = [g for g in got if g and hmac.compare_digest(g.encode(), token.encode())]
            self._set_cookie = bool(ok) and not self.headers.get("Cookie", "").count(COOKIE + "=" + token)
            return bool(ok)

        def _deny(self) -> None:
            self._send(b'{"error":"token required: open the URL with ?token=..."}', "application/json", 401)

        def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
            self.send_response(code)
            if getattr(self, "_set_cookie", False):
                self.send_header("Set-Cookie", f"{COOKIE}={token}; Path=/; SameSite=Strict; HttpOnly")
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(json.dumps(obj, separators=(",", ":"), default=str).encode(), "application/json", code)

        def _radio(self) -> None:
            from urllib.parse import parse_qs

            q = parse_qs(self.path.partition("?")[2])
            car = (q.get("car") or [""])[0]
            mid = (q.get("i") or [""])[0]
            text = rt.wall_text(car, int(mid) if mid.isdigit() else None)
            if not text:
                return self._json({"error": "no radio message for this car"}, 404)
            msg = {"text": text}
            try:
                from ..voice import tts

                if not tts.available():
                    return self._json({"error": "speech not installed: pip install -e .[tts]"}, 404)
                wav = tts.speak(msg["text"])  # cached per text
            except Exception as exc:
                return self._json({"error": f"speech failed: {exc}"}, 500)
            self._send(wav.read_bytes(), "audio/wav")

        def _teamradio(self) -> None:
            from urllib.parse import parse_qs

            q = parse_qs(self.path.partition("?")[2])
            car = (q.get("car") or [""])[0]
            i = (q.get("i") or [""])[0]
            f = rt.team_radio_file(car, int(i)) if i.isdigit() and len(i) <= 5 else None
            if f is None:
                return self._json({"error": "no such radio clip"}, 404)
            self._send(f.read_bytes(), "audio/mpeg")

        def _same_origin(self) -> bool:
            """A browser page on another site must not drive the local pit wall (no CORS, and Origin must match Host)."""
            from urllib.parse import urlsplit

            origin = self.headers.get("Origin")
            return origin is None or urlsplit(origin).netloc == self.headers.get("Host", "")

        def _body(self) -> bytes | None:
            n = self.headers.get("Content-Length", "")
            if not n.isdigit() or int(n) > MAX_BODY:
                self._json({"ok": False, "error": "missing or too large body"}, 413)
                return None
            return self.rfile.read(int(n))

        def _replay(self, path: str, body: bytes) -> None:
            try:
                req = json.loads(body.decode("utf-8") or "{}")
                if not isinstance(req, dict):
                    raise ValueError
            except ValueError:
                return self._json({"ok": False, "error": "body must be a JSON object"}, 400)
            if path.endswith("/pause"):
                out = rt.replay_pause()
            elif path.endswith("/resume"):
                out = rt.replay_resume()
            elif path.endswith("/speed"):
                out = rt.replay_speed(req.get("speed"))
            else:
                out = rt.replay_seek(lap=req.get("lap"), mark=req.get("mark"))
            err = out.get("error", "")
            self._json(out, 200 if out.get("ok") else 409 if "disabled" in err else 422)

        def do_POST(self) -> None:  # noqa: N802
            from urllib.parse import parse_qs

            path, _, query = self.path.partition("?")
            try:
                if not self._authorised():
                    return self._deny()
                if path not in ("/api/ask", "/api/ask_audio") and path not in REPLAY_POSTS:
                    return self._json({"error": "not found"}, 404)
                if not self._same_origin():
                    return self._json({"ok": False, "error": "cross-origin request refused"}, 403)
                body = self._body()
                if body is None:
                    return
                if path in REPLAY_POSTS:
                    return self._replay(path, body)
                if path == "/api/ask":
                    try:
                        req = json.loads(body.decode("utf-8") or "{}")
                        car, text = str(req.get("car") or ""), str(req.get("text") or "")
                    except (ValueError, AttributeError):
                        return self._json({"ok": False, "error": "body must be JSON: {car, text}"}, 400)
                    out = rt.ask(car, text)
                else:
                    car = (parse_qs(query).get("car") or [""])[0]
                    out = rt.ask_audio(car, body)
                self._json(out, 200 if out.get("ok") else 422 if out.get("error") in ("no speech heard", "empty question") else 503)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
            except Exception as exc:  # the loop must never be hurt by a bad question
                self._json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 500)

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            try:
                if not self._authorised():
                    return self._deny()
                if path in PAGES:
                    f = STATIC / PAGES[path]
                    self._send(f.read_bytes(), TYPES[f.suffix])
                elif path == "/api/snapshot":
                    self._send(rt.latest_bytes, "application/json")
                elif path == "/api/calls":
                    self._json(rt.calls())
                elif path == "/api/alerts":
                    self._json(rt.alerts())
                elif path == "/api/health":
                    self._json(rt.health())
                elif path == "/api/replay/marks":
                    self._json(rt.replay_marks())
                elif path == "/api/stream":
                    self._stream()
                elif path == "/api/radio.wav":
                    self._radio()
                elif path == "/api/teamradio":
                    self._teamradio()
                else:
                    self._json({"error": "not found"}, 404)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def _stream(self) -> None:
            if len(rt.listeners) >= max_viewers:
                return self._json({"error": "too many viewers"}, 503)
            self.connection.settimeout(WRITE_TIMEOUT_S)  # a write that blocks this long means the client is gone or stuck
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            q = rt.subscribe()
            try:
                self.wfile.write(b"retry: 2000\nevent: snapshot\ndata: " + rt.latest_bytes + b"\n\n")
                self.wfile.flush()
                idle = 0
                while not rt.stop_event.is_set():
                    try:
                        msg = q.get(timeout=1.0)
                    except queue.Empty:
                        idle += 1
                        if idle >= 10:  # keep-alive comment every ~10 s
                            self.wfile.write(b": keep-alive\n\n")
                            self.wfile.flush()
                            idle = 0
                        continue
                    idle = 0
                    if msg["event"] in ("snapshot", "pos"):
                        self.wfile.write(b"event: " + msg["event"].encode() + b"\ndata: " + msg["data"] + b"\n\n")
                    else:
                        self.wfile.write(b"event: status\ndata: " + json.dumps(msg).encode() + b"\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                pass
            finally:
                rt.unsubscribe(q)

    return Handler


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def serve(rt, host: str = "127.0.0.1", port: int = 8765, token: str | None = None,
          max_viewers: int = MAX_VIEWERS) -> Server:
    """Start serving ``rt`` in a background thread. ``port=0`` picks a free port (see ``server_address``)."""
    server = Server((host, port), _handler(rt, token or None, max_viewers))
    server.request_queue_size = 128
    threading.Thread(target=server.serve_forever, name="pitsense-http", daemon=True).start()
    return server
