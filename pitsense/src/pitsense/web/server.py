"""HTTP server for the pit wall: static page, JSON API and a server-sent-events stream.

    GET /                  dashboard
    GET /api/snapshot      latest snapshot (the full Snapshot dict plus an "extra" block)
    GET /api/calls         {"current": [...], "log": [...]}
    GET /api/alerts        {"active": [...], "log": [...]}
    GET /api/health        loop status, throughput, snapshot latency, model bundle
    GET /api/stream        server-sent events: "snapshot" (the snapshot JSON), "pos" (car positions, ~3 Hz), "status"
    GET /api/radio.wav     ?car=N[&i=ID]: our pit wall's message (latest, or by id), spoken (Piper TTS; 404 if not installed)
    GET /api/teamradio     ?car=N&i=K: the K-th real driver radio mp3 of car N (published clips only, audio/mpeg)
    POST /api/ask          {"car": "16", "text": "what if we box now?"}: answer a what-if or fact question; the Q&A joins the
                           conversation and the answer can be spoken via /api/radio.wav?car=N&i=<reply id>
    POST /api/ask_audio    ?car=N, body = recorded audio (webm/opus, ogg, wav): transcribed on the server, then as /api/ask

The GET endpoints are read-only; the two POSTs only add a question and its answer to the conversation. Binds to
127.0.0.1 unless told otherwise; a POST from a page on another origin is refused.
"""

from __future__ import annotations

import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

STATIC = Path(__file__).parent
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
         ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml"}
MAX_BODY = 8_000_000  # bytes: a question is a sentence; a recording a few seconds of opus
PAGES = {"/": "index.html", "/index.html": "index.html", "/app.js": "app.js", "/style.css": "style.css"}


def _handler(rt):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "PitSense"

        def log_message(self, *args) -> None:  # quiet
            pass

        def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
            self.send_response(code)
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

        def do_POST(self) -> None:  # noqa: N802
            from urllib.parse import parse_qs

            path, _, query = self.path.partition("?")
            try:
                if path not in ("/api/ask", "/api/ask_audio"):
                    return self._json({"error": "not found"}, 404)
                if not self._same_origin():
                    return self._json({"ok": False, "error": "cross-origin request refused"}, 403)
                body = self._body()
                if body is None:
                    return
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


def serve(rt, host: str = "127.0.0.1", port: int = 8765) -> Server:
    """Start serving ``rt`` in a background thread. ``port=0`` picks a free port (see ``server_address``)."""
    server = Server((host, port), _handler(rt))
    threading.Thread(target=server.serve_forever, name="pitsense-http", daemon=True).start()
    return server
