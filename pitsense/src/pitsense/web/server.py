"""HTTP server for the pit wall: static page, JSON API and a server-sent-events stream.

    GET /                  dashboard
    GET /api/snapshot      latest snapshot (the full Snapshot dict plus an "extra" block)
    GET /api/calls         {"current": [...], "log": [...]}
    GET /api/alerts        {"active": [...], "log": [...]}
    GET /api/health        loop status, throughput, snapshot latency, model bundle
    GET /api/stream        server-sent events: "snapshot" (the snapshot JSON), "status"
    GET /api/radio.wav     ?car=N: that car's latest radio message, spoken (Piper TTS; 404 if not installed)

Read-only: no endpoint changes anything. Binds to 127.0.0.1 unless told otherwise.
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

            car = (parse_qs(self.path.partition("?")[2]).get("car") or [""])[0]
            msg = getattr(rt, "radio", {}).get(car)
            if not msg:
                return self._json({"error": "no radio message for this car"}, 404)
            try:
                from ..voice import tts

                if not tts.available():
                    return self._json({"error": "speech not installed: pip install -e .[tts]"}, 404)
                wav = tts.speak(msg["text"])  # cached per text
            except Exception as exc:
                return self._json({"error": f"speech failed: {exc}"}, 500)
            self._send(wav.read_bytes(), "audio/wav")

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
                    if msg["event"] == "snapshot":
                        self.wfile.write(b"event: snapshot\ndata: " + msg["data"] + b"\n\n")
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
