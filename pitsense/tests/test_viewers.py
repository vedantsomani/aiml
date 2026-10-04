"""Many browsers at once: concurrent SSE clients, a stalled one, the access token, the phone layout files."""

import json
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from pitsense.pitwall.runtime import PitWallRuntime, ReplaySource
from pitsense.web import server as srv
from pitsense.web.server import serve

from .conftest import make_log

WEB = Path(srv.__file__).parent


def make_rt():
    rt = PitWallRuntime(ReplaySource(make_log(), 0), models=False)
    rt.start()
    assert rt.wait(30)
    return rt


def open_sse(port, path="/api/stream", extra=""):
    s = socket.create_connection(("127.0.0.1", port), timeout=10)
    s.sendall(f"GET {path} HTTP/1.1\r\nHost: x\r\n{extra}\r\n".encode())
    return s


def wait_for(cond, secs=5.0):
    end = time.time() + secs
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_many_sse_clients_and_a_stalled_one_do_not_stall_publish():
    rt = make_rt()
    server = serve(rt, port=0)
    port = server.server_address[1]
    socks = []
    try:
        base = []
        for _ in range(20):
            t0 = time.perf_counter()
            rt.publish()
            base.append(time.perf_counter() - t0)
        for _ in range(30):
            socks.append(open_sse(port))
        slow = open_sse(port)  # never reads
        slow.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)
        socks.append(slow)
        assert wait_for(lambda: json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health").read())["viewers"] == 31)
        lat = []
        for _ in range(300):  # far more than the 8-message queues hold, nobody drains the slow one
            t0 = time.perf_counter()
            rt.publish()
            lat.append(time.perf_counter() - t0)
        assert max(lat) < 0.5 and sorted(lat)[len(lat) // 2] < 5 * max(sorted(base)[len(base) // 2], 0.002)
        assert all(q.qsize() <= 8 for q in rt.listeners)
        # a healthy client still gets data
        socks[0].settimeout(5)
        assert b"event: snapshot" in socks[0].recv(1 << 20)
    finally:
        for s in socks:
            s.close()
        server.shutdown()
        rt.stop()
    assert wait_for(lambda: rt.health()["viewers"] == 0, 12)


def test_token_enforced_on_get_post_and_stream():
    rt = make_rt()
    server = serve(rt, port=0, token="s3cret")
    port = server.server_address[1]
    base = f"http://127.0.0.1:{port}"

    def code(req):
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, r
        except urllib.error.HTTPError as e:
            return e.code, e

    try:
        for p in ("/", "/app.js", "/api/snapshot", "/api/health", "/api/stream"):
            assert code(base + p)[0] == 401
            assert code(base + p + "?token=wrong")[0] == 401
        assert code(base + "/api/health?token=s3cret")[0] == 200
        st, r = code(base + "/?token=s3cret")
        cookie = r.headers["Set-Cookie"].split(";")[0]
        assert st == 200 and cookie.startswith("pitsense_token=")
        assert code(urllib.request.Request(base + "/api/snapshot", headers={"Cookie": cookie}))[0] == 200
        assert code(urllib.request.Request(base + "/api/snapshot", headers={"X-PitSense-Token": "s3cret"}))[0] == 200
        post = lambda q, **h: urllib.request.Request(  # noqa: E731
            base + "/api/ask" + q, data=b'{"car":"22","text":"gap ahead?"}', headers={"Content-Type": "application/json", **h})
        assert code(post(""))[0] == 401
        assert code(post("?token=nope"))[0] == 401
        assert code(post("?token=s3cret"))[0] in (200, 422, 503)
        s = open_sse(port)
        assert b" 401 " in s.recv(200)
        s.close()
    finally:
        server.shutdown()
        rt.stop()


def test_no_token_means_open_and_viewer_cap():
    rt = make_rt()
    server = serve(rt, port=0, max_viewers=2)
    port = server.server_address[1]
    socks = [open_sse(port), open_sse(port)]
    try:
        assert wait_for(lambda: len(rt.listeners) == 2)
        s = open_sse(port)
        s.settimeout(5)
        assert b" 503 " in s.recv(200)
        s.close()
    finally:
        for s in socks:
            s.close()
        server.shutdown()
        rt.stop()


def test_snapshot_lists_teams_for_the_per_viewer_selector():
    rt = make_rt()
    try:
        teams = rt.latest["extra"]["teams"]
        assert isinstance(teams, dict) and all(isinstance(v, list) for v in teams.values())
    finally:
        rt.stop()


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_app_js_parses():
    subprocess.run(["node", "--check", str(WEB / "app.js")], check=True)


def test_css_has_breakpoints_and_touch_rules():
    css = (WEB / "style.css").read_text(encoding="utf-8")
    assert "max-width:699px" in css  # phone
    assert "min-width:700px" in css and "max-width:1100px" in css  # tablet
    assert "pointer:coarse" in css and ".tabs" in css and ".callbar" in css
    html = (WEB / "index.html").read_text(encoding="utf-8")
    for t in ("tower", "map", "radio", "team", "alerts"):
        assert f'data-tab="{t}"' in html
    assert "width=device-width" in html and 'id="teamsel"' in html and 'id="callbar"' in html
