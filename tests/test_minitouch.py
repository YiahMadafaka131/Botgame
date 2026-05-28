"""Unit tests for the minitouch backend.

A real device is unavailable in CI, so we drive `MiniTouch` against an
in-process TCP server that speaks the minitouch banner and records bytes the
client sends. That gives end-to-end protocol coverage without ADB.
"""

import socket
import threading

import pytest

from botgame.adb import MiniTouch
from botgame.adb.device import AdbDevice


class _StubDevice(AdbDevice):
    def __init__(self, screen=(1080, 1920)):
        super().__init__(serial="stub")
        self._screen = screen

    def screen_size(self):  # type: ignore[override]
        return self._screen

    def shell(self, *args):  # type: ignore[override]
        # Used by `key()`; record the call so tests can assert it.
        self.shell_calls.append(args)
        return ""

    shell_calls: list = []


class _FakeServer:
    """Listens on 127.0.0.1, sends a banner, records bytes received."""

    BANNER = b"v 1\n^ 10 32767 32767 255\n$ 12345\n"

    def __init__(self):
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(1)
        self.port = self.sock.getsockname()[1]
        self.received = b""
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        try:
            self.sock.settimeout(3.0)
            conn, _ = self.sock.accept()
            with conn:
                conn.sendall(self.BANNER)
                conn.settimeout(2.0)
                while not self._stop.is_set():
                    try:
                        chunk = conn.recv(4096)
                    except socket.timeout:
                        continue
                    if not chunk:
                        break
                    self.received += chunk
        except OSError:
            pass

    def drain(self, timeout: float = 2.0) -> None:
        """Block until the server thread observes EOF and exits its read loop."""
        self._thread.join(timeout=timeout)

    def close(self):
        self._stop.set()
        try:
            self.sock.close()
        except OSError:
            pass
        self._thread.join(timeout=2.0)


@pytest.fixture()
def server():
    s = _FakeServer()
    yield s
    s.close()


def _connect(device, server) -> MiniTouch:
    return MiniTouch(device, host_port=server.port, forward=False)


def test_banner_is_parsed(server):
    device = _StubDevice()
    mt = _connect(device, server)
    info = mt.connect()
    assert info.version == 1
    assert info.max_contacts == 10
    assert info.max_x == 32767
    assert info.max_y == 32767
    assert info.max_pressure == 255
    assert info.pid == 12345
    mt.close()


def test_tap_sends_down_commit_up_commit(server):
    device = _StubDevice(screen=(1080, 1920))
    with _connect(device, server) as mt:
        mt.tap(540, 960)  # center of screen
    server.drain()
    # Center of a (1080, 1920) screen → roughly half of (max_x, max_y).
    text = server.received.decode("ascii")
    assert text.startswith("d 0 ")
    assert "c\n" in text
    assert "u 0" in text
    # Coordinates land near the touch-panel midpoint (allow rounding slack).
    first = text.splitlines()[0].split()
    tx, ty = int(first[2]), int(first[3])
    assert 16000 <= tx <= 16800
    assert 16000 <= ty <= 16800


def test_swipe_path_streams_continuous_motion(server):
    device = _StubDevice(screen=(1000, 1000))
    points = [(0, 0), (500, 500), (1000, 1000)]
    with _connect(device, server) as mt:
        mt.swipe_path(points, duration_ms=60)
    server.drain()
    text = server.received.decode("ascii")
    lines = [ln for ln in text.splitlines() if ln]
    # 1 down + 2 moves (with wait+commit pairs) + 1 up + 1 final commit.
    downs = [ln for ln in lines if ln.startswith("d 0 ")]
    moves = [ln for ln in lines if ln.startswith("m 0 ")]
    ups = [ln for ln in lines if ln.startswith("u 0")]
    assert len(downs) == 1
    assert len(moves) == 2
    assert len(ups) == 1
    assert any(ln.startswith("w ") for ln in lines)


def test_coords_are_clamped_to_touch_panel_range(server):
    device = _StubDevice(screen=(1000, 1000))
    with _connect(device, server) as mt:
        mt.tap(99999, -50)  # absurd OOB coords
    server.drain()
    text = server.received.decode("ascii")
    first = text.splitlines()[0].split()
    tx, ty = int(first[2]), int(first[3])
    assert 0 <= tx <= 32767
    assert 0 <= ty <= 32767


def test_key_falls_back_to_adb_shell(server):
    device = _StubDevice()
    device.shell_calls = []
    with _connect(device, server) as mt:
        mt.key("KEYCODE_BACK")
    assert ("input", "keyevent", "KEYCODE_BACK") in device.shell_calls


def test_swipe_path_rejects_short_input(server):
    device = _StubDevice()
    with _connect(device, server) as mt:
        with pytest.raises(ValueError):
            mt.swipe_path([(0, 0)])
