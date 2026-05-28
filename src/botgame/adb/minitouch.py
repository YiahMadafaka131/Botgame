"""High-frequency multitouch input via the minitouch TCP protocol.

`adb shell input` spawns a new JVM-backed process per gesture (tens of ms of
overhead) and cannot stream a curved path as a single continuous gesture.
minitouch (DeviceFarmer) is a small native binary that listens on a UNIX
abstract socket and accepts a tiny text protocol — `d`/`m`/`u`/`c`/`w` — so
the PC can stream points at hundreds of Hz with per-finger pressure.

Prereqs (one-off):
    adb push minitouch /data/local/tmp/minitouch
    adb shell chmod 755 /data/local/tmp/minitouch
    adb shell /data/local/tmp/minitouch         # leave running

The MiniTouch class then:
    1. forwards a local TCP port to `localabstract:minitouch`
    2. reads the header (max_x, max_y, max_pressure, max_contacts)
    3. rescales screen-pixel coords to touch-panel coords on the fly

Interface mirrors `TouchInput`: tap / swipe / swipe_path / key, so bots can
swap backends without changes.
"""

from __future__ import annotations

import socket
import time
from dataclasses import dataclass

from .device import AdbDevice, AdbError

Point = tuple[int, int]


@dataclass
class MinitouchInfo:
    """Header fields announced by the minitouch daemon on connect."""

    version: int
    max_contacts: int
    max_x: int
    max_y: int
    max_pressure: int
    pid: int


class MiniTouch:
    """Drop-in replacement for `TouchInput` that streams over minitouch."""

    def __init__(
        self,
        device: AdbDevice,
        host_port: int = 1111,
        screen_size: tuple[int, int] | None = None,
        default_pressure: int | None = None,
        connect_timeout: float = 5.0,
        forward: bool = True,
    ):
        self.device = device
        self.host_port = host_port
        self.connect_timeout = connect_timeout
        self._sock: socket.socket | None = None
        self.info: MinitouchInfo | None = None
        self._screen: tuple[int, int] = screen_size or device.screen_size()
        self._default_pressure = default_pressure
        if forward:
            self._setup_forward()

    def _setup_forward(self) -> None:
        # `adb forward tcp:<port> localabstract:minitouch`
        proc_cmd = ["forward", f"tcp:{self.host_port}", "localabstract:minitouch"]
        import subprocess

        proc = subprocess.run(
            self.device._base_cmd() + proc_cmd, capture_output=True, text=True
        )
        if proc.returncode != 0:
            raise AdbError(
                "adb forward to localabstract:minitouch failed: "
                f"{proc.stderr.strip()}. Did you push and start the binary?"
            )

    def connect(self) -> MinitouchInfo:
        if self._sock is not None:
            assert self.info is not None
            return self.info
        sock = socket.create_connection(
            ("127.0.0.1", self.host_port), timeout=self.connect_timeout
        )
        self._sock = sock
        header = self._read_header(sock)
        self.info = header
        if self._default_pressure is None:
            self._default_pressure = max(1, header.max_pressure // 2)
        return header

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def __enter__(self) -> "MiniTouch":
        self.connect()
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    @staticmethod
    def _read_header(sock: socket.socket) -> MinitouchInfo:
        buf = b""
        deadline = time.monotonic() + 3.0
        while buf.count(b"\n") < 3:
            if time.monotonic() > deadline:
                raise AdbError("minitouch: timed out reading banner")
            chunk = sock.recv(256)
            if not chunk:
                raise AdbError("minitouch: socket closed before banner")
            buf += chunk
        lines = buf.split(b"\n")
        try:
            v = int(lines[0].split()[1])
            caret = lines[1].split()
            max_contacts = int(caret[1])
            max_x = int(caret[2])
            max_y = int(caret[3])
            max_pressure = int(caret[4])
            pid = int(lines[2].split()[1])
        except (IndexError, ValueError) as exc:
            raise AdbError(f"minitouch: bad banner: {buf!r}") from exc
        return MinitouchInfo(v, max_contacts, max_x, max_y, max_pressure, pid)

    def _send(self, payload: str) -> None:
        if self._sock is None:
            self.connect()
        assert self._sock is not None
        self._sock.sendall(payload.encode("ascii"))

    def _rescale(self, x: int, y: int) -> tuple[int, int]:
        assert self.info is not None
        sw, sh = self._screen
        tx = int(round(x / max(1, sw - 1) * self.info.max_x))
        ty = int(round(y / max(1, sh - 1) * self.info.max_y))
        # Clamp to the touch-panel range.
        tx = max(0, min(self.info.max_x, tx))
        ty = max(0, min(self.info.max_y, ty))
        return tx, ty

    # ---- TouchInput-compatible API -------------------------------------

    def tap(self, x: int, y: int) -> None:
        self.connect()
        tx, ty = self._rescale(int(x), int(y))
        p = self._default_pressure or 50
        self._send(f"d 0 {tx} {ty} {p}\nc\nu 0\nc\n")

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 200) -> None:
        # Linearize as 16-point path so it's a real continuous gesture.
        steps = 16
        path = [
            (
                int(round(x1 + (x2 - x1) * i / (steps - 1))),
                int(round(y1 + (y2 - y1) * i / (steps - 1))),
            )
            for i in range(steps)
        ]
        self.swipe_path(path, duration_ms=duration_ms)

    def swipe_path(self, points: list[Point], duration_ms: int = 300) -> None:
        if len(points) < 2:
            raise ValueError("swipe_path needs at least 2 points")
        self.connect()
        p = self._default_pressure or 50
        wait_ms = max(1, duration_ms // max(1, len(points) - 1))

        x0, y0 = points[0]
        tx, ty = self._rescale(int(x0), int(y0))
        out = [f"d 0 {tx} {ty} {p}\nc\n"]
        for (x, y) in points[1:]:
            tx, ty = self._rescale(int(x), int(y))
            out.append(f"m 0 {tx} {ty} {p}\nw {wait_ms}\nc\n")
        out.append("u 0\nc\n")
        self._send("".join(out))

    def key(self, keycode: str) -> None:
        """minitouch doesn't speak keyevents; fall back to `adb shell input`."""
        self.device.shell("input", "keyevent", keycode)
