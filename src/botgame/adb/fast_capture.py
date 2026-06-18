"""High-fps screen capture by streaming H.264 from `screenrecord`.

`adb exec-out screenrecord --output-format=h264 -` streams encoded video to
stdout. A background thread reads the byte stream, decodes packets with PyAV,
and keeps only the most recent decoded frame; `grab()` returns it. This avoids
the per-frame `screencap` round-trip and lifts capture from ~1-5 fps to ~30 fps
on most devices.

`screenrecord` caps each session at 180 seconds. `restart_secs` (default 170)
cycles the subprocess before the device kills it, so the stream keeps running
for arbitrarily long sessions.

PyAV must be installed: `pip install av`.
"""

from __future__ import annotations

import subprocess
import threading
import time
from typing import Optional

import numpy as np

from .device import AdbDevice, AdbError


class FastCapture:
    """Streaming RGB capture backed by a background H.264 decoder thread."""

    def __init__(
        self,
        device: AdbDevice,
        size: tuple[int, int] | None = None,
        bit_rate: int = 8_000_000,
        restart_secs: float = 170.0,
        chunk_size: int = 4096,
    ):
        self.device = device
        self.size = size
        self.bit_rate = bit_rate
        self.restart_secs = restart_secs
        self.chunk_size = chunk_size

        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._stop = threading.Event()
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def start(self) -> None:
        try:
            import av  # noqa: F401
        except ImportError as exc:
            raise AdbError(
                "PyAV is required for FastCapture. Install with `pip install av`."
            ) from exc
        if self._thread is not None:
            return
        self._stop.clear()
        self._error = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
        self._thread = None
        self._proc = None

    def __enter__(self) -> "FastCapture":
        self.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.stop()

    def grab(self, timeout: float = 5.0) -> np.ndarray:
        """Return the most recent decoded frame as (H, W, 3) uint8 RGB.

        Blocks up to `timeout` seconds for the first frame to arrive.
        """
        deadline = time.monotonic() + timeout
        while True:
            if self._error is not None:
                raise self._error
            with self._lock:
                if self._latest is not None:
                    return self._latest.copy()
            if time.monotonic() > deadline:
                raise AdbError("FastCapture: no frame received within timeout")
            time.sleep(0.02)

    def _spawn(self) -> subprocess.Popen:
        args: list[str] = [
            "screenrecord",
            "--output-format=h264",
            f"--bit-rate={self.bit_rate}",
        ]
        if self.size is not None:
            args.append(f"--size={self.size[0]}x{self.size[1]}")
        args.append("-")
        return self.device.popen_exec_out(*args)

    def _run(self) -> None:
        import av

        try:
            while not self._stop.is_set():
                self._proc = self._spawn()
                started = time.monotonic()
                codec = av.CodecContext.create("h264", "r")
                try:
                    self._decode_loop(codec, started)
                finally:
                    self._kill_proc(self._proc)
                    self._proc = None
        except BaseException as exc:  # noqa: BLE001
            self._error = exc

    def _decode_loop(self, codec, started: float) -> None:
        import av

        assert self._proc is not None
        stdout = self._proc.stdout
        assert stdout is not None
        while not self._stop.is_set():
            chunk = stdout.read(self.chunk_size)
            if not chunk:
                return
            try:
                packets = codec.parse(chunk)
            except av.AVError:
                continue
            for packet in packets:
                try:
                    frames = codec.decode(packet)
                except av.AVError:
                    continue
                for frame in frames:
                    arr = frame.to_ndarray(format="rgb24")
                    with self._lock:
                        self._latest = arr
            if time.monotonic() - started > self.restart_secs:
                return

    @staticmethod
    def _kill_proc(proc: subprocess.Popen | None) -> None:
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            proc.kill()
