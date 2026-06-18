import threading
import time

import numpy as np
import pytest

from botgame.adb import FastCapture
from botgame.adb.device import AdbDevice, AdbError


class _DummyDevice(AdbDevice):
    """Stub that never touches a real `adb` binary."""

    def __init__(self):  # noqa: D401 - simple override
        super().__init__(serial="dummy")

    def popen_exec_out(self, *args):  # type: ignore[override]
        raise AssertionError("subprocess must be mocked in tests")


def test_grab_returns_latest_frame():
    fc = FastCapture(_DummyDevice())
    frame = np.full((4, 4, 3), 200, dtype=np.uint8)
    with fc._lock:
        fc._latest = frame

    out = fc.grab(timeout=0.5)
    assert out.shape == (4, 4, 3)
    assert (out == 200).all()
    # `grab` must return a copy so callers cannot mutate the slot.
    out[0, 0, 0] = 0
    assert fc._latest[0, 0, 0] == 200


def test_grab_times_out_without_frame():
    fc = FastCapture(_DummyDevice())
    with pytest.raises(AdbError, match="no frame received"):
        fc.grab(timeout=0.1)


def test_grab_raises_propagated_decoder_error():
    fc = FastCapture(_DummyDevice())
    fc._error = AdbError("boom")
    with pytest.raises(AdbError, match="boom"):
        fc.grab(timeout=0.1)


def test_grab_unblocks_when_frame_arrives_late():
    fc = FastCapture(_DummyDevice())
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    def _publish():
        time.sleep(0.1)
        with fc._lock:
            fc._latest = frame

    t = threading.Thread(target=_publish)
    t.start()
    try:
        out = fc.grab(timeout=2.0)
    finally:
        t.join()
    assert out.shape == (2, 2, 3)


def test_start_raises_clean_error_when_pyav_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "av":
            raise ImportError("no av")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    fc = FastCapture(_DummyDevice())
    with pytest.raises(AdbError, match="PyAV is required"):
        fc.start()


def test_stop_is_idempotent_without_start():
    fc = FastCapture(_DummyDevice())
    fc.stop()  # must not raise
    assert fc._thread is None
    assert fc._proc is None


def test_context_manager_starts_and_stops(monkeypatch):
    """The context manager wires start/stop without needing PyAV at decode time."""
    import sys
    import types

    monkeypatch.setitem(sys.modules, "av", types.ModuleType("av"))
    monkeypatch.setattr(FastCapture, "_run", lambda self: self._stop.wait())

    fc = FastCapture(_DummyDevice())
    with fc as cap:
        assert cap is fc
        assert fc._thread is not None
        assert fc._thread.is_alive()
    assert fc._thread is None
