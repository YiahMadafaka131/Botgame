"""Touch input injection over ADB.

Baseline uses `adb shell input` (tap/swipe). It is simple and portable but
produces "clean" events: a single tap lands on the exact pixel and a swipe is
a perfect straight line. Humanization (jitter, curved paths, timing) lives in
`botgame.humanize` and is composed on top of these raw primitives by the bot.

For high-frequency, pressure-aware multitouch, replace this with a minitouch
backend later; the interface (`tap`, `swipe`, `swipe_path`) stays the same.
"""

from __future__ import annotations

from .device import AdbDevice

Point = tuple[int, int]


class TouchInput:
    def __init__(self, device: AdbDevice):
        self.device = device

    def tap(self, x: int, y: int) -> None:
        self.device.shell("input", "tap", str(int(x)), str(int(y)))

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 200) -> None:
        self.device.shell(
            "input",
            "swipe",
            str(int(x1)),
            str(int(y1)),
            str(int(x2)),
            str(int(y2)),
            str(int(duration_ms)),
        )

    def swipe_path(self, points: list[Point], duration_ms: int = 300) -> None:
        """Approximate a curved gesture by chaining straight `input swipe` segments.

        `input swipe` cannot follow a curve, so we split the path into segments
        and spread the total duration across them. This is a baseline; a
        minitouch backend would stream the points as a single continuous motion.
        """
        if len(points) < 2:
            raise ValueError("swipe_path needs at least 2 points")
        seg_count = len(points) - 1
        seg_ms = max(1, duration_ms // seg_count)
        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            self.swipe(x1, y1, x2, y2, seg_ms)

    def key(self, keycode: str) -> None:
        """Send a key event, e.g. 'KEYCODE_BACK' or '4'."""
        self.device.shell("input", "keyevent", keycode)
