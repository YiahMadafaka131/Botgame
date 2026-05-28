"""Append-only JSONL action log.

Every action the bot takes is recorded with a timestamp and the humanization
level in effect, so a detector's verdicts can be cross-referenced against
ground-truth bot activity (what was done, when, and how human-like it was).
"""

from __future__ import annotations

import json
import os
import time
from typing import Any


class TelemetryLogger:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self._fh = open(path, "a", encoding="utf-8")

    def log(self, action: str, **fields: Any) -> None:
        record = {"ts": time.time(), "action": action, **fields}
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "TelemetryLogger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
