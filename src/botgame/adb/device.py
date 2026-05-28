"""Thin wrapper around the `adb` CLI for a single USB-connected device."""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass


class AdbError(RuntimeError):
    pass


def _adb_binary() -> str:
    adb = shutil.which("adb")
    if adb is None:
        raise AdbError(
            "`adb` not found on PATH. Install Android platform-tools and make "
            "sure `adb` is callable from your shell."
        )
    return adb


def list_devices() -> list[str]:
    """Return serials of devices in the `device` state (USB-debugging authorized)."""
    out = subprocess.run(
        [_adb_binary(), "devices"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    serials = []
    for line in out.splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        serial, _, state = line.partition("\t")
        if state.strip() == "device":
            serials.append(serial.strip())
    return serials


@dataclass
class AdbDevice:
    """A single ADB target. Leave `serial=None` to auto-pick the only device."""

    serial: str | None = None
    timeout: float = 20.0

    @classmethod
    def autoconnect(cls, timeout: float = 20.0) -> "AdbDevice":
        devices = list_devices()
        if not devices:
            raise AdbError(
                "No authorized devices. Connect by USB, enable USB debugging, and "
                "accept the RSA prompt on the phone (`adb devices` should list it)."
            )
        if len(devices) > 1:
            raise AdbError(
                f"Multiple devices found ({devices}); pass an explicit serial."
            )
        return cls(serial=devices[0], timeout=timeout)

    def _base_cmd(self) -> list[str]:
        cmd = [_adb_binary()]
        if self.serial:
            cmd += ["-s", self.serial]
        return cmd

    def shell(self, *args: str) -> str:
        """Run `adb shell <args>` and return stdout as text."""
        proc = subprocess.run(
            self._base_cmd() + ["shell", *args],
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        if proc.returncode != 0:
            raise AdbError(f"adb shell {' '.join(args)} failed: {proc.stderr.strip()}")
        return proc.stdout

    def exec_out(self, *args: str) -> bytes:
        """Run `adb exec-out <args>` and return raw stdout bytes (binary-safe)."""
        proc = subprocess.run(
            self._base_cmd() + ["exec-out", *args],
            capture_output=True,
            timeout=self.timeout,
        )
        if proc.returncode != 0:
            raise AdbError(
                f"adb exec-out {' '.join(args)} failed: "
                f"{proc.stderr.decode(errors='replace').strip()}"
            )
        return proc.stdout

    def screen_size(self) -> tuple[int, int]:
        """Return (width, height) in pixels from `wm size`."""
        out = self.shell("wm", "size")
        # Prefer the override size if present, else the physical size.
        match = re.search(r"Override size:\s*(\d+)x(\d+)", out) or re.search(
            r"Physical size:\s*(\d+)x(\d+)", out
        )
        if not match:
            raise AdbError(f"Could not parse screen size from: {out!r}")
        return int(match.group(1)), int(match.group(2))
