"""Extract timestamped frames from a gameplay video.

OpenCV is imported lazily so the rest of the dataset package (parsers,
alignment) stays usable and testable without the heavier dependency.
"""

from __future__ import annotations

from collections.abc import Iterator


def iter_frames(path: str, target_fps: float | None = None) -> Iterator[tuple[float, "object"]]:
    """Yield (timestamp_seconds, frame_rgb) pairs from the video at `path`.

    If `target_fps` is set, frames are subsampled to roughly that rate. Frames
    are RGB (H, W, 3) uint8 arrays. Timestamps are relative to the video start.
    """
    try:
        import cv2  # noqa: PLC0415  (lazy: only needed for actual video decoding)
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "opencv-python is required for video decoding: pip install opencv-python"
        ) from exc

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    step = 1
    if target_fps and src_fps > 0:
        step = max(1, round(src_fps / target_fps))

    idx = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if idx % step == 0:
                t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                yield t, frame_bgr[:, :, ::-1].copy()  # BGR -> RGB
            idx += 1
    finally:
        cap.release()
