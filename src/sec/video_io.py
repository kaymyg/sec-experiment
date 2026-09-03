"""Frame extraction helpers.

Uses OpenCV's VideoCapture directly (no shelling out) so this works anywhere
OpenCV is installed. Frames are returned as RGB uint8 arrays.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


def iter_frames(video_path: str | Path, max_frames: int | None = None) -> Iterator[np.ndarray]:
    """Yield frames from a video file as RGB uint8 HxWx3 arrays."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    try:
        count = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            yield cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            count += 1
            if max_frames is not None and count >= max_frames:
                break
    finally:
        cap.release()


def video_fps(video_path: str | Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()
    if not fps or fps <= 0:
        raise ValueError(f"Could not determine FPS for {video_path}")
    return float(fps)


def load_image(image_path: str | Path) -> np.ndarray:
    """Load a still image as RGB uint8 HxWx3, matching iter_frames' format."""
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not open image: {image_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_to_match(image: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    h, w = target_hw
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)
