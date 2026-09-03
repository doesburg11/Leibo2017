"""Rollout visualization: render a played-out episode's global RGB frames
(from `GatheringEnv.render()` / `WolfpackEnv.render()`) to an animated GIF.

This is the graphics-utility counterpart to the sibling
SequentialSocialDilemmas repo's episode-rollout video tooling
(`visualization/visualizer_rllib.py` + `utility_funcs.make_video_from_rgb_imgs`,
which shell out to OpenCV for an .mp4). Built on Pillow instead of OpenCV/
ffmpeg since Pillow is already a transitive dependency of matplotlib (see
requirements.txt) and a GIF needs no video codec.
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image


def save_rollout_gif(frames: list[np.ndarray], out_path: str, fps: int = 10, scale: int = 8) -> None:
    """Write `frames` (a list of (H, W, 3) uint8 RGB arrays, e.g. collected
    by calling `env.render()` once per step of an episode) as a looping GIF
    at `out_path`. Each frame is nearest-neighbor upscaled by `scale` first
    since a raw grid frame is one pixel per cell.

    Written to a same-directory temp file and moved into place with
    `os.replace` so a concurrent reader (or a second run writing the same
    path) never observes a partially-written file.

    Note: Pillow's GIF writer merges consecutive pixel-identical frames
    into one with a longer display duration (e.g. an all-STAND_STILL
    rollout) -- expected GIF-format behavior, not a bug here; the output
    file's frame count can be less than `len(frames)`.
    """
    if not frames:
        raise ValueError("save_rollout_gif: frames is empty")
    if fps <= 0:
        raise ValueError(f"save_rollout_gif: fps must be positive, got {fps}")
    if scale <= 0:
        raise ValueError(f"save_rollout_gif: scale must be positive, got {scale}")
    images = [
        Image.fromarray(frame).resize((frame.shape[1] * scale, frame.shape[0] * scale), Image.NEAREST)
        for frame in frames
    ]
    tmp_path = f"{out_path}.tmp"
    images[0].save(
        tmp_path,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / fps),
        loop=0,
    )
    os.replace(tmp_path, out_path)
