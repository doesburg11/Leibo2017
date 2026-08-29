"""Shared gridworld primitives used by both Gathering and Wolfpack.

Both games in Leibo et al. (2017) share the same low-level engine: agents
occupy cells of a 2D grid, face one of 4 cardinal directions, move with
agent-centered actions, and observe an oriented crop of the map rendered as
an RGB image (Sec. 4: "Observations O(s, i) in R^{3x16x21} ... depended on
the player's current position and orientation. The observation window
extended 15 grid squares ahead and 10 grid squares from side to side.").

Orientation convention: 0=North, 1=East, 2=South, 3=West, clockwise.
"""

from __future__ import annotations

import numpy as np

NORTH, EAST, SOUTH, WEST = 0, 1, 2, 3
DIRS = np.array([(-1, 0), (0, 1), (1, 0), (0, -1)])  # (drow, dcol) per orientation

# Shared 8-action set (Sec. 4): forward, backward, strafe-left, strafe-right,
# rotate-left, rotate-right, beam, stand-still.
STEP_FORWARD, STEP_BACKWARD, STEP_LEFT, STEP_RIGHT = 0, 1, 2, 3
ROTATE_LEFT, ROTATE_RIGHT, USE_BEAM, STAND_STILL = 4, 5, 6, 7
NUM_ACTIONS = 8

OBS_FORWARD = 15  # cells ahead of the agent
OBS_SIDE = 10  # cells to each side of the agent
OBS_HEIGHT = OBS_FORWARD + 1  # + the agent's own row
OBS_WIDTH = 2 * OBS_SIDE + 1  # + the agent's own column
OBS_SHAPE = (3, OBS_HEIGHT, OBS_WIDTH)  # (C, H, W), matches paper's R^{3x16x21}


def move_delta(orientation: int, action: int) -> tuple[int, int]:
    """Row/col delta for a movement action, relative to facing `orientation`."""
    if action == STEP_FORWARD:
        d = DIRS[orientation]
    elif action == STEP_BACKWARD:
        d = -DIRS[orientation]
    elif action == STEP_LEFT:
        d = DIRS[(orientation - 1) % 4]
    elif action == STEP_RIGHT:
        d = DIRS[(orientation + 1) % 4]
    else:
        return 0, 0
    return int(d[0]), int(d[1])


def rotate_orientation(orientation: int, action: int) -> int:
    if action == ROTATE_LEFT:
        return (orientation - 1) % 4
    if action == ROTATE_RIGHT:
        return (orientation + 1) % 4
    return orientation


def beam_cells(row: int, col: int, orientation: int, grid_shape: tuple[int, int], max_range: int) -> list[tuple[int, int]]:
    """Cells struck by a straight beam fired forward from (row, col)."""
    dr, dc = DIRS[orientation]
    h, w = grid_shape
    cells = []
    r, c = row, col
    for _ in range(max_range):
        r, c = r + dr, c + dc
        if not (0 <= r < h and 0 <= c < w):
            break
        cells.append((r, c))
    return cells


_SQ_HALF = max(OBS_FORWARD, OBS_SIDE)  # 15: half-size of the pre-rotation square crop


def local_observation(rgb: np.ndarray, row: int, col: int, orientation: int) -> np.ndarray:
    """Crop + orient an agent-centered observation window out of a full RGB grid.

    `rgb` has shape (H, W, 3). Returns shape OBS_SHAPE = (3, 16, 21): the
    agent always "faces up" in its own observation, matching the paper's
    description of an agent-centered, orientation-dependent view.

    Implementation note: rather than rotating the whole grid and tracking
    where the agent's point lands (easy to get the handedness of the point
    formula backwards, and it silently produces a *rotated* observation
    that's just internally self-consistent in the wrong direction), we
    first cut a (2*_SQ_HALF+1)-square patch centered exactly on the agent.
    A square array's center index is a fixed point of any 90-degree
    rotation, so after rotating that square patch the agent is still
    dead-center, with no coordinate bookkeeping required; only the final
    asymmetric (16, 21) window then needs to be sliced out.
    """
    h, w, _ = rgb.shape
    pad = _SQ_HALF
    padded = np.zeros((h + 2 * pad, w + 2 * pad, 3), dtype=rgb.dtype)
    padded[pad:pad + h, pad:pad + w] = rgb
    pr, pc = row + pad, col + pad

    square = padded[pr - _SQ_HALF:pr + _SQ_HALF + 1, pc - _SQ_HALF:pc + _SQ_HALF + 1]
    rotated = np.rot90(square, k=orientation, axes=(0, 1))  # sign verified against beam/move directions

    center = _SQ_HALF
    top, bottom = 0, center + 1  # OBS_FORWARD cells "ahead" + the agent's own row
    left, right = center - OBS_SIDE, center + OBS_SIDE + 1
    crop = rotated[top:bottom, left:right]
    return np.transpose(crop, (2, 0, 1)).copy()  # (3, H, W)
