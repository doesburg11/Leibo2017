"""Wolfpack, the pack-hunting SSD from Leibo et al. (2017), Sec. 5.2.

Two learning wolves chase a (scripted, non-learning) prey. When either wolf
touches the prey, every wolf within `capture_radius` of the prey at that
moment shares a reward of `r_team`; a wolf that captures alone, with its
partner outside the radius, gets `r_lone` instead (partner gets 0). The
prey then respawns and the episode continues.

The paper never specifies the prey's own behavior (only that "two players
(wolves) chase a third player (the prey)" -- with no third learning agent
described), the map layout/size, or the exact shape of the "diamond" capture
region beyond Fig. 3's illustration. See the top-level README's "Blind
spots" section for the choices made here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from leibo2017.envs.grid_utils import (
    NUM_ACTIONS,
    OBS_SHAPE,
    ROTATE_LEFT,
    ROTATE_RIGHT,
    STAND_STILL,
    USE_BEAM,
    local_observation,
    move_delta,
    rotate_orientation,
)

EMPTY, WALL = 0, 1
COLOR_WALL = (128, 128, 128)
COLOR_SELF = (0, 0, 255)
COLOR_TEAMMATE = (100, 180, 255)  # "light-blue in its teammate's view" (Sec. 4)
COLOR_PREY = (255, 0, 0)


@dataclass
class WolfpackConfig:
    capture_radius: float = 2.0
    r_team: float = 5.0
    r_lone: float = 1.0
    episode_length: int = 1000
    height: int = 21
    width: int = 21
    prey_flee_prob: float = 0.5  # own choice: prey sometimes flees the nearer wolf; see README
    prey_detection_range: int = 3  # own choice: prey only flees a wolf this close; otherwise wanders.
    # Tuned (not paper-specified) so that occasional captures are reachable
    # by chance under epsilon-greedy random exploration -- a much larger
    # detection range or flee probability lets a deterministic evader
    # retreat to a permanently safe corner of a bounded arena against
    # non-pursuing random wolves, which would leave DQN training with no
    # reward signal to ever bootstrap from.


class WolfpackEnv:
    """Two-wolf independent-learner Wolfpack game with a scripted prey."""

    num_agents = 2
    num_actions = NUM_ACTIONS
    obs_shape = OBS_SHAPE

    def __init__(self, config: WolfpackConfig | None = None, rng: np.random.Generator | None = None):
        self.cfg = config or WolfpackConfig()
        self.rng = rng or np.random.default_rng()
        self._build_static_map()
        self.reset()

    def _build_static_map(self) -> None:
        cfg = self.cfg
        grid = np.full((cfg.height, cfg.width), EMPTY, dtype=np.int8)
        grid[0, :] = WALL
        grid[-1, :] = WALL
        grid[:, 0] = WALL
        grid[:, -1] = WALL
        self._static_grid = grid

    def reset(self):
        cfg = self.cfg
        self._t = 0
        h, w = cfg.height, cfg.width
        self.wolves = [
            {"row": 2, "col": 2, "orientation": 1},
            {"row": h - 3, "col": 2, "orientation": 1},
        ]
        self.prey = {"row": h // 2, "col": w // 2}
        self.captures = 0
        self.wolves_per_capture_sum = 0
        return self._observations()

    def _render_rgb(self, viewer_idx: int) -> np.ndarray:
        grid = self._static_grid
        rgb = np.zeros((*grid.shape, 3), dtype=np.uint8)
        rgb[grid == WALL] = COLOR_WALL
        for i, wf in enumerate(self.wolves):
            rgb[wf["row"], wf["col"]] = COLOR_SELF if i == viewer_idx else COLOR_TEAMMATE
        rgb[self.prey["row"], self.prey["col"]] = COLOR_PREY
        return rgb

    def _observations(self):
        return tuple(
            local_observation(self._render_rgb(i), w["row"], w["col"], w["orientation"])
            for i, w in enumerate(self.wolves)
        )

    def _move_prey(self) -> None:
        cfg = self.cfg
        pr, pc = self.prey["row"], self.prey["col"]
        dists = [abs(pr - w["row"]) + abs(pc - w["col"]) for w in self.wolves]
        nearest_dist = min(dists)
        nearest = self.wolves[int(np.argmin(dists))]
        candidates = [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]
        # Only flee a wolf that is actually close (own choice, not specified
        # by the paper -- see README). An unconditional "always maximize
        # distance from the nearest wolf" prey can retreat into a permanent
        # safe corner of a bounded arena and then never be approached again
        # by chance, which would make the task unlearnable via epsilon-greedy
        # exploration (no reward is ever observed to bootstrap learning from).
        if nearest_dist < cfg.prey_detection_range and self.rng.random() < cfg.prey_flee_prob:
            def score(d):
                nr, nc = pr + d[0], pc + d[1]
                if self._static_grid[nr, nc] == WALL:
                    return -1e9
                return abs(nr - nearest["row"]) + abs(nc - nearest["col"])

            dr, dc = max(candidates, key=score)
        else:
            self.rng.shuffle(candidates)
            dr, dc = next(d for d in candidates if self._static_grid[pr + d[0], pc + d[1]] != WALL)
        self.prey["row"], self.prey["col"] = pr + dr, pc + dc

    def step(self, actions):
        cfg = self.cfg
        rewards = [0.0, 0.0]
        order = self.rng.permutation(self.num_agents)
        occupied = {(w["row"], w["col"]) for w in self.wolves}

        for i in order:
            w = self.wolves[i]
            a = int(actions[i])
            if a in (ROTATE_LEFT, ROTATE_RIGHT):
                w["orientation"] = rotate_orientation(w["orientation"], a)
            elif a in (USE_BEAM, STAND_STILL):
                pass  # Wolfpack has no beam mechanic; kept as an inert no-op for a shared action space.
            else:
                dr, dc = move_delta(w["orientation"], a)
                nr, nc = w["row"] + dr, w["col"] + dc
                if self._static_grid[nr, nc] != WALL and (nr, nc) not in occupied:
                    occupied.discard((w["row"], w["col"]))
                    w["row"], w["col"] = nr, nc
                    occupied.add((nr, nc))

        # Move the prey before checking for a capture: a capture is any
        # overlap between a wolf and the prey once both have moved this
        # step, regardless of which one stepped onto the other. Checking
        # only "did a wolf just step onto the prey" (i.e. before the prey's
        # own move) misses the case where the prey's random wander/flee
        # step lands it on a wolf that stayed still or moved elsewhere.
        self._move_prey()

        capturer = None
        for i, w in enumerate(self.wolves):
            if (w["row"], w["col"]) == (self.prey["row"], self.prey["col"]):
                capturer = i
                break

        if capturer is not None:
            pr, pc = self.prey["row"], self.prey["col"]
            in_radius = [
                (abs(w["row"] - pr) + abs(w["col"] - pc)) <= cfg.capture_radius for w in self.wolves
            ]
            n_in_radius = sum(in_radius)
            self.captures += 1
            self.wolves_per_capture_sum += n_in_radius
            if n_in_radius >= 2:
                for i in range(self.num_agents):
                    if in_radius[i]:
                        rewards[i] += cfg.r_team
            else:
                rewards[capturer] += cfg.r_lone
            self._respawn_prey()

        self._t += 1
        done = self._t >= cfg.episode_length
        avg_wolves_per_capture = (self.wolves_per_capture_sum / self.captures) if self.captures else 0.0
        info = {"captures": self.captures, "avg_wolves_per_capture": avg_wolves_per_capture}
        return self._observations(), rewards, done, info

    def _respawn_prey(self) -> None:
        h, w = self.cfg.height, self.cfg.width
        occupied = {(wf["row"], wf["col"]) for wf in self.wolves}
        for _ in range(200):  # bounded: a pathological config (tiny arena, many wolves) must not hang
            r = self.rng.integers(1, h - 1)
            c = self.rng.integers(1, w - 1)
            if (r, c) not in occupied:
                self.prey["row"], self.prey["col"] = int(r), int(c)
                return
        raise RuntimeError("WolfpackEnv: no free cell to respawn the prey (arena too small for the wolf count)")
