"""Gathering, the apple-collection SSD from Leibo et al. (2017), Sec. 5.1.

Two players move around an orchard collecting apples (+1 reward each,
removed on pickup, respawns after `n_apple` steps). Each player can also
fire a straight beam; a player hit twice is "tagged" and removed from the
game for `n_tagged` steps. Tagging gives no reward -- its only purpose is to
remove a rival from competition over apples.

The paper does not publish the exact map (only the illustrative Fig. 3
schematic, a "plus"-shaped apple field with the two players approaching
from opposite ends of an open corridor) -- see the "Blind spots" section of
the top-level README for what is reconstructed here vs. taken verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from leibo2017.envs.grid_utils import (
    NUM_ACTIONS,
    OBS_SHAPE,
    ROTATE_LEFT,
    ROTATE_RIGHT,
    STAND_STILL,
    USE_BEAM,
    beam_cells,
    local_observation,
    move_delta,
    rotate_orientation,
)

EMPTY, WALL, APPLE = 0, 1, 2
BEAM_RANGE = 15  # matches the forward extent of the observation window

# Colors for rendering the grid to RGB before cropping (Sec. 4: agents
# render blue to themselves, light-blue to a teammate, red to an opponent --
# there is no teammate in a 2-player Gathering game, so only self/opponent
# colors are used here).
COLOR_EMPTY = (0, 0, 0)
COLOR_WALL = (128, 128, 128)
COLOR_APPLE = (0, 255, 0)
COLOR_SELF = (0, 0, 255)
COLOR_OPPONENT = (255, 0, 0)
COLOR_BEAM = (255, 255, 0)


@dataclass
class _Player:
    row: int
    col: int
    orientation: int
    start_row: int
    start_col: int
    tagged_timer: int = 0
    hits_taken: int = 0  # beam hits accumulated since the last reset (2 -> tagged)
    beam_uses: int = 0
    apples_collected: int = 0
    active_steps: int = 0  # steps not removed from the game (for beam-use-rate normalization)


@dataclass
class GatheringConfig:
    n_apple: int = 20  # respawn delay for a collected apple, in steps
    n_tagged: int = 25  # steps a tagged player is removed from the game
    episode_length: int = 1000
    height: int = 13
    width: int = 33
    # "Plus"-shaped orchard occupying the central third of the map, per the
    # Fig. 3 schematic. See README "Blind spots" -- exact layout unpublished.
    orchard_arm_len: int = 5
    orchard_arm_width: int = 3


class GatheringEnv:
    """Two-player independent-learner Gathering game.

    Step API mirrors the essentials of a Gym env without adding a Gym
    dependency: `reset() -> obs`, `step(actions) -> (obs, rewards, done,
    info)`, where `obs` is a tuple of two (3, 16, 21) uint8 arrays and
    `actions`/`rewards` are length-2 sequences (player 0, player 1).
    """

    num_agents = 2
    num_actions = NUM_ACTIONS
    obs_shape = OBS_SHAPE

    def __init__(self, config: GatheringConfig | None = None, rng: np.random.Generator | None = None):
        self.cfg = config or GatheringConfig()
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

        cy, cx = cfg.height // 2, cfg.width // 2
        aw, al = cfg.orchard_arm_width // 2, cfg.orchard_arm_len
        self._apple_sites: list[tuple[int, int]] = []
        for r in range(cy - aw, cy + aw + 1):
            for c in range(cx - al, cx + al + 1):
                if 0 < r < cfg.height - 1 and 0 < c < cfg.width - 1:
                    self._apple_sites.append((r, c))
        for r in range(cy - al, cy + al + 1):
            for c in range(cx - aw, cx + aw + 1):
                if 0 < r < cfg.height - 1 and 0 < c < cfg.width - 1:
                    self._apple_sites.append((r, c))
        self._apple_sites = sorted(set(self._apple_sites))
        self._static_grid = grid
        # Players start at opposite ends of the corridor, facing the orchard.
        self._start_positions = [
            (cy, 2, 1),  # (row, col, orientation=EAST)
            (cy, cfg.width - 3, 3),  # orientation=WEST
        ]

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        self._t = 0
        self._apple_present = {site: True for site in self._apple_sites}
        self._apple_respawn_at: dict[tuple[int, int], int] = {}
        self.players = [
            _Player(row=r, col=c, orientation=o, start_row=r, start_col=c)
            for (r, c, o) in self._start_positions
        ]
        return self._observations()

    def _grid_occupancy(self) -> np.ndarray:
        grid = self._static_grid.copy()
        for site, present in self._apple_present.items():
            if present:
                grid[site] = APPLE
        return grid

    def _render_rgb(self, viewer_idx: int) -> np.ndarray:
        grid = self._grid_occupancy()
        rgb = np.zeros((*grid.shape, 3), dtype=np.uint8)
        rgb[grid == WALL] = COLOR_WALL
        rgb[grid == APPLE] = COLOR_APPLE
        for i, p in enumerate(self.players):
            if p.tagged_timer > 0:
                continue
            color = COLOR_SELF if i == viewer_idx else COLOR_OPPONENT
            rgb[p.row, p.col] = color
        return rgb

    def _observations(self) -> tuple[np.ndarray, np.ndarray]:
        obs = []
        for i, p in enumerate(self.players):
            rgb = self._render_rgb(i)
            obs.append(local_observation(rgb, p.row, p.col, p.orientation))
        return tuple(obs)

    def step(self, actions):
        cfg = self.cfg
        rewards = [0.0, 0.0]
        beams_fired: list[tuple[int, list[tuple[int, int]]]] = []

        # Tick existing cooldowns down *before* this step's beam hits are
        # resolved, so a player newly tagged this step gets the full
        # n_tagged steps of removal rather than n_tagged - 1 (a player
        # already mid-cooldown ticks normally).
        for p in self.players:
            if p.tagged_timer > 0:
                p.tagged_timer -= 1

        order = self.rng.permutation(self.num_agents)
        occupied = {(p.row, p.col) for p in self.players if p.tagged_timer == 0}

        for i in order:
            p = self.players[i]
            if p.tagged_timer > 0:
                continue
            a = int(actions[i])
            p.active_steps += 1
            if a in (ROTATE_LEFT, ROTATE_RIGHT):
                p.orientation = rotate_orientation(p.orientation, a)
            elif a == USE_BEAM:
                p.beam_uses += 1
                cells = beam_cells(p.row, p.col, p.orientation, self._static_grid.shape, BEAM_RANGE)
                # A beam is blocked by the orchard's outer wall/border but
                # passes freely over open floor and apples (unspecified by
                # the paper; treated as line-of-sight over the floor).
                blocked = []
                for (r, c) in cells:
                    if self._static_grid[r, c] == WALL:
                        break
                    blocked.append((r, c))
                beams_fired.append((i, blocked))
            elif a == STAND_STILL:
                pass
            else:
                dr, dc = move_delta(p.orientation, a)
                nr, nc = p.row + dr, p.col + dc
                if self._static_grid[nr, nc] != WALL and (nr, nc) not in occupied:
                    occupied.discard((p.row, p.col))
                    p.row, p.col = nr, nc
                    occupied.add((nr, nc))
                    site = (nr, nc)
                    if self._apple_present.get(site):
                        self._apple_present[site] = False
                        self._apple_respawn_at[site] = self._t + cfg.n_apple
                        p.apples_collected += 1
                        rewards[i] += 1.0

        for shooter_idx, cells in beams_fired:
            for j, p in enumerate(self.players):
                if j == shooter_idx or p.tagged_timer > 0:
                    continue
                if (p.row, p.col) in cells:
                    p.hits_taken += 1
                    if p.hits_taken >= 2:
                        p.tagged_timer = cfg.n_tagged
                        p.hits_taken = 0
                        p.row, p.col = p.start_row, p.start_col

        for site, ready_at in list(self._apple_respawn_at.items()):
            if self._t + 1 >= ready_at:
                self._apple_present[site] = True
                del self._apple_respawn_at[site]

        self._t += 1
        done = self._t >= cfg.episode_length
        info = {
            "beam_use_rate": [
                (p.beam_uses / p.active_steps) if p.active_steps > 0 else 0.0 for p in self.players
            ],
        }
        return self._observations(), rewards, done, info
