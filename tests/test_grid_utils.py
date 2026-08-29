import numpy as np

from leibo2017.envs.grid_utils import (
    NORTH, EAST, SOUTH, WEST,
    STEP_FORWARD, STEP_LEFT, STEP_RIGHT,
    beam_cells, local_observation, move_delta,
)


def test_observation_is_orientation_invariant_in_agents_own_frame():
    """Regression test: an earlier version of local_observation had the
    wrong np.rot90 sign, which put "ahead" at increasing row instead of
    decreasing row for East/West orientations specifically (silently wrong
    -- it still produced a *rotated* observation, just the wrong one).
    The real correctness criterion is not "what compass direction is
    drawn where" but "does the agent's own forward/left/right axis line up
    the same way in every orientation," since that's what the network
    actually conditions on.
    """
    h, w = 30, 30
    agent_r, agent_c = 15, 15
    for orientation in (NORTH, EAST, SOUTH, WEST):
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for (r, c) in beam_cells(agent_r, agent_c, orientation, (h, w), 4):
            rgb[r, c] = [255, 0, 0]
        dr, dc = move_delta(orientation, STEP_RIGHT)
        rgb[agent_r + dr, agent_c + dc] = [0, 0, 255]
        dr, dc = move_delta(orientation, STEP_LEFT)
        rgb[agent_r + dr, agent_c + dc] = [0, 255, 0]

        obs = local_observation(rgb, agent_r, agent_c, orientation)

        beam_rows = np.argwhere(obs[0] == 255)[:, 0]
        assert set(beam_rows.tolist()) == {11, 12, 13, 14}, orientation
        right_pos = np.argwhere(obs[2] == 255).tolist()
        left_pos = np.argwhere(obs[1] == 255).tolist()
        assert right_pos == [[15, 11]], orientation
        assert left_pos == [[15, 9]], orientation


def test_local_observation_shape():
    rgb = np.zeros((30, 30, 3), dtype=np.uint8)
    obs = local_observation(rgb, 15, 15, NORTH)
    assert obs.shape == (3, 16, 21)


def test_move_delta_forward_matches_beam_direction():
    for orientation in (NORTH, EAST, SOUTH, WEST):
        fwd = move_delta(orientation, STEP_FORWARD)
        first_beam_cell = beam_cells(10, 10, orientation, (30, 30), 3)[0]
        assert first_beam_cell == (10 + fwd[0], 10 + fwd[1])
