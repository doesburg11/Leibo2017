import numpy as np
import pytest
from PIL import Image

from leibo2017.envs.gathering import COLOR_BEAM, COLOR_FACING_MARKER, GatheringEnv, STAND_STILL, USE_BEAM
from leibo2017.envs.wolfpack import WolfpackEnv
from leibo2017.envs.wolfpack import COLOR_FACING_MARKER as WOLFPACK_COLOR_FACING_MARKER
from leibo2017.plotting.video import save_rollout_gif


def _has_color(frame: np.ndarray, color: tuple[int, int, int]) -> bool:
    return bool(np.any(np.all(frame == color, axis=-1)))


def test_gathering_render_returns_full_map_rgb_frame():
    env = GatheringEnv()
    frame = env.render()
    assert frame.shape == (env.cfg.height, env.cfg.width, 3)
    assert frame.dtype == np.uint8


def test_wolfpack_render_returns_full_map_rgb_frame():
    env = WolfpackEnv()
    frame = env.render()
    assert frame.shape == (env.cfg.height, env.cfg.width, 3)
    assert frame.dtype == np.uint8


def test_gathering_render_shows_facing_marker_for_active_players():
    env = GatheringEnv()
    frame = env.render()
    assert _has_color(frame, COLOR_FACING_MARKER)


def test_gathering_render_shows_fired_beam_on_next_frame():
    env = GatheringEnv(rng=np.random.default_rng(0))
    frame_before = env.render()
    assert not _has_color(frame_before, COLOR_BEAM)

    env.step([USE_BEAM, STAND_STILL])
    frame_after = env.render()
    assert _has_color(frame_after, COLOR_BEAM)

    # The beam flash is one-frame only -- it must not persist once no beam
    # was fired on the most recent step.
    env.step([STAND_STILL, STAND_STILL])
    frame_next = env.render()
    assert not _has_color(frame_next, COLOR_BEAM)


def test_wolfpack_render_shows_facing_marker_for_wolves():
    env = WolfpackEnv()
    frame = env.render()
    assert _has_color(frame, WOLFPACK_COLOR_FACING_MARKER)


def test_save_rollout_gif_writes_all_frames_when_they_differ(tmp_path):
    env = GatheringEnv(rng=np.random.default_rng(0))
    frames = [env.render()]
    for _ in range(5):
        env.step([0, 0])
        frames.append(env.render())

    out_path = tmp_path / "rollout.gif"
    save_rollout_gif(frames, str(out_path), fps=10, scale=2)

    assert out_path.exists()
    with Image.open(out_path) as im:
        assert im.n_frames == len(frames)
        assert im.size == (env.cfg.width * 2, env.cfg.height * 2)


def test_save_rollout_gif_handles_identical_consecutive_frames(tmp_path):
    """Pillow's GIF writer merges consecutive pixel-identical frames into
    one with a longer display duration -- expected GIF-format behavior, not
    a bug in save_rollout_gif. This just checks a static rollout (e.g. all
    STAND_STILL) still produces a valid, openable file instead of crashing
    or silently corrupting the output."""
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    out_path = tmp_path / "static.gif"
    save_rollout_gif([frame] * 5, str(out_path), fps=10, scale=2)

    assert out_path.exists()
    with Image.open(out_path) as im:
        assert im.n_frames >= 1


def test_wolfpack_rollout_saves_to_gif(tmp_path):
    env = WolfpackEnv(rng=np.random.default_rng(0))
    frames = [env.render()]
    for _ in range(5):
        env.step([0, 0])
        frames.append(env.render())

    out_path = tmp_path / "wolfpack_rollout.gif"
    save_rollout_gif(frames, str(out_path), fps=10, scale=2)
    assert out_path.exists()


def test_save_rollout_gif_rejects_empty_frames(tmp_path):
    with pytest.raises(ValueError):
        save_rollout_gif([], str(tmp_path / "empty.gif"))


def test_save_rollout_gif_rejects_nonpositive_fps(tmp_path):
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        save_rollout_gif([frame], str(tmp_path / "x.gif"), fps=0)


def test_save_rollout_gif_rejects_nonpositive_scale(tmp_path):
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        save_rollout_gif([frame], str(tmp_path / "x.gif"), scale=0)
