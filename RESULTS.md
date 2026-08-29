# Results

**Status (2026-08-29): initial build complete. 23/23 unit tests passing. All
five top-level experiment scripts verified to run end-to-end at smoke-test
scale (thousands, not tens-of-millions, of steps). No paper-scale
(40,000,000-step) run has been attempted — see README's "Running at paper
scale vs. as a smoke test."** This file records what was actually verified,
including two real bugs the verification process caught, not a claim of a
completed replication.

## 1. Observation rotation — a real, silent bug caught before it shipped

The agent-centered observation crop (`leibo2017/envs/grid_utils.py::local_
observation`) rotates the world so the agent always "faces up" in its own
view, regardless of true orientation. The first implementation rotated the
whole padded grid with `np.rot90(k=-orientation)` and separately
hand-derived a point-mapping formula to track where the agent landed in the
rotated array.

Directly testing the *only property that actually matters* — that beam
direction, forward-move, and left/right-move all land in the same
agent-relative position in the observation regardless of true compass
orientation — caught that this was wrong specifically for East/West
orientations (North/South happened to work, since a 180°-related pair of
bugs partially canceled out). The array was rotated one way; the
hand-derived point formula tracked the opposite rotation. The result
wasn't a crash or an obviously-wrong image — it was a *plausible-looking*,
internally coherent but wrong observation, exactly the kind of bug that
would have silently taught two agents-out-of-four orientations a
systematically mirrored view of their own beam and turning direction.

Fixed by dropping the point-tracking approach entirely: crop a symmetric
square patch centered on the agent first (a square's center index is a
fixed point of any 90° rotation, so no coordinate bookkeeping is needed
after rotating it), then slice the final asymmetric window out of that.
Verified with `tests/test_grid_utils.py::
test_observation_is_orientation_invariant_in_agents_own_frame`, which
checks beam cells and forward/left/right move targets land in identical
observation coordinates for all four true orientations.

## 2. Tag-cooldown off-by-one

`GatheringEnv.step` originally decremented every tagged player's cooldown
timer at the *end* of the same step that a second beam hit set it. A player
tagged this step immediately lost one step of its `n_tagged` cooldown,
making the actual removal duration `n_tagged - 1` rather than `n_tagged`.
Caught by `tests/test_gathering_env.py::
test_two_beam_hits_tags_and_removes_player`, which asserts the exact
cooldown value after a second hit. Fixed by moving the decrement to the
start of `step`, before that step's own tagging is resolved, so a
newly-tagged player's timer starts ticking the step *after* it's set.

## 3. Wolfpack's scripted prey was initially uncatchable

Wolfpack has no learning prey to fall back on (see README "Blind spots"),
so a scripted stand-in was needed. The first version made the prey
unconditionally flee toward maximum Manhattan distance from the nearest
wolf. Directly measuring capture counts under a uniform-random wolf policy
— the same exploration regime epsilon-greedy DQN starts from — found
**zero captures in 50,000 steps**. Tracing the prey's trajectory showed why:
it finds a corner of the bounded 21×21 arena that's already maximally far
from both (randomly-walking, non-pursuing) wolves and stays there, since
"maximize distance from the nearest wolf" has no notion of "already safe
enough to stop." Undirected random-walk wolves essentially never reach that
corner by chance within any reasonable step budget. This would have left
DQN training with no reward signal to ever bootstrap from, regardless of
`--total-steps`.

Fixed by only fleeing when a wolf is within `prey_detection_range` (tuned
to 3 cells) and only with probability `prey_flee_prob` (tuned to 0.5)
even then. Re-measured under the same uniform-random wolf policy:

| detection range | flee prob | captures / 20,000 random steps |
|---|---|---|
| 6 | 0.8 | 1 |
| 3 | 0.8 | 6 |
| 3 | 0.5 | 14 |
| 2 | 0.5 | 29 |
| 2 | 0.3 | 44 |
| 1 | 0.5 | 53 |

`(3, 0.5)` was kept as the default: frequent enough that early random
exploration can find a reward signal, not so passive that the task requires
no real pursuit skill (captures under pure random play come back almost
entirely solo, `avg_wolves_per_capture ≈ 1.0` — the team-capture bonus the
paper's whole experiment is about is not something a random policy stumbles
into, only something a trained one can achieve).

## 4. Smoke-test verification of all five run scripts

With the above two environment bugs fixed, each top-level script was run
at a small scale purely to confirm the full env → independent-DQN training
→ social-behavior-metric → plotting pipeline executes correctly and
produces non-degenerate numbers (not that training converged):

- `run_experiment1_gathering.py` (2×2 grid, 300 steps/cell): ran clean,
  produced a heatmap and `results.json` with distinct aggressiveness values
  per cell.
- `run_experiment2_wolfpack.py`: at 300–3,000 steps/cell, all cells came
  back `NaN` for wolves-per-capture — expected, not a bug: an episode is
  1,000 steps, a 300-step budget never completes one, and even a completed
  short episode has low odds of any capture happening within it (see §3).
  At 8,000 steps/cell, real numbers appeared (`wolves_per_capture` from
  1.00 to 1.25 across a radius/bonus 2×2 grid), qualitatively consistent
  with the paper's claim that larger radius/bonus should push cooperation
  up from the solo-capture baseline of 1.0.
- `run_experiment3_agent_params.py`: all six ablation panels (discount,
  batch size, network size, for both games) ran end-to-end at 3,000
  steps/point without error.
- `run_egta_gathering.py` / `run_egta_wolfpack.py` (pool size 2, a few
  thousand steps/policy): produced degenerate all-zero payoff estimates at
  very low step counts (undertrained greedy policies collect nothing / catch
  nothing), and clearly non-degenerate estimates once given a bit more
  training (e.g. `R=212.30, P=133.80, S=154.50, T=144.70` for Gathering at
  15,000 steps/policy) — confirming the EGTA sampling and classification
  machinery itself is correct, independent of whether the underlying
  policies are any good yet. The `--sweep` scatter extension was also
  exercised and produces a Fig. 6-shaped multi-point plot.

No claim beyond "the pipeline runs correctly end-to-end and produces
sane, non-degenerate numbers at small scale" should be drawn from any of
the above — none of these runs come close to the paper's own 40,000,000-
step-per-condition budget, and no statistical comparison across seeds has
been attempted.

## 5. Codex second-opinion review — two more real bugs caught

Per standing project instructions, `codex exec` reviewed
`grid_utils.py`/`gathering.py`/`wolfpack.py`/`dqn.py`/`egta.py`
independently after the above. It found two further genuine bugs beyond
the ones self-caught above:

- **Wolfpack missed captures caused by the prey's own movement.** The
  capture check ran *before* `_move_prey()`, so it only ever caught "a
  wolf stepped onto the prey," never "the prey wandered onto a wolf that
  stayed still." Reproduced directly (a fixed RNG seed with a stationary
  wolf and a prey scripted to step onto it registered zero captures).
  Fixed by moving `_move_prey()` before the capture check and checking
  overlap using both entities' final positions for the step.
- **Gathering could reactivate a tagged player onto an occupied cell.**
  The original fix for the tag-cooldown off-by-one (§2 above) still
  teleported a reactivating player straight to `start_row`/`start_col`
  without checking whether the other player was currently standing there;
  Codex reproduced both players ending up on the same cell. Fixed by
  making reactivation lazy (position isn't touched at tag time — it
  doesn't matter while the player is excluded from rendering/occupancy —
  and is assigned to a free cell, preferring the start cell, only once the
  cooldown actually reaches zero) and by restructuring the cooldown
  decrement to exclude a player tagged in the same step, which also fixed
  a related semantic gap Codex flagged (a player's removal now lasts
  exactly `n_tagged` future steps, not `n_tagged - 1` in one code path and
  `n_tagged` in another depending on which step the tag landed in).

It also flagged that `DQNConfig.seed` seeded the replay-buffer/epsilon
RNGs but not `nn.Linear`'s weight initialization (which draws from
PyTorch's global RNG) — two agents built with the same seed had different
initial Q-networks, undermining reproducibility of any seeded experiment
sweep. Fixed with an explicit `torch.manual_seed(seed)` before network
construction.

Two low-severity robustness gaps were also fixed: `estimate_payoff_matrix`
now rejects empty policy pools / non-positive sample counts up front
instead of failing with a generic index or RNG error several lines later,
and `WolfpackEnv._respawn_prey`'s retry loop is now bounded (200 attempts,
then a clear `RuntimeError`) instead of an unbounded `while True` that
could hang given a pathological arena/wolf-count config.

Codex's concurrency note (that `ReplayBuffer` and `DQNAgent` aren't
thread-safe) was reviewed and not acted on: nothing in this codebase runs
training across threads or processes sharing an agent, so there's no
actual race to fix — noted here rather than silently dropped.

All fixes have regression tests; 29/29 tests passing after this round.
