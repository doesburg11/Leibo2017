# Multi-agent RL in Sequential Social Dilemmas — Leibo et al. (2017)

A from-scratch replication of:

> Leibo, J. Z., Zambaldi, V., Lanctot, M., Marecki, J., & Graepel, T. (2017).
> *Multi-agent Reinforcement Learning in Sequential Social Dilemmas.* AAMAS 2017.

The paper's question: matrix games like Prisoner's Dilemma treat "cooperate"
and "defect" as atomic, one-shot choices. Real social dilemmas are
temporally extended — cooperativeness is a property of a *policy*, not a
single action, and has to be inferred from a whole trajectory of behavior.
Leibo et al. formalize this as a **Sequential Social Dilemma (SSD)**: a
partially-observable Markov game that, when you empirically extract the
payoffs induced by "cooperative" vs. "defecting" *policies*, reduces to a
classical matrix-game social dilemma (Prisoner's Dilemma, Chicken, or Stag
Hunt) — while the *sequential* structure (how hard cooperation/defection are
to *implement*, not just their payoff values) produces qualitatively
different, sometimes opposite, predictions about when cooperation emerges.

This repo rebuilds their two environments (**Gathering**, an apple-collection
game where a costly "tagging" beam can remove a rival, and **Wolfpack**, a
pack-hunting game where two wolves share a bonus for capturing prey
together) and their independent-DQN training setup directly, then
reproduces their three experiments: the abundance/conflict-cost and
radius/group-bonus sweeps (Fig. 4), the empirical game-theoretic analysis
that classifies the induced matrix game (Fig. 5–6), and the DQN
hyperparameter ablations (Fig. 7).

This is a sibling project to
[SequentialSocialDilemmas](https://github.com/doesburg11/SequentialSocialDilemmas)
(a port of Vinitsky et al.'s modernized RLlib codebase for Cleanup/Harvest/
Gathering/Switch, trained with PPO/IMPALA). That repo is an engineering fork
of someone else's environment design, doesn't include Wolfpack, and doesn't
run the paper's own independent, non-target-shared, growing-replay-buffer
DQN setup. This repo is the from-scratch faithful baseline instead: both of
the paper's own games, its own (much simpler) agent architecture, and its
own three named experiments, by default. An optional Ray RLlib backend can
also be layered on top of the same environments as a cross-check — see
"Optional: RLlib backend" below — without touching any of that default
setup.

## The mechanism

Two independent DQN agents (Sec. 3.1) — each owning its own tiny Q-network
and its own replay buffer, never communicating, each treating the other
purely as part of a non-stationary environment — learn by ordinary
Q-learning gradient descent on transitions sampled from a size-capped,
constantly-refreshed ("growing") batch. Cooperation and defection are never
given as literal actions; they're empirical properties of trained policies,
measured via a social behavior metric (beam-use rate for Gathering,
wolves-per-capture for Wolfpack) and via **empirical game-theoretic
analysis (EGTA)**: sample "cooperative" and "defecting" trained policies
against each other, average the resulting returns into an empirical payoff
matrix (R, P, S, T), and classify it against the same inequalities that
define a matrix-game social dilemma.

- **Gathering** (`leibo2017/envs/gathering.py`): two players collect apples
  (+1 reward, removed on pickup, respawns after `n_apple` steps) in an open
  map with a central orchard. Either player can fire a beam; two hits tag a
  rival, removing them from the game for `n_tagged` steps — a costly action
  whose only function is eliminating competition for apples.
- **Wolfpack** (`leibo2017/envs/wolfpack.py`): two wolves chase a scripted
  prey. A wolf that touches the prey alone gets `r_lone`; if its partner is
  also within `capture_radius`, both get the larger `r_team` instead.
- **Independent DQN** (`leibo2017/agents/dqn.py`): a plain 2-hidden-layer,
  32-unit-by-default MLP over the flattened (3, 16, 21) observation crop,
  trained with the update rule from Sec. 3.1, `epsilon`-greedy with linear
  decay from 1.0 to 0.1 (Sec. 4).
- **EGTA** (`leibo2017/analysis/egta.py`): the Fig. 5 workflow — sample
  policy pairs from cooperator/defector pools, play them out, average
  returns into (R, P, S, T), classify by `fear = P - S` / `greed = T - R`
  into the Fig. 6 quadrant scheme.

## What's matched from the paper vs. necessarily adapted

**Matched, using the paper's own stated numbers where it gives any:**

- Shared 8-action space exactly as listed in Sec. 4: step forward/backward,
  strafe left/right, rotate left/right, beam, stand still.
- Observation shape `(3, 16, 21)` (RGB, 15 cells ahead + the agent's own
  row, 10 cells each side + the agent's own column), agent-centered and
  orientation-dependent (Sec. 4).
- Episode length 1,000 steps; default per-step discount 0.99 (Sec. 4).
- Default network: two hidden layers, 32 units, ReLU, 8 output units
  (Sec. 4).
- Epsilon-greedy exploration decaying linearly from 1.0 to 0.1 (Sec. 4).
- Default replay-buffer ("batch") size 1e5, a FIFO buffer that grows to a
  cap and is then constantly refreshed by discarding old transitions, not
  an unbounded buffer (Sec. 3.1 / footnote 1).
- Q-learning update rule exactly as written in Sec. 3.1 (`Q ← Q + α[r + γ
  max Q' − Q]`), trained by gradient descent on the MSE Bellman residual
  over uniformly-sampled batch transitions.
- Full independence between the two learners: no shared weights, no
  communication, no centralized value function.
- Gathering's reward (+1/apple, apple removed and respawns after
  `n_apple`), tagging mechanic (2 beam hits → removed for `n_tagged`
  steps, no reward for tagging), and its role as the sole source of
  conflict (Sec. 5.1).
- Wolfpack's reward structure (`r_lone` solo capture vs. `r_team` joint
  capture when a partner is within `capture_radius`), Sec. 5.2.
- The Fig. 4 sweep axes (Gathering: `N_apple` abundance × `N_tagged`
  conflict-cost; Wolfpack: capture radius × group-capture bonus) and the
  Fig. 7 ablation factors (discount, batch size, network size).
- The EGTA classification scheme itself: `fear = P - S`, `greed = T - R`,
  and the four-way Fig. 6 quadrant split (Prisoner's Dilemma / Chicken /
  Stag Hunt / non-SSD), implemented exactly as stated in Sec. 2.2 and Eqs.
  1–4 — see `leibo2017/analysis/egta.py`'s docstring and
  `tests/test_egta.py` (checked against the paper's own canonical PD/
  Chicken/Stag Hunt payoff numbers from Fig. 1).

**Blind spots — where the paper doesn't say enough to reproduce byte-for-byte,
and what was chosen instead (all in code, clearly marked):**

- **Map layout and size for both games are never published** — only Fig.
  3's small illustrative screenshot (a "plus"-shaped apple orchard for
  Gathering; an open arena of unstated size for Wolfpack). `gathering.py`
  reconstructs a 13×33 map with a plus-shaped orchard in the middle third
  and players starting at opposite ends of the corridor, matching Fig. 3's
  visual proportions; `wolfpack.py` uses an arbitrary 21×21 open arena.
  Neither is verified against DeepMind's actual level file, which isn't
  published in the paper. Light corroboration, not confirmation: Vinitsky
  et al.'s independent Gathering implementation (ported in the sibling
  [SequentialSocialDilemmas](https://github.com/doesburg11/SequentialSocialDilemmas)
  repo) also lands on a similarly-shaped multi-cluster apple orchard and
  the identical `(3, 16, 21)` observation shape — but it's their own
  reconstruction from the same Fig. 3 schematic, not a second, independent
  data point grounded in DeepMind's actual level file, so it doesn't
  settle the ambiguity, just shows two independent guesses converging on
  the same general shape.
- **The Q-network's exact architecture is ambiguous.** Sec. 4 says only
  "two hidden layers with 32 units, interleaved with rectified linear
  layers" — no mention of convolution, despite the network's input being
  an RGB image and the paper citing a DQN lineage (Mnih et al. 2015) that
  uses convolutional layers over pixel input. Taken literally, this is
  read here as a plain MLP over the flattened observation
  (`leibo2017/agents/dqn.py::QNetwork`). A convolutional reading is
  equally defensible from the surrounding context; it just isn't what the
  sentence itself says.
- **Learning rate and optimizer are never given** — the paper only says
  "trained through gradient descent." Adam with `lr=1e-4` is used here
  (`DQNConfig.learning_rate`), an arbitrary but standard choice for this
  network scale.
- **Whether a separate target network is used for the `max_a' Q(s',a')`
  term is not shown.** The literal Sec. 3.1 equation has no target
  network, but the paper frames its whole approach as an application of
  the cited Mnih et al. (2015) DQN, which does use one. A target network
  is included by default (`DQNConfig.use_target_network=True`,
  `target_update_interval=1000`) since omitting it against a paper that
  explicitly leans on Mnih et al. felt like the wrong default; it can be
  turned off to match the bare equation exactly.
- **The epsilon-decay schedule's length is unstated** — "decaying linearly
  over time (from 1.0 to 0.1)" doesn't say over how many steps. Made
  configurable (`DQNConfig.epsilon_decay_steps`), defaulting to 1,000,000.
- **A real inconsistency in the paper itself, not just an omission:**
  Sec. 3.1's text says batch sizes of "1e5 (our default) and 1e6" are
  compared, but Fig. 7's own legend labels the two compared batch-size
  curves "1e+04" and "1e+05". `run_experiment3_agent_params.py`
  reproduces Fig. 7's literal legend values (1e4 vs. 1e5) for that
  specific ablation, while the single-run default elsewhere stays at 1e5
  per the Sec. 3.1 text — see that script's docstring.
- **Fig. 7's network-size ablation (16 vs. 64) doesn't include the Sec. 4
  stated default (32)** — that's the paper's own experimental design, not
  a gap on this end, just worth knowing going in: the ablation brackets
  the default rather than including it.
- **Wolfpack's prey behavior is never specified at all.** Sec. 5.2 calls
  it "a third player" but never states whether it's a learning agent, a
  scripted opponent, or something else — only that "the wolves learn to
  catch the prey over the course of training," implying the prey itself
  is not the subject of the reported learning curves. This repo scripts
  the prey (`WolfpackEnv._move_prey`) to flee the nearest wolf only when
  one is within `prey_detection_range` (default 3 cells), and only with
  probability `prey_flee_prob` (default 0.5) when it does. Both constants
  were tuned, not guessed once and left: an earlier version had the prey
  unconditionally maximize distance from the nearest wolf, which let it
  retreat into a permanently safe corner of the bounded arena that
  undirected wolves essentially never reached again — verified directly
  (`0` captures in 50,000 random-policy steps) — which would leave
  epsilon-greedy exploration with no reward signal to ever bootstrap
  learning from. The tuned values give real DQN training something to
  find (`~1` capture per ~1,400 random steps, confirmed empirically) while
  still requiring a wolf to actually approach.
- **The exact shape/trigger of Wolfpack's capture mechanic is
  under-specified.** "If an agent is inside the blue diamond-shaped region
  around the prey when a capture occurs" is read here as: a capture is
  triggered by any wolf occupying the prey's exact cell; the resulting
  reward (`r_team` to both, or `r_lone` to the capturer alone) depends on
  whether the *other* wolf is within `capture_radius` (Manhattan/L1
  distance, matching "diamond-shaped") at that instant. Whether the
  original used Euclidean distance, a different trigger radius for the
  capture itself vs. the team-bonus radius, or some other convention isn't
  stated.
- **Gathering's beam range, line-of-sight, and hit-memory are
  unspecified.** Implemented here: beam range of 15 cells (matching the
  observation's forward extent), blocked only by walls (passes freely over
  open floor and apples), and hit count that accumulates indefinitely
  (never decays) until the second hit triggers a tag. None of this is
  stated in the paper.
- **Movement-collision resolution when two agents contest the same cell**
  (order of resolution, whether a move silently fails or one agent yields)
  isn't specified; this repo resolves actions in a random per-step order
  and simply blocks a move into an already-occupied cell.
- **How many random seeds went into each Fig. 4 heatmap cell or each Fig. 7
  line-plot point is never stated** — Fig. 7 shows error bars, implying
  multiple seeds/runs were averaged, but the count isn't given. All
  `run_experiment*.py` scripts here run one seed per grid cell / curve
  point by default; wrap them over multiple `--seed` values yourself for
  error bars.
- **Fig. 6's fine-grained scatter (dozens of points spread across all three
  dilemma types) is the least-specified part of the whole paper.** Sec.
  5.1 clearly describes deriving *one* aggregate (R, P, S, T) estimate from
  two fixed policy pools (Π^C from the high-abundance/low-cost corner, Π^D
  from the low-abundance/high-cost corner) — that single-estimate
  procedure is what `run_egta_gathering.py` / `run_egta_wolfpack.py`
  implement by default, and it's directly traceable to the text. But the
  text also says this "generates estimates of R, P, S, and T for the game
  corresponding to each abundance/conflict-cost level tested" — implying
  many distinct payoff-matrix points, one per swept condition, which is
  exactly what Fig. 6 shows, without ever explaining how a *specific*
  swept condition's own (R, P, S, T) tuple is derived from just two
  policy pools. The `--sweep` flag on both EGTA scripts produces a
  Fig. 6-shaped scatter using an explicit, disclosed extension: each swept
  condition's own trained pair supplies its self-play return as R, while
  the fixed Π^D reference pool supplies P/S/T by cross-play against it.
  **This is a documented interpretation, not a verified reproduction of
  DeepMind's actual method** — treat the resulting scatter's overall
  *shape* (a spread across dilemma types) as illustrative, not the paper's
  own numbers.

## Running at paper scale vs. as a smoke test

The paper trains each Fig. 4/7 grid point for **40,000,000 steps**. Across
even a modest sweep (a 5×5 grid, say) that's on the order of a billion
environment steps per figure — a training budget the original authors ran
on DeepMind's own compute, not something this repo attempts to reproduce on
a single machine. Every `run_*.py` script defaults to a much smaller
`--total-steps` (tens of thousands) purely as a smoke test that exercises
the full pipeline (env → independent DQN training → social-behavior metric
→ plot) correctly; it does not claim converged, paper-scale results. Pass
`--total-steps 40000000` (and a long-running machine, ideally GPU-backed —
this repo trains on CUDA automatically when available via PyTorch) to
attempt the real thing.

## Running it

```bash
# Option A: plain pip, into whatever Python/venv you already have active
pip install -r requirements.txt

# Option B: conda, into a local .conda/ this repo owns (not a
# name-registered env in conda's default envs directory -- the --prefix
# flag is what makes it project-local; plain `-f environment.yml` alone
# does not reliably honor the file's own `prefix:` line)
./create_conda_env.sh
conda activate ./.conda

pytest tests/ -q

# Smoke tests (small step counts; see above)
python run_experiment1_gathering.py          # Fig. 4 top: Gathering heatmap
python run_experiment2_wolfpack.py           # Fig. 4 bottom: Wolfpack heatmap
python run_experiment3_agent_params.py       # Fig. 7: discount/batch/network-size ablations
python run_egta_gathering.py --sweep         # Fig. 5-6 left: Gathering EGTA + scatter
python run_egta_wolfpack.py --sweep          # Fig. 5-6 right: Wolfpack EGTA + scatter
```

Each script writes a `results.json` and one or more `.png` figures to
`output/<script-name>/` (override with `--out-dir`). Run
`python run_experiment1_gathering.py --help` (etc.) for the full set of
sweep/step/seed overrides.

## Rendering an episode

`render_rollout.py` renders a played-out Gathering or Wolfpack episode to an
animated GIF — the graphics-utility counterpart to the sibling
SequentialSocialDilemmas repo's `visualization/visualizer_rllib.py`
rollout-to-video tooling (there: RLlib rollout + OpenCV → `.mp4`; here: this
repo's own env/agents + Pillow → `.gif`, no new heavy dependency). Both
`GatheringEnv` and `WolfpackEnv` expose a `render()` method returning the
same full-map RGB frame each already builds internally to construct agent
observations, just not cropped/oriented to any one agent.

```bash
python render_rollout.py --game gathering --random-policy    # fast sanity check, no training
python render_rollout.py --game wolfpack --total-steps 20000 # trains briefly, then renders a greedy rollout
```

Writes `<out-dir>/<game>_rollout.gif` (default `output/render_rollout/`).

## Optional: RLlib backend

Everything above trains with `leibo2017/agents/dqn.py` — a direct,
literal implementation of the paper's own independent-DQN method, and the
only thing every `run_*.py` script and the README's fidelity claims are
about. `run_rllib_train.py` is a separate, purely additive way to train
the *exact same* `GatheringEnv`/`WolfpackEnv` with Ray RLlib's PPO or DQN
instead — useful for checking whether a standard, well-tuned RL library
reaches different conclusions than the paper's own simple setup on
identical environments. It does not touch `leibo2017/agents/dqn.py`,
doesn't change what any existing script does, and isn't required for
anything else in this repo — the base `pip install -r requirements.txt`
has no Ray/RLlib dependency at all.

It also isn't a path to reproducing Fig. 4/6/7 via RLlib: it reports raw
per-agent episode return from RLlib's own training loop, not the paper's
beam-use-rate / wolves-per-capture social-behavior metrics (wiring those
up would need a custom RLlib callback, not built here).

```bash
pip install -r requirements-rllib.txt   # ray[rllib]; tested against ray==2.58.0
python run_rllib_train.py --game gathering --algo PPO --iterations 10
python run_rllib_train.py --game wolfpack --algo DQN --iterations 10 --num-env-runners 30 --gpu
```

By default training is single-process (`num_env_runners=0`, matching the
paper's tiny-scale setup) and CPU-only. `--num-env-runners N` parallelizes
env rollouts across `N` remote workers (real speedup, since env stepping is
usually the bottleneck for envs this small); `--gpu` puts the (also tiny,
2×32-unit) network's training on GPU, which the model is too small to
benefit from much but doesn't hurt either.

Both scripts call `algo.save(checkpoint_dir=...)` when training finishes,
to `output/rllib_checkpoints/<game>_<algo>/` by default (override with
`--checkpoint-dir`). This is a real, reloadable RLlib checkpoint (model
weights included) — distinct from `Algorithm.build()`'s own
`~/ray_results/<timestamp>/` trial directory, which Ray creates as a side
effect regardless and stays empty here: `algorithm_state.pkl`,
`.../rl_module/module_state.pkl` etc. only get written by this explicit
`algo.save()` call, or by driving training through a `ray.tune.Tuner` (not
what these scripts do) instead of a manual `algo.train()` loop.

`leibo2017/envs/rllib_wrappers.py` adapts both envs' plain list-indexed
`reset()`/`step()` API to RLlib's dict-keyed `MultiAgentEnv` API, one
independent `PolicySpec` per agent (no parameter sharing, matching the
paper's own independence assumption), fcnet `[hidden_size, hidden_size]`
matching Sec. 4's default network size. Observations are flattened and
scaled to `[0, 1]` float32 rather than left as the raw `(3, 16, 21)`
uint8 array `leibo2017/agents/dqn.py` uses directly: RLlib's default
Catalog auto-detects any 3D Box as an image and tries to pick a default
CNN, which has no preset for this shape and raises `ValueError` — flattening
sidesteps that and keeps the same literal "MLP, not a described conv
stack" reading used everywhere else in this repo.

`render_rllib_rollout.py` is the RLlib counterpart to `render_rollout.py`
above: it trains the same `run_rllib_train.py` config, then rolls out one
greedy episode and renders it to a GIF, printing per-agent return and the
env's info dict (`captures`/`avg_wolves_per_capture` for Wolfpack,
`beam_use_rate` for Gathering) along the way. The new API stack has no
working `Algorithm.compute_single_action` for multi-agent envs, so it reads
actions from each agent's trained `RLModule` directly — `Columns.ACTIONS`
for DQN (already the greedy/exploit action), or a deterministic sample from
`action_dist_inputs` for PPO.

```bash
python render_rllib_rollout.py --game wolfpack --algo PPO --iterations 10
```

Writes `<out-dir>/<game>_<algo>_rollout.gif` (default
`output/render_rllib_rollout/`) and, like `run_rllib_train.py`, saves a
checkpoint to `output/rllib_checkpoints/<game>_<algo>/` before rendering.

## Known gaps from the paper

- No attempt at the paper's actual 40,000,000-steps-per-condition training
  scale — see "Running at paper scale" above.
- Fig. 6's exact per-condition EGTA construction isn't reproducible from
  the text as written; the `--sweep` scatter is a disclosed interpretation,
  not a verified match to DeepMind's method (see "Blind spots").
- Wolfpack's prey is a scripted, tuned stand-in, not a verified match to
  whatever the original "third player" actually was.
- No convolutional-network variant was tried; only the literal
  "two hidden layers, 32 units" MLP reading of Sec. 4.

## Acknowledgments

Developed with AI coding assistance from [Claude](https://claude.com/claude-code) (Anthropic), which does the implementation, with [Codex](https://openai.com/codex) (OpenAI) acting as an independent second opinion, peer-reviewing Claude's nontrivial code changes.

## References

- Leibo, J. Z., Zambaldi, V., Lanctot, M., Marecki, J., & Graepel, T.
  (2017). *Multi-agent Reinforcement Learning in Sequential Social
  Dilemmas.* AAMAS 2017.
- Mnih, V., et al. (2015). *Human-level control through deep reinforcement
  learning.* Nature, 518(7540), 529–533. (The DQN this paper's Q-learning
  update is built on top of.)
- [SequentialSocialDilemmas](https://github.com/doesburg11/SequentialSocialDilemmas)
  — a sibling repo porting Vinitsky et al.'s modernized-RLlib
  Cleanup/Harvest/Gathering/Switch environments (PPO/IMPALA/DQN), credited
  in full there. That repo is an engineering port of someone else's
  environment design; this one is the from-scratch, paper-faithful
  baseline covering both of Leibo et al.'s own games and their own
  independent-DQN method.
