# Wolfpack

Two independent-learner wolves chase a scripted, non-learning prey on a
21×21 grid (each wolf's own local observation crop is smaller, 16×21,
matching the paper's `R^{3×16×21}`). When a wolf touches the prey, every wolf within `capture_radius`
of the prey *at that moment* shares a reward of `r_team`; a wolf that
captures alone, with its partner too far away, gets the smaller `r_lone`
instead (partner gets nothing). The prey then respawns and the episode
continues. This is Leibo et al. (2017)'s Sec. 5.2 game — see
[`__init__.py`](__init__.py) for the exact mechanics and reward values
(`r_team=5.0`, `r_lone=1.0` by default).

## Is this actually a social dilemma?

Open question — not settled by anything measured in this repo. Worth
being precise about what would actually have to be true for it to count.

**There's no explicit cooperate/defect action.** A wolf chooses an action
every step — move, rotate, or stand still (the action space also includes
a beam-use action, but Wolfpack gives it no effect, kept only so both
games share one action space) — and ends up capturing the prey whenever
its final position overlaps the prey's *after* both have moved that step
(see `step()`), not via any dedicated "capture" action. Landing on the
prey alone still pays `r_lone > 0`, so there's no discrete moment where a
wolf gives something up by capturing solo, unlike Gathering's tagging
beam (an explicit, costly, unambiguously aggressive action). So if you're
looking for a tempting alternative move a wolf could make instead of
cooperating, there isn't one at the single-action level.

**The paper's own narrative motivation** (Sec. 5.2) is a pack-hunting SSD
where a coordinated capture pays more than a solo one — not a claim that
Wolfpack is *always* a Stag Hunt specifically. The paper's own reported
experiments have Wolfpack's empirical classification vary with parameters
like capture radius and the group-capture bonus, landing in Chicken,
Stag Hunt, or Prisoner's Dilemma territory depending on the setting — and
`r_team > r_lone` alone doesn't determine that: `egta.py`'s `R`/`P` are
measured *episode returns* under actual trained policies, not the raw
per-capture reward values, so capture-rate differences between policy
types can outweigh the reward ratio. The Stag Hunt story requires one
further thing to actually hold: that pursuing solo captures costs a wolf
future joint-capture opportunities (drifting away from its partner to
chase alone), so that waiting/coordinating risks ending up with nothing
if the partner doesn't reciprocate (`fear = P − S > 0`). That "if" is
doing all the work and is exactly the part this repo hasn't verified for
this implementation's specific defaults.

**Why the skeptical reading is reasonable**: since capturing solo never
costs anything in the moment, the only way "chasing solo" can actually be
a worse *strategy* is if it has a real opportunity cost — spending time
away from your partner instead of positioned for a joint capture. Whether
that opportunity cost is large enough to make solo-rushing a genuine
temptation, or whether "grab the prey whenever you're near it" is just
weakly dominant with no real trade-off either way, is an empirical
question about this specific environment's dynamics (map size, prey
speed, capture radius), not something derivable from the reward ratio
alone.

[`leibo2017/analysis/egta.py`](../../analysis/egta.py) implements the
paper's own way to actually answer this: play trained cooperator-pool and
defector-pool policies against each other, average the resulting returns
into an *empirical* payoff matrix `(R, P, S, T)`, and classify it
(`PayoffEstimate.classify()`) as `R ≤ P` → "Non-SSD (R<P)" (no dilemma:
mutual cooperation isn't even better than mutual defection); else, among
`R > P` outcomes, `fear ≤ 0` and `greed ≤ 0` → "Non-SSD (R>P)" (no
dilemma: cooperation is better and neither risk nor temptation pulls
anyone away from it); `fear > 0`, `greed ≤ 0` → **Stag Hunt**; `fear ≤ 0`,
`greed > 0` → Chicken; both `> 0` → Prisoner's Dilemma. This repo hasn't
run that classification at meaningful scale for Wolfpack specifically
(see the top-level `RESULTS.md`'s EGTA section), so which of these five
outcomes this implementation's defaults actually land in is still open.

## Training result: consistent coordination emerged

Trained with Ray RLlib's DQN backend (optional, additive to this repo's
own from-scratch independent-DQN implementation — see the top-level
README's "Optional: RLlib backend" section, [`run_rllib_train.py`](../../../run_rllib_train.py)
and [`render_rllib_rollout.py`](../../../render_rllib_rollout.py)), 10,000
iterations, 30 parallel env runners, GPU-backed learner:

| iterations | mean episode return | eval capture rate |
|---|---|---|
| 1–1,000 | 2.67 | — |
| 9,001–10,000 | **76.04** | **11 captures/episode, every one a coordinated 2-wolf catch** |

The final greedy eval episode: **11/11 captures with `avg_wolves_per_capture=2.0`** —
never once a solo grab. Both wolves earned `r_team` (5.0 × 11 = 55.0 each)
every single time. Whatever the underlying dynamics turn out to be, this
particular trained policy consistently positioned both wolves together
before every capture rather than settling for solo grabs along the way —
that's a real, observed behavior, whether or not it reflects overcoming a
genuine dilemma in the game-theoretic sense discussed above.

![Wolfpack eval rollout, 11 coordinated captures](wolfpack_captures_demo.gif)

*Gold flash marks the exact frame of each capture (the prey respawning
elsewhere the same frame would otherwise make a catch easy to miss in the
animation); the white marker one cell ahead of each wolf shows its facing
direction — see `render()` in `__init__.py`.*

### Getting there took three fixes, not just more compute

Naively scaling up (parallel env runners, bigger batch size) barely moved
the needle — mean return crept from 0.087 to 0.111 over several attempts.
The actual breakthrough was realizing RLlib's DQN, with `training_intensity`
left at its default, does **exactly one gradient update per training
iteration** regardless of how much data gets collected — so 1,000
iterations trained the network on only 1,000 total batches, no matter how
many parallel env runners were collecting experience. Setting
`--updates-per-iteration 20` (an explicit `training_intensity`) was the
single change that turned a flat, noisy ~0.1 mean return into a steep,
still-climbing curve reaching 76+ by iteration 10,000. Two smaller,
related fixes (correctly scaling DQN's epsilon-decay schedule and
target-network update frequency for parallel env runners, both of which
silently assumed a single env runner) compounded on top of that. See the
top-level README's "Optional: RLlib backend" section and the git history
for the full diagnostic trail.

### Reproducing

```bash
pip install -r requirements-rllib.txt   # from repo root
python render_rllib_rollout.py --game wolfpack --algo DQN \
    --iterations 10000 --num-env-runners 30 --gpu \
    --train-batch-size 256 --updates-per-iteration 20
```

Saves a checkpoint to `~/simulation_results/ray_results/DQN_Leibo2017_Wolfpack_<timestamp>/`
and renders a greedy eval episode to
`output/render_rllib_rollout/wolfpack_dqn_rollout.gif`.
