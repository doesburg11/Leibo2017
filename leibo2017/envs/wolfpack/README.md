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

**Why the skeptical reading is reasonable — and may go further than just
doubting greed.** Compare cooperate (`C`: stay near your partner, pursue
joint captures) against defect (`D`: rush independently, grab the prey
whenever available) as whole-episode strategies — this is exactly how
`egta.py`'s cooperator-pool/defector-pool methodology operationalizes
them. Four cells: `R` (both `C`) — coordinated, mostly team captures;
`P` (both `D`) — independent, still nets occasional solo captures; `T`
(`D` vs `C`) — the defector rushes ahead uncontested; `S` (`C` vs `D`) —
the cooperator's partner isn't around when a capture opportunity appears.

The usual Stag Hunt story needs `S` to be low — the classic "sucker's
payoff" of getting nothing for trying to coordinate with a partner who
didn't reciprocate. But that requires a cooperative policy to actually
*forgo* captures while waiting around for its partner. Since touching the
prey is never locally costly (previous point), there's no reason a
sensible `C` policy would ever decline an available capture just because
it's "trying to cooperate" — it would grab `r_lone` whenever the chance
arose too, same as `D` would, just from a position that's usually closer
to its partner. If that's right, `S` isn't the empty-handed sucker's
payoff the Stag Hunt story needs — a cooperator still captures sometimes
— which means `fear = P − S` could end up small or even `≤ 0`, not just
`greed`. That would land the classification in **Non-SSD (R>P)**:
cooperation still pays better in aggregate, but neither fear nor greed
actually pulls anyone away from it, so by the paper's own scheme it
wouldn't count as a dilemma at all.

Whether that's actually how it plays out — or whether the opportunity
cost of drifting away from your partner *is* large enough to make
solo-rushing a genuine temptation after all — is an empirical question
about this specific environment's dynamics (map size, prey speed,
capture radius), not something derivable from the reward ratio alone.

### What the reward ratio alone gives you: an idealized one-shot matrix

Building a textbook-style, single-shot payoff matrix directly from
`r_team`/`r_lone` — `C` = commit to hunting together, `D` = rush the prey
alone, and two idealizing assumptions (a clean simultaneous single
choice; if both defect, it's a 50/50 race for the same prey, so each
nets `r_lone / 2` in expectation) —

| | Partner: **C** | Partner: **D** |
|---|---|---|
| **You: C** | 5, 5 | 0, 1 |
| **You: D** | 1, 0 | 0.5, 0.5 |

`R=5 > P=0.5`, `fear = P−S = 0.5 > 0`, `greed = T−R = 1−5 = −4 ≤ 0` →
**Stag Hunt**, matching the paper's narrative motivation exactly. This
*is* a genuine social dilemma by `classify()`'s own criteria — two
equilibria, mutual cooperation Pareto-better, but reaching it needs
trusting your partner won't leave you with the sucker's payoff. It just
isn't proof that Wolfpack's actual sequential dynamics reduce to it —
the idealizing assumptions (especially "a cooperator gets `S=0`, exactly
zero") are exactly what the previous section's skepticism questions.

### What's actually measured (smoke-test scale): the idealization doesn't hold

Running `run_egta_wolfpack.py` for real (defaults: pool size 4,
20,000 training steps/policy — trains Pi^C at high capture-radius/bonus
vs. Pi^D at low radius/bonus, per the paper's Fig. 5-6 method, then
cross-plays them over full episodes):

```
R=0.425  P=0.200  S=0.300  T=0.275
fear=-0.100  greed=-0.150  ->  Non-SSD (R>P)
```

Both `fear` and `greed` came back negative. In particular `S=0.300 > P=0.200` —
the "cooperator" facing a defecting partner is *not* left empty-handed;
it still out-earns the mutual-defection baseline, consistent with the
mechanism the skeptical reading predicted (a sensible cooperative policy
still grabs available captures rather than deliberately abstaining). At this
scale, real trained-policy Wolfpack lands in **Non-SSD (R>P)**: mutual
cooperation is still better in aggregate, but no dilemma actually pulls
anyone away from it.

Caveats before reading too much into this: smoke-test scale (20,000
steps/policy vs. the paper's 40,000,000), pool size 4, `n_egta_samples=20` —
nowhere near enough to be a confident measurement, and result variance
across seeds hasn't been checked. It's a real number, not a guess, but
it's one data point, not a settled answer for this environment.

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
