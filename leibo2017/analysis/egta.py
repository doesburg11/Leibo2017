"""Empirical game-theoretic analysis (EGTA), Sec. 2.2 / 5.1 / Fig. 5-6.

Given a pool of policies trained to be "cooperative" (Pi^C) and a pool
trained to be "defecting" (Pi^D) under two different environment settings,
estimate the induced empirical payoff matrix (R, P, S, T) by playing sampled
policy pairs against each other, then classify the result against the
social-dilemma inequalities (Sec. 2, Eqs. 1-4) into Prisoner's Dilemma /
Chicken / Stag Hunt / non-SSD, matching Fig. 6's quadrant scheme:

    fear  = P - S   (x-axis)
    greed = T - R   (y-axis)

    R <= P                    -> "Non-SSD (R<P)"   (violates Eq. 1 outright)
    R > P, fear<=0, greed<=0  -> "Non-SSD (R>P)"    (no fear or greed motive)
    R > P, fear>0,  greed<=0  -> "Stag Hunt"
    R > P, fear<=0, greed>0   -> "Chicken"
    R > P, fear>0,  greed>0   -> "Prisoner's Dilemma"
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from leibo2017.agents.dqn import DQNAgent


def play_episode(env, policy_pair: tuple[DQNAgent, DQNAgent]) -> tuple[float, float]:
    """Run one greedy (epsilon=0) episode; return each player's total return."""
    obs = env.reset()
    done = False
    totals = [0.0, 0.0]
    while not done:
        actions = [agent.act(o, greedy=True) for agent, o in zip(policy_pair, obs)]
        obs, rewards, done, _info = env.step(actions)
        totals[0] += rewards[0]
        totals[1] += rewards[1]
    return totals[0], totals[1]


@dataclass
class PayoffEstimate:
    R: float
    P: float
    S: float
    T: float

    @property
    def fear(self) -> float:
        return self.P - self.S

    @property
    def greed(self) -> float:
        return self.T - self.R

    def classify(self) -> str:
        if self.R <= self.P:
            return "Non-SSD (R<P)"
        if self.fear <= 0 and self.greed <= 0:
            return "Non-SSD (R>P)"
        if self.fear > 0 and self.greed <= 0:
            return "Stag Hunt"
        if self.fear <= 0 and self.greed > 0:
            return "Chicken"
        return "Prisoner's Dilemma"


def estimate_payoff_matrix(
    env_factory,
    pool_c: list[DQNAgent],
    pool_d: list[DQNAgent],
    n_samples: int,
    rng: np.random.Generator | None = None,
) -> PayoffEstimate:
    """Sample `n_samples` policy pairs from each of the 4 role combinations
    (CC, DD, CD, DC) and average returns into R, P, S, T (Sec. 5.1's
    "repeated until convergence of the cell values")."""
    rng = rng or np.random.default_rng()
    r_vals, p_vals, s_vals, t_vals = [], [], [], []

    for _ in range(n_samples):
        c1, c2 = pool_c[rng.integers(len(pool_c))], pool_c[rng.integers(len(pool_c))]
        r1, r2 = play_episode(env_factory(), (c1, c2))
        r_vals += [r1, r2]

        d1, d2 = pool_d[rng.integers(len(pool_d))], pool_d[rng.integers(len(pool_d))]
        p1, p2 = play_episode(env_factory(), (d1, d2))
        p_vals += [p1, p2]

        c, d = pool_c[rng.integers(len(pool_c))], pool_d[rng.integers(len(pool_d))]
        s1, t2 = play_episode(env_factory(), (c, d))  # player 0 cooperates, player 1 defects
        s_vals.append(s1)
        t_vals.append(t2)

        d, c = pool_d[rng.integers(len(pool_d))], pool_c[rng.integers(len(pool_c))]
        t1, s2 = play_episode(env_factory(), (d, c))  # player 0 defects, player 1 cooperates
        t_vals.append(t1)
        s_vals.append(s2)

    return PayoffEstimate(
        R=float(np.mean(r_vals)),
        P=float(np.mean(p_vals)),
        S=float(np.mean(s_vals)),
        T=float(np.mean(t_vals)),
    )
