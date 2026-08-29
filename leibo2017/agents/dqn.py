"""Independent DQN agent, matching Leibo et al. (2017) Sec. 3.1 / 4.

Each of the two players owns its own Q_i : O_i x A_i -> R, "represented by a
deep Q-network." Networks are, per Sec. 4, "two hidden layers with 32 units,
interleaved with rectified linear layers" projecting to 8 output units --
read literally (no mention of convolution), this is a plain MLP over the
flattened (3, 16, 21) observation, which is what is implemented here. Update
rule (Sec. 3.1, Eq. after "Each agent updates its policy given a stored
batch"):

    Q_i(s, a) <- Q_i(s, a) + alpha [ r + gamma * max_a' Q_i(s', a') - Q_i(s, a) ]

trained "through gradient descent on the mean squared Bellman residual ...
with the expectation taken over transitions uniformly sampled from the
batch." The batch (Sec. 3.1, footnote 1: "sometimes called a replay
buffer") is a "growing batch" that is size-capped and constantly refreshed
by discarding old data -- i.e. a plain FIFO replay buffer, which is what
`ReplayBuffer` below implements.

Agents are trained fully independently (Sec. 3.1's "independence
assumption"): no parameter sharing, no communication, no centralized
critic. Each treats the other purely as part of a non-stationary
environment.

Not specified by the paper and chosen here (see README "Blind spots"):
learning rate, optimizer (Adam is used; the paper only says "gradient
descent"), and whether a separate target network is used for the max_a'
term. A target network is included by default since the paper's Q-learning
update is presented as an application of the cited Mnih et al. (2015) DQN,
which uses one; it can be disabled to match the bare equation literally.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from leibo2017.envs.grid_utils import OBS_SHAPE


@dataclass
class DQNConfig:
    hidden_size: int = 32  # Sec. 4 default: "two hidden layers with 32 units"
    num_actions: int = 8
    discount: float = 0.99  # Sec. 4: "default per-time-step discount rate was 0.99"
    batch_capacity: int = 100_000  # Sec. 3.1: "batch sizes of 1e5 (our default)"
    learning_rate: float = 1e-4  # unspecified by the paper; own choice
    train_batch_size: int = 32  # unspecified by the paper; standard DQN-era default
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1  # Sec. 4: "epsilon decaying linearly over time (from 1.0 to 0.1)"
    epsilon_decay_steps: int = 1_000_000  # unspecified schedule length; own choice
    target_update_interval: int = 1_000  # see module docstring: target net is an added assumption
    use_target_network: bool = True
    min_buffer_size: int = 1_000  # steps of pure exploration before training starts
    seed: int | None = None


class QNetwork(nn.Module):
    def __init__(self, hidden_size: int, num_actions: int):
        super().__init__()
        in_dim = int(np.prod(OBS_SHAPE))
        self.fc1 = nn.Linear(in_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


class ReplayBuffer:
    """Fixed-capacity FIFO buffer -- the paper's "growing-but-capped, constantly refreshed" batch."""

    def __init__(self, capacity: int, rng: random.Random):
        self.capacity = capacity
        self._data = deque(maxlen=capacity)
        self._rng = rng

    def push(self, obs, action, reward, next_obs, done):
        self._data.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        batch = self._rng.sample(self._data, batch_size)
        obs, action, reward, next_obs, done = zip(*batch)
        return (
            np.stack(obs),
            np.array(action, dtype=np.int64),
            np.array(reward, dtype=np.float32),
            np.stack(next_obs),
            np.array(done, dtype=np.float32),
        )

    def __len__(self):
        return len(self._data)


class DQNAgent:
    """One independent learner. Owns its own network, buffer, and epsilon schedule."""

    def __init__(self, config: DQNConfig | None = None, device: str = "cpu"):
        self.cfg = config or DQNConfig()
        self.device = torch.device(device)
        gen = torch.Generator().manual_seed(self.cfg.seed) if self.cfg.seed is not None else None
        self._torch_rng = gen
        self._py_rng = random.Random(self.cfg.seed)
        self._np_rng = np.random.default_rng(self.cfg.seed)

        self.q_net = QNetwork(self.cfg.hidden_size, self.cfg.num_actions).to(self.device)
        if self.cfg.use_target_network:
            self.target_net = QNetwork(self.cfg.hidden_size, self.cfg.num_actions).to(self.device)
            self.target_net.load_state_dict(self.q_net.state_dict())
        else:
            self.target_net = self.q_net
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.cfg.learning_rate)
        self.buffer = ReplayBuffer(self.cfg.batch_capacity, self._py_rng)

        self.total_steps = 0
        self.train_steps = 0

    def epsilon(self) -> float:
        cfg = self.cfg
        frac = min(1.0, self.total_steps / max(1, cfg.epsilon_decay_steps))
        return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)

    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        eps = 0.0 if greedy else self.epsilon()
        if not greedy and self._np_rng.random() < eps:
            return int(self._np_rng.integers(self.cfg.num_actions))
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0) / 255.0
            q = self.q_net(x)
            return int(torch.argmax(q, dim=1).item())

    def observe(self, obs, action, reward, next_obs, done) -> None:
        self.buffer.push(obs, action, reward, next_obs, done)
        self.total_steps += 1

    def train_step(self) -> float | None:
        cfg = self.cfg
        if len(self.buffer) < max(cfg.min_buffer_size, cfg.train_batch_size):
            return None
        obs, action, reward, next_obs, done = self.buffer.sample(cfg.train_batch_size)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device) / 255.0
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device) / 255.0
        action_t = torch.as_tensor(action, dtype=torch.int64, device=self.device)
        reward_t = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        done_t = torch.as_tensor(done, dtype=torch.float32, device=self.device)

        q_values = self.q_net(obs_t).gather(1, action_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_obs_t).max(dim=1).values
            target = reward_t + cfg.discount * (1.0 - done_t) * next_q
        loss = F.mse_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.train_steps += 1

        if cfg.use_target_network and self.train_steps % cfg.target_update_interval == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())
