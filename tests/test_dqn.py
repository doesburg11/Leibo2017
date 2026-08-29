import numpy as np

from leibo2017.agents.dqn import DQNAgent, DQNConfig
from leibo2017.envs.grid_utils import OBS_SHAPE


def test_act_returns_valid_action():
    agent = DQNAgent(DQNConfig(seed=0))
    obs = np.zeros(OBS_SHAPE, dtype=np.uint8)
    a = agent.act(obs)
    assert 0 <= a < agent.cfg.num_actions


def test_epsilon_decays_linearly_from_1_to_0_1():
    agent = DQNAgent(DQNConfig(seed=0, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_steps=100))
    assert agent.epsilon() == 1.0
    agent.total_steps = 50
    assert abs(agent.epsilon() - 0.55) < 1e-6
    agent.total_steps = 1000
    assert abs(agent.epsilon() - 0.1) < 1e-9


def test_train_step_learns_without_error():
    agent = DQNAgent(DQNConfig(seed=0, min_buffer_size=4, train_batch_size=4, batch_capacity=100))
    obs = np.zeros(OBS_SHAPE, dtype=np.uint8)
    next_obs = np.ones(OBS_SHAPE, dtype=np.uint8)
    for _ in range(10):
        agent.observe(obs, 0, 1.0, next_obs, False)
    loss = agent.train_step()
    assert loss is not None
    assert loss >= 0.0


def test_target_network_updates_on_schedule():
    agent = DQNAgent(DQNConfig(seed=0, min_buffer_size=2, train_batch_size=2, target_update_interval=3))
    rng = np.random.default_rng(0)
    # Non-zero, varied observations: an all-zero input makes dL/dW1 exactly
    # zero by construction (backprop through a linear layer scales the
    # upstream gradient by the input), which would make this test pass or
    # fail based on an artifact of the *test's* input rather than of the
    # target-network update logic under test.
    for _ in range(4):
        obs = rng.integers(0, 256, size=OBS_SHAPE, dtype=np.uint8)
        next_obs = rng.integers(0, 256, size=OBS_SHAPE, dtype=np.uint8)
        agent.observe(obs, int(rng.integers(8)), 1.0, next_obs, False)
    before = agent.target_net.fc1.weight.clone()
    for _ in range(3):
        agent.train_step()
    after = agent.target_net.fc1.weight
    assert not torch_allclose(before, after)


def torch_allclose(a, b):
    import torch
    return torch.allclose(a, b)


def test_seed_makes_network_initialization_reproducible():
    """Regression test (Codex review, 2026-08-29): DQNConfig.seed seeded the
    buffer/epsilon RNGs but not nn.Linear's weight init, which draws from
    PyTorch's global RNG -- so two agents built with the same seed used to
    get different initial Q-networks."""
    a = DQNAgent(DQNConfig(seed=123))
    b = DQNAgent(DQNConfig(seed=123))
    assert torch_allclose(a.q_net.fc1.weight, b.q_net.fc1.weight)
    assert torch_allclose(a.q_net.out.bias, b.q_net.out.bias)

    c = DQNAgent(DQNConfig(seed=456))
    assert not torch_allclose(a.q_net.fc1.weight, c.q_net.fc1.weight)
