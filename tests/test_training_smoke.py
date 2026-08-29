"""Tiny end-to-end smoke tests -- not a claim of learning, just that the
full env/agent/training-loop wiring runs without crashing."""
from leibo2017.training.train_gathering import run_gathering_training
from leibo2017.training.train_wolfpack import run_wolfpack_training


def test_gathering_training_runs():
    result = run_gathering_training(
        n_apple=10, n_tagged=10, total_steps=60, seed=0,
        dqn_kwargs=dict(min_buffer_size=8, train_batch_size=8, batch_capacity=200, epsilon_decay_steps=60),
    )
    assert 0.0 <= result["aggressiveness"] or result["aggressiveness"] != result["aggressiveness"]  # nan allowed
    assert len(result["agents"]) == 2


def test_wolfpack_training_runs():
    result = run_wolfpack_training(
        capture_radius=3.0, r_team=5.0, total_steps=60, seed=0,
        dqn_kwargs=dict(min_buffer_size=8, train_batch_size=8, batch_capacity=200, epsilon_decay_steps=60),
    )
    assert len(result["agents"]) == 2
