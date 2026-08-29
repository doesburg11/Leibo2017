# TODO

- Run the experiments for real, not just smoke tests, using the
  project-local conda env (`./create_conda_env.sh` then
  `conda activate ./.conda`). Every `run_*.py` script currently defaults
  to a small `--total-steps` smoke-test scale (see README's "Running at
  paper scale vs. as a smoke test") — no one has actually run
  `run_experiment1_gathering.py` / `run_experiment2_wolfpack.py` /
  `run_experiment3_agent_params.py` / `run_egta_gathering.py` /
  `run_egta_wolfpack.py` long enough to produce a real, converged Fig.
  4/6/7-style result yet.
