import pytest

from leibo2017.analysis.egta import PayoffEstimate, estimate_payoff_matrix


def test_classify_prisoners_dilemma():
    # Canonical PD payoffs from Fig. 1: R=3, P=1, S=0, T=4.
    est = PayoffEstimate(R=3, P=1, S=0, T=4)
    assert est.fear == 1  # P - S = 1 > 0
    assert est.greed == 1  # T - R = 1 > 0
    assert est.classify() == "Prisoner's Dilemma"


def test_classify_stag_hunt():
    # Canonical Stag Hunt payoffs: R=4, P=1, S=0, T=3.
    est = PayoffEstimate(R=4, P=1, S=0, T=3)
    assert est.fear == 1  # P - S = 1 > 0
    assert est.greed == -1  # T - R = -1 <= 0
    assert est.classify() == "Stag Hunt"


def test_classify_chicken():
    # Canonical Chicken payoffs: R=3, P=0, S=1, T=4.
    est = PayoffEstimate(R=3, P=0, S=1, T=4)
    assert est.fear == -1  # P - S <= 0
    assert est.greed == 1  # T - R > 0
    assert est.classify() == "Chicken"


def test_classify_non_ssd_r_less_than_p():
    est = PayoffEstimate(R=1, P=2, S=0, T=3)
    assert est.classify() == "Non-SSD (R<P)"


def test_classify_non_ssd_no_fear_or_greed():
    est = PayoffEstimate(R=3, P=1, S=3, T=1)
    assert est.fear <= 0
    assert est.greed <= 0
    assert est.classify() == "Non-SSD (R>P)"


def test_estimate_payoff_matrix_rejects_empty_pools():
    with pytest.raises(ValueError):
        estimate_payoff_matrix(lambda: None, [], [object()], n_samples=5)
    with pytest.raises(ValueError):
        estimate_payoff_matrix(lambda: None, [object()], [], n_samples=5)


def test_estimate_payoff_matrix_rejects_non_positive_samples():
    with pytest.raises(ValueError):
        estimate_payoff_matrix(lambda: None, [object()], [object()], n_samples=0)
