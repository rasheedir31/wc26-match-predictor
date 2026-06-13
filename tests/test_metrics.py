# Evaluation metric tests with hand-checkable values.

from __future__ import annotations

import numpy as np

from wc26.evaluate import metrics


def test_perfect_predictions_are_optimal() -> None:
    proba = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    y = np.array([0, 2])
    assert metrics.log_loss(proba, y) < 1e-10
    assert metrics.brier_score(proba, y) == 0.0
    assert metrics.rps(proba, y) == 0.0
    assert metrics.accuracy(proba, y) == 1.0


def test_uniform_brier_and_rps_known_values() -> None:
    proba = np.array([[1 / 3, 1 / 3, 1 / 3]])
    y = np.array([0])  # home
    # Brier: (1/3-1)^2 + (1/3)^2 + (1/3)^2 = 4/9 + 1/9 + 1/9 = 6/9.
    assert np.isclose(metrics.brier_score(proba, y), 6 / 9)
    # RPS: cdf_pred=[1/3,2/3], cdf_obs=[1,1]; ((2/3)^2+(1/3)^2)/2 = (4/9+1/9)/2 = 5/18.
    assert np.isclose(metrics.rps(proba, y), 5 / 18)


def test_rps_rewards_ordinal_closeness() -> None:
    # True outcome is home (0). A prediction leaning to the *adjacent* draw should
    # score better (lower RPS) than one leaning to the far-away away outcome.
    near = np.array([[0.5, 0.4, 0.1]])
    far = np.array([[0.5, 0.1, 0.4]])
    y = np.array([0])
    assert metrics.rps(near, y) < metrics.rps(far, y)


def test_calibration_curve_shape() -> None:
    rng = np.random.default_rng(0)
    proba = rng.dirichlet([1, 1, 1], size=500)
    y = rng.integers(0, 3, size=500)
    curve = metrics.calibration_curve(proba, y, class_idx=0, n_bins=10)
    assert set(curve) == {"mean_predicted", "empirical", "count"}
    assert len(curve["mean_predicted"]) == len(curve["empirical"]) == len(curve["count"])
    assert sum(curve["count"]) == 500
