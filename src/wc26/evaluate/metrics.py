# Probabilistic evaluation metrics for 1X2 forecasts.
#
# All take ``proba`` of shape (n, 3) in (home, draw, away) order and integer truths
# ``y_true`` in {0, 1, 2}. Lower is better for log loss / Brier / RPS; higher for
# accuracy. RPS is the football-standard *ordered* metric and is included explicitly.

from __future__ import annotations

import numpy as np

_EPS = 1e-15
_N_CLASSES = 3


def _onehot(y_true: np.ndarray) -> np.ndarray:
    oh = np.zeros((len(y_true), _N_CLASSES))
    oh[np.arange(len(y_true)), y_true] = 1.0
    return oh


def log_loss(proba: np.ndarray, y_true: np.ndarray) -> float:
    # Mean negative log-likelihood of the realised outcomes.
    p = np.clip(proba, _EPS, 1.0)
    return float(-np.log(p[np.arange(len(y_true)), y_true]).mean())


def brier_score(proba: np.ndarray, y_true: np.ndarray) -> float:
    # Multiclass Brier score: mean squared error vs the one-hot outcome (range 0-2).
    return float(((proba - _onehot(y_true)) ** 2).sum(axis=1).mean())


def rps(proba: np.ndarray, y_true: np.ndarray) -> float:
    # Ranked Probability Score for ordered outcomes home < draw < away.
    #
    # RPS = 1/(r-1) * sum_{i=1}^{r-1} ( CDF_pred(i) - CDF_obs(i) )^2, averaged over
    # matches. Rewards probability mass placed *near* the true ordered category.
    pred_cdf = np.cumsum(proba, axis=1)[:, :-1]  # drop last (always 1)
    obs_cdf = np.cumsum(_onehot(y_true), axis=1)[:, :-1]
    return float((((pred_cdf - obs_cdf) ** 2).sum(axis=1) / (_N_CLASSES - 1)).mean())


def accuracy(proba: np.ndarray, y_true: np.ndarray) -> float:
    # Top-1 accuracy (argmax prediction == outcome).
    return float((proba.argmax(axis=1) == y_true).mean())


def all_metrics(proba: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    # Convenience bundle of the four headline metrics.
    return {
        "log_loss": log_loss(proba, y_true),
        "brier": brier_score(proba, y_true),
        "rps": rps(proba, y_true),
        "accuracy": accuracy(proba, y_true),
    }


def calibration_curve(
    proba: np.ndarray, y_true: np.ndarray, class_idx: int, n_bins: int = 10
) -> dict[str, list[float]]:
    # Reliability curve for one class (one-vs-rest).
    #
    # Bins predicted probabilities for ``class_idx`` and reports, per non-empty bin,
    # the mean predicted probability vs the empirical frequency of that outcome.
    # A well-calibrated model lies on the diagonal.
    p = proba[:, class_idx]
    hit = (y_true == class_idx).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_id = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)

    mean_pred: list[float] = []
    emp_freq: list[float] = []
    counts: list[float] = []
    for b in range(n_bins):
        mask = bin_id == b
        if not mask.any():
            continue
        mean_pred.append(float(p[mask].mean()))
        emp_freq.append(float(hit[mask].mean()))
        counts.append(float(mask.sum()))
    return {"mean_predicted": mean_pred, "empirical": emp_freq, "count": counts}
