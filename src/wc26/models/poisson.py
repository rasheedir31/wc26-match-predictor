# Dixon-Coles bivariate-Poisson model - implemented from scratch.
#
# This is the statistical centerpiece. We hand-write the log-likelihood and fit it by
# maximum likelihood (scipy is used only as the numerical optimiser; the model, its
# gradient, the low-score correction, and the time-decay weighting are all our own).
#
# Model
# -----
# Each team ``i`` has an attack strength ``a_i`` and a defence strength ``d_i``; there
# is one home-advantage term ``gamma``. For a match between home ``h`` and away ``a``
# the expected goals are log-linear::
#
#     lambda = exp(a_h - d_a + gamma)        # home goal rate
#     mu     = exp(a_a - d_h)                 # away goal rate         (gamma off at neutral venues)
#
# Goals are Poisson, but independent Poissons mis-price low scores (too few draws), so
# Dixon & Coles (1997) multiply the joint pmf by a correction ``tau`` on the four
# low-score cells, governed by a single dependence parameter ``rho``::
#
#     tau(0,0) = 1 - lambda*mu*rho      tau(0,1) = 1 + lambda*rho
#     tau(1,0) = 1 + mu*rho             tau(1,1) = 1 - rho
#     tau(x,y) = 1                      otherwise
#
# Recent matches matter more, so each match carries an exponential time-decay weight
# ``w = exp(-xi * age_in_days)`` with ``xi = ln 2 / half_life`` (a match one half-life
# old counts half as much).
#
# Log-likelihood (dropping the constant ``log x! + log y!`` factorial terms)::
#
#     LL = sum_m w_m [ log tau_m + x_m*log(lambda_m) - lambda_m + y_m*log(mu_m) - mu_m ]
#
# We minimise ``-LL`` plus a small ridge penalty ``reg*(sum a_i^2 + sum d_i^2)`` which
# both regularises sparsely-observed teams and removes the attack/defence level
# redundancy (shifting all ``a`` and all ``d`` by the same constant leaves every
# ``lambda``/``mu`` unchanged). An analytic gradient is supplied so the ~few-hundred
# -parameter fit converges quickly.

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from wc26 import schema
from wc26.config import settings
from wc26.models.base import MatchPredictor, normalize_proba

_EPS = 1e-12


def _decay_weights(dates: pd.Series, half_life_days: float) -> np.ndarray:
    # Exponential time-decay weights relative to the most recent match.
    ref = dates.max()
    age_days = (ref - dates).dt.days.to_numpy(dtype=float)
    xi = np.log(2.0) / half_life_days
    return np.exp(-xi * age_days)


def _tau(x: np.ndarray, y: np.ndarray, lam: np.ndarray, mu: np.ndarray, rho: float) -> np.ndarray:
    # Dixon-Coles low-score correction for observed score pairs (vectorised).
    out = np.ones_like(lam)
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    out[m00] = 1.0 - lam[m00] * mu[m00] * rho
    out[m01] = 1.0 + lam[m01] * rho
    out[m10] = 1.0 + mu[m10] * rho
    out[m11] = 1.0 - rho
    return out


class DixonColesModel(MatchPredictor):
    # Dixon-Coles bivariate Poisson, fit by weighted MLE.

    name = "dixon_coles"

    def __init__(
        self,
        half_life_days: float | None = None,
        ridge: float | None = None,
        max_goals: int | None = None,
    ) -> None:
        self.half_life_days = (
            settings.model.dc_time_decay_half_life_days
            if half_life_days is None
            else half_life_days
        )
        self.ridge = settings.model.dc_ridge if ridge is None else ridge
        self.max_goals = settings.model.dc_max_goals if max_goals is None else max_goals
        self.teams_: list[str] = []
        self.team_idx_: dict[str, int] = {}
        self.attack_: np.ndarray = np.empty(0)
        self.defense_: np.ndarray = np.empty(0)
        self.gamma_: float = 0.0
        self.rho_: float = 0.0

    # --- fitting ---------------------------------------------------------------

    def _unpack(self, params: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, float, float]:
        return params[:n], params[n : 2 * n], float(params[2 * n]), float(params[2 * n + 1])

    def fit(self, train: pd.DataFrame) -> DixonColesModel:
        teams = sorted(set(train[schema.COL_HOME_TEAM]) | set(train[schema.COL_AWAY_TEAM]))
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        h = train[schema.COL_HOME_TEAM].map(idx).to_numpy()
        a = train[schema.COL_AWAY_TEAM].map(idx).to_numpy()
        x = train[schema.COL_HOME_SCORE].to_numpy(dtype=float)
        y = train[schema.COL_AWAY_SCORE].to_numpy(dtype=float)
        home_term = 1.0 - train[schema.COL_NEUTRAL].to_numpy(dtype=float)  # gamma off if neutral
        w = _decay_weights(train[schema.COL_DATE], self.half_life_days)

        def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
            attack, defense, gamma, rho = self._unpack(params, n)
            lin_h = attack[h] - defense[a] + gamma * home_term
            lin_a = attack[a] - defense[h]
            lam = np.exp(lin_h)
            mu = np.exp(lin_a)
            tau = _tau(x, y, lam, mu, rho)
            tau_safe = np.clip(tau, _EPS, None)

            # Negative weighted log-likelihood (+ ridge).
            nll = np.sum(w * (lam - x * lin_h + mu - y * lin_a - np.log(tau_safe)))
            nll += self.ridge * (np.sum(attack**2) + np.sum(defense**2))

            # --- analytic gradient ---
            # d/dlin_h = w[(lam - x) - (lam/tau) dtau/dlam] ; symmetric for away.
            dtau_dlam = np.zeros_like(lam)
            dtau_dmu = np.zeros_like(mu)
            dtau_drho = np.zeros_like(lam)
            m00 = (x == 0) & (y == 0)
            m01 = (x == 0) & (y == 1)
            m10 = (x == 1) & (y == 0)
            m11 = (x == 1) & (y == 1)
            dtau_dlam[m00] = -mu[m00] * rho
            dtau_dmu[m00] = -lam[m00] * rho
            dtau_drho[m00] = -lam[m00] * mu[m00]
            dtau_dlam[m01] = rho
            dtau_drho[m01] = lam[m01]
            dtau_dmu[m10] = rho
            dtau_drho[m10] = mu[m10]
            dtau_drho[m11] = -1.0

            g_lin_h = w * ((lam - x) - (lam / tau_safe) * dtau_dlam)
            g_lin_a = w * ((mu - y) - (mu / tau_safe) * dtau_dmu)

            grad = np.zeros_like(params)
            np.add.at(grad, h, g_lin_h)  # attack[home]
            np.add.at(grad, a, g_lin_a)  # attack[away]
            np.add.at(grad, n + a, -g_lin_h)  # defense[away]
            np.add.at(grad, n + h, -g_lin_a)  # defense[home]
            grad[:n] += 2.0 * self.ridge * attack
            grad[n : 2 * n] += 2.0 * self.ridge * defense
            grad[2 * n] = np.sum(g_lin_h * home_term)  # gamma
            grad[2 * n + 1] = np.sum(-w * dtau_drho / tau_safe)  # rho
            return nll, grad

        x0 = np.zeros(2 * n + 2)
        x0[2 * n] = 0.25  # sensible initial home advantage
        bounds = [(None, None)] * (2 * n) + [(None, None), (-0.2, 0.2)]
        res = minimize(objective, x0, jac=True, method="L-BFGS-B", bounds=bounds)

        attack, defense, gamma, rho = self._unpack(res.x, n)
        # Mean-centre attack/defence for interpretability (prediction is invariant).
        self.teams_ = teams
        self.team_idx_ = idx
        self.attack_ = attack - attack.mean()
        self.defense_ = defense - defense.mean()
        self.gamma_ = gamma
        self.rho_ = rho
        return self

    # --- prediction ------------------------------------------------------------

    def _rates(self, fixtures: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        # Home/away goal rates (lambda, mu) for each fixture; unknown teams -> average (0).

        def att(s: pd.Series) -> np.ndarray:
            return np.array(
                [self.attack_[self.team_idx_[t]] if t in self.team_idx_ else 0.0 for t in s]
            )

        def dfn(s: pd.Series) -> np.ndarray:
            return np.array(
                [self.defense_[self.team_idx_[t]] if t in self.team_idx_ else 0.0 for t in s]
            )

        home_term = 1.0 - fixtures[schema.COL_NEUTRAL].to_numpy(dtype=float)
        lam = np.exp(
            att(fixtures[schema.COL_HOME_TEAM])
            - dfn(fixtures[schema.COL_AWAY_TEAM])
            + self.gamma_ * home_term
        )
        mu = np.exp(att(fixtures[schema.COL_AWAY_TEAM]) - dfn(fixtures[schema.COL_HOME_TEAM]))
        return lam, mu

    def predict_proba(self, fixtures: pd.DataFrame) -> np.ndarray:
        lam, mu = self._rates(fixtures)
        g = np.arange(self.max_goals + 1)

        # Poisson pmf over 0..G for each fixture: shape (n, G+1).
        logpmf_h = g[None, :] * np.log(lam[:, None] + _EPS) - lam[:, None] - gammaln(g + 1)[None, :]
        logpmf_a = g[None, :] * np.log(mu[:, None] + _EPS) - mu[:, None] - gammaln(g + 1)[None, :]
        hx = np.exp(logpmf_h)
        ay = np.exp(logpmf_a)

        # Uncorrected 1X2 from independent Poissons, via cumulative sums:
        #   P(home win) = sum_x hx[x] * P(away < x);  draw = sum_x hx[x]*ay[x]
        cum_ay = np.cumsum(ay, axis=1)
        below = cum_ay - ay  # P(away < x)
        say = ay.sum(axis=1, keepdims=True)
        above = say - cum_ay  # P(away > x)
        p_home = np.sum(hx * below, axis=1)
        p_draw = np.sum(hx * ay, axis=1)
        p_away = np.sum(hx * above, axis=1)

        # Dixon-Coles correction: tau != 1 only on the four low-score cells, so add
        # the delta = base_cell * (tau - 1) to the region each cell belongs to.
        rho = self.rho_
        d00 = hx[:, 0] * ay[:, 0] * (-lam * mu * rho)  # (0,0) draw
        d11 = hx[:, 1] * ay[:, 1] * (-rho)  # (1,1) draw
        d01 = hx[:, 0] * ay[:, 1] * (lam * rho)  # (0,1) away win
        d10 = hx[:, 1] * ay[:, 0] * (mu * rho)  # (1,0) home win
        p_home = p_home + d10
        p_draw = p_draw + d00 + d11
        p_away = p_away + d01

        proba = np.column_stack([p_home, p_draw, p_away])
        proba = np.clip(proba, 0.0, None)  # guard tiny negatives from truncation/correction
        return normalize_proba(proba)
