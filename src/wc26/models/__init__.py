# Models, all exposing the same ``predict() -> P(home win / draw / away win)``
# contract (see ``base.py``):
#
# - ``elo.py``      - Elo rating system, implemented from scratch (baseline + feature).
# - ``poisson.py``  - Dixon-Coles Poisson, hand-written log-likelihood + MLE.
# - ``logistic.py`` - multinomial logistic regression (1X2).
# - ``gbm.py``      - XGBoost.
#
#
