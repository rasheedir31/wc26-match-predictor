# Monte Carlo simulation: structure, conservation invariants, and determinism.

from __future__ import annotations

import numpy as np
import pytest

from wc26.features.featurizer import MatchupFeaturizer
from wc26.ingest.fixtures import load_wc26_groups
from wc26.models.elo import EloModel
from wc26.simulate import bracket
from wc26.simulate.montecarlo import TournamentSimulator


def test_seeding_order_examples() -> None:
    assert bracket.seeding_order(4) == [1, 4, 2, 3]
    assert bracket.seeding_order(8) == [1, 8, 4, 5, 2, 7, 3, 6]
    order = bracket.seeding_order(32)
    assert sorted(order) == list(range(1, 33))  # a valid permutation of 32 seeds


@pytest.fixture
def simulator(feature_frame) -> TournamentSimulator:
    model = EloModel().fit(feature_frame)
    featurizer = MatchupFeaturizer.fitted_on(feature_frame)
    return TournamentSimulator.from_model(model, featurizer, load_wc26_groups())


def test_simulation_conservation_invariants(simulator) -> None:
    out = simulator.run(n_runs=200, seed=1)
    assert len(out) == 48  # all teams reported

    # Exactly N teams occupy each round per tournament, so per-team probabilities sum
    # to the round's slot count.
    expected = {
        bracket.STAGE_R32: 32,
        bracket.STAGE_R16: 16,
        bracket.STAGE_QF: 8,
        bracket.STAGE_SF: 4,
        bracket.STAGE_FINAL: 2,
        bracket.STAGE_CHAMPION: 1,
    }
    for stage, slots in expected.items():
        assert np.isclose(out[stage].sum(), slots, atol=1e-9)


def test_progression_is_monotone(simulator) -> None:
    out = simulator.run(n_runs=200, seed=1)
    # A team cannot reach a later stage more often than an earlier one.
    for a, b in zip(bracket.KNOCKOUT_STAGES, bracket.KNOCKOUT_STAGES[1:], strict=False):
        assert (out[a] >= out[b] - 1e-12).all()


def test_simulation_is_deterministic(simulator) -> None:
    a = simulator.run(n_runs=100, seed=7)
    b = simulator.run(n_runs=100, seed=7)
    assert a.equals(b)
