import math

import numpy as np
import pandas as pd

from readmission.monitoring import (categorical_shares, numeric_edges, numeric_shares,
                                    psi_from_shares)


def test_psi_identical_is_zero():
    assert psi_from_shares([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]) == 0


def test_psi_known_value():
    # two bins, 50/50 -> 25/75: (0.25-0.5)ln(0.5) + (0.75-0.5)ln(1.5)
    expected = -0.25 * math.log(0.5) + 0.25 * math.log(1.5)
    assert math.isclose(psi_from_shares([0.5, 0.5], [0.25, 0.75]), expected, rel_tol=1e-9)
    assert math.isclose(expected, 0.2747, abs_tol=1e-4)


def test_psi_symmetric_and_handles_empty_bins():
    a, b = [0.5, 0.5, 0.0], [0.4, 0.4, 0.2]
    assert math.isclose(psi_from_shares(a, b), psi_from_shares(b, a), rel_tol=1e-9)
    assert np.isfinite(psi_from_shares(a, b))


def test_numeric_shares_quantile_bins_sum_to_one_and_track_missing():
    ref = pd.Series(np.arange(100, dtype=float))
    edges = numeric_edges(ref, 4)
    shares = numeric_shares(ref, edges)
    assert np.allclose(shares[:-1], 0.25) and shares[-1] == 0
    s2 = numeric_shares(pd.Series([1.0, np.nan]), edges)
    assert math.isclose(s2.sum(), 1) and s2[-1] == 0.5


def test_categorical_shares_unseen_level_goes_to_other_bucket():
    s = categorical_shares(pd.Series(["a", "b", "z"]), ["a", "b"])
    assert np.allclose(s, [1 / 3, 1 / 3, 1 / 3])
