"""Population Stability Index (PSI) drift checks against a stored training profile.

PSI = sum_i (a_i - e_i) * ln(a_i / e_i), where e_i / a_i are the share of the reference
(training) and new batch in bin i. Common rule of thumb: < 0.1 stable, 0.1-0.25 moderate
shift, > 0.25 major shift.

Usage:
    python -m readmission.monitor path/to/new_batch.csv[.gz]
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import CATEGORICAL_FEATURES, NUMERIC_FEATURES

EPS = 1e-4
MODERATE, MAJOR = 0.1, 0.25


def psi_from_shares(expected: np.ndarray, actual: np.ndarray, eps: float = EPS) -> float:
    e = np.clip(np.asarray(expected, dtype=float), eps, None)
    a = np.clip(np.asarray(actual, dtype=float), eps, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def numeric_edges(values: pd.Series, n_bins: int = 10) -> list[float]:
    """Quantile bin edges from the reference data (duplicates removed; open-ended ends)."""
    qs = np.nanquantile(values.astype(float), np.linspace(0, 1, n_bins + 1)[1:-1])
    return sorted(set(float(q) for q in qs))


def numeric_shares(values: pd.Series, edges: list[float]) -> np.ndarray:
    v = values.astype(float).to_numpy()
    idx = np.searchsorted(edges, v[~np.isnan(v)], side="right")
    counts = np.bincount(idx, minlength=len(edges) + 1).astype(float)
    counts = np.append(counts, np.isnan(v).sum())  # last bin = missing
    return counts / max(len(v), 1)


def categorical_shares(values: pd.Series, levels: list[str]) -> np.ndarray:
    s = values.astype(str)
    counts = np.array([(s == lvl).sum() for lvl in levels] + [(~s.isin(levels)).sum()], float)
    return counts / max(len(s), 1)


def build_profile(features: pd.DataFrame, n_bins: int = 10) -> dict:
    """Reference distribution of every model feature (output of engineer_features)."""
    profile: dict = {"numeric": {}, "categorical": {}}
    for c in NUMERIC_FEATURES:
        edges = numeric_edges(features[c], n_bins)
        profile["numeric"][c] = {"edges": edges,
                                 "shares": numeric_shares(features[c], edges).tolist()}
    for c in CATEGORICAL_FEATURES:
        levels = sorted(features[c].astype(str).unique().tolist())
        profile["categorical"][c] = {"levels": levels,
                                     "shares": categorical_shares(features[c], levels).tolist()}
    return profile


def drift_report(profile: dict, features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c, ref in profile["numeric"].items():
        rows.append((c, "numeric", psi_from_shares(ref["shares"], numeric_shares(features[c], ref["edges"]))))
    for c, ref in profile["categorical"].items():
        rows.append((c, "categorical",
                     psi_from_shares(ref["shares"], categorical_shares(features[c], ref["levels"]))))
    out = pd.DataFrame(rows, columns=["feature", "type", "psi"])
    out["status"] = np.select([out.psi >= MAJOR, out.psi >= MODERATE], ["MAJOR", "moderate"], "ok")
    return out.sort_values("psi", ascending=False).reset_index(drop=True)
