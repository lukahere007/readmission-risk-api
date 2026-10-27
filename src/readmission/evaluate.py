"""Metrics helpers: discrimination, calibration, operating point, subgroups."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def core_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    prev = float(np.mean(y))
    return {
        "n": int(len(y)),
        "prevalence": prev,
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        # Brier of always predicting the prevalence: the no-skill reference
        "brier_baseline": float(prev * (1 - prev)),
        "log_loss": float(log_loss(y, p)),
        "mean_predicted": float(np.mean(p)),
    }


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Quantile-binned ECE: weighted mean |observed rate - mean predicted| per bin."""
    bins = pd.qcut(p, n_bins, labels=False, duplicates="drop")
    df = pd.DataFrame({"y": y, "p": p, "b": bins})
    g = df.groupby("b").agg(obs=("y", "mean"), pred=("p", "mean"), n=("y", "size"))
    return float(np.sum(g.n * (g.obs - g.pred).abs()) / g.n.sum())


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    obs, pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"mean_predicted": pred, "observed_rate": obs})


def operating_point(y: np.ndarray, p: np.ndarray, top_frac: float = 0.10) -> dict:
    """Flag the top `top_frac` of encounters by predicted risk for follow-up."""
    threshold = float(np.quantile(p, 1 - top_frac))
    flag = p >= threshold
    tp = int(np.sum(flag & (y == 1)))
    prev = float(np.mean(y))
    precision = tp / max(int(flag.sum()), 1)
    return {
        "top_fraction": top_frac,
        "threshold": threshold,
        "n_flagged": int(flag.sum()),
        "n_positive": int(np.sum(y)),
        "true_positives": tp,
        "recall": tp / max(int(np.sum(y)), 1),
        "precision": precision,
        "lift": precision / prev if prev else float("nan"),
        "flagged_per_1000": 1000 * float(flag.mean()),
        "caught_per_1000": 1000 * tp / len(y),
    }


def bootstrap_auc_ci(y: np.ndarray, p: np.ndarray, n_boot: int = 300, seed: int = 0,
                     alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        stats.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def subgroup_table(df: pd.DataFrame, group_col: str, y_col: str, p_col: str,
                   threshold: float, min_n: int = 200) -> pd.DataFrame:
    """Per-subgroup discrimination and behaviour at a shared threshold."""
    rows = []
    for level, g in df.groupby(group_col, observed=True):
        y, p = g[y_col].to_numpy(), g[p_col].to_numpy()
        flag = p >= threshold
        row = {
            "attribute": group_col, "group": str(level), "n": len(g), "positives": int(y.sum()),
            "prevalence": float(y.mean()), "mean_predicted": float(p.mean()),
            "flag_rate": float(flag.mean()),
            "recall": float((flag & (y == 1)).sum() / max(y.sum(), 1)),
            "precision": float((flag & (y == 1)).sum() / max(flag.sum(), 1)),
            "roc_auc": np.nan, "auc_ci_low": np.nan, "auc_ci_high": np.nan,
        }
        if len(g) >= min_n and 0 < y.sum() < len(y):
            row["roc_auc"] = float(roc_auc_score(y, p))
            row["auc_ci_low"], row["auc_ci_high"] = bootstrap_auc_ci(y, p)
        rows.append(row)
    return pd.DataFrame(rows)
