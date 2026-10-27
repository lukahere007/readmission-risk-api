"""Static figures for the README (matplotlib, headless)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"gradient_boosting": "#2a78d6", "logistic_regression": "#eb6834"}
LABELS = {"gradient_boosting": "Gradient boosting", "logistic_regression": "Logistic regression"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
})


def _order(probs: dict) -> list[str]:
    return [k for k in ["gradient_boosting", "logistic_regression"] if k in probs]


def roc_pr(y: np.ndarray, probs: dict[str, np.ndarray], path: Path) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
    for k in _order(probs):
        fpr, tpr, _ = roc_curve(y, probs[k])
        a1.plot(fpr, tpr, color=COLORS[k], lw=2, label=LABELS[k])
        prec, rec, _ = precision_recall_curve(y, probs[k])
        a2.plot(rec, prec, color=COLORS[k], lw=2, label=LABELS[k])
    a1.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1, label="Chance")
    a2.axhline(y.mean(), ls="--", color=MUTED, lw=1, label=f"Prevalence ({y.mean():.1%})")
    a1.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (held-out patients)")
    a2.set(xlabel="Recall", ylabel="Precision", title="Precision-recall (held-out patients)",
           ylim=(0, 1))
    a1.legend(loc="lower right")
    a2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def calibration(y: np.ndarray, probs: dict[str, np.ndarray], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    lim = 0
    for k in _order(probs):
        obs, pred = calibration_curve(y, probs[k], n_bins=10, strategy="quantile")
        ax.plot(pred, obs, marker="o", ms=6, lw=2, color=COLORS[k], label=LABELS[k],
                mec=SURFACE, mew=1.5)
        lim = max(lim, pred.max(), obs.max())
    lim = min(1, lim * 1.1)
    ax.plot([0, lim], [0, lim], ls="--", color=MUTED, lw=1, label="Perfect calibration")
    ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="Mean predicted risk (decile)",
           ylabel="Observed 30-day readmission rate", title="Calibration (held-out patients)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def subgroups(table: pd.DataFrame, path: Path, overall_auc: float | None = None,
              model_label: str = "") -> None:
    t = table.dropna(subset=["roc_auc"]).copy()
    t["label"] = t["attribute"].map({"age_group": "Age", "gender": "Sex",
                                     "race_recorded": "Race"}) + ": " + t["group"] + \
        "  (n=" + t["n"].map("{:,}".format) + ")"
    t = t.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(t) + 1.2))
    yy = np.arange(len(t))
    ax.hlines(yy, t["auc_ci_low"], t["auc_ci_high"], color="#86b6ef", lw=3)
    ax.plot(t["roc_auc"], yy, "o", color="#2a78d6", ms=8, mec=SURFACE, mew=1.5)
    if overall_auc is not None:
        ax.axvline(overall_auc, color=MUTED, ls="--", lw=1)
        ax.text(overall_auc, -0.9, f" overall {overall_auc:.3f}", color=MUTED, fontsize=9, va="center")
    ax.set_yticks(yy, t["label"])
    ax.set(xlabel="ROC-AUC with 95% bootstrap CI", title=f"ROC-AUC by subgroup, held-out patients\n({model_label}; groups with n < 200 omitted)")
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-1.4, len(t) - 0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
