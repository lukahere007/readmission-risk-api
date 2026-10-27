"""Train, evaluate and package the readmission model. Fully reproducible (fixed seed).

    python -m readmission.train            # full run: CV, leakage check, figures, artefact
    python -m readmission.train --fast     # skip CV + leakage experiment (artefact only)

Outputs
  models/model.joblib, models/metadata.json, models/reference_profile.json
  reports/metrics.json, reports/*.csv, figures/*.png
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, ShuffleSplit

from . import plots
from .data import (FEATURES, GROUP, INPUT_COLUMNS, RAW_PATH, ROOT, TARGET, build_cohort,
                   engineer_features, load_raw)
from .evaluate import (bootstrap_auc_ci, calibration_table, core_metrics,
                       expected_calibration_error, operating_point, subgroup_table)
from .models import SEED, gbm_pipeline, logistic_pipeline
from .monitoring import build_profile

MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIG_DIR = ROOT / "figures"
TEST_SIZE = 0.2
TOP_FRACTION = 0.10


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def grouped_split(df: pd.DataFrame, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
    return next(gss.split(df, df[TARGET], groups=df[GROUP]))


def grouped_cv(make_model, X: pd.DataFrame, y: np.ndarray, groups: pd.Series,
               n_splits: int = 5) -> dict:
    aucs, aps = [], []
    for tr, va in GroupKFold(n_splits=n_splits).split(X, y, groups):
        m = make_model().fit(X.iloc[tr], y[tr])
        p = m.predict_proba(X.iloc[va])[:, 1]
        aucs.append(roc_auc_score(y[va], p))
        aps.append(core_metrics(y[va], p)["pr_auc"])
    return {"roc_auc_mean": float(np.mean(aucs)), "roc_auc_std": float(np.std(aucs)),
            "pr_auc_mean": float(np.mean(aps)), "pr_auc_std": float(np.std(aps))}


def leakage_experiment(df: pd.DataFrame, seeds=range(5)) -> pd.DataFrame:
    """Same model, same test size: random encounter-level split vs split by patient."""
    rows = []
    X, y = df[INPUT_COLUMNS], df[TARGET].to_numpy()
    for s in seeds:
        for name, splitter in [
            ("random (by encounter)", ShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=s)),
            ("grouped (by patient)", GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=s)),
        ]:
            tr, te = next(splitter.split(X, y, groups=df[GROUP] if "grouped" in name else None))
            shared = len(set(df[GROUP].iloc[tr]) & set(df[GROUP].iloc[te]))
            p = gbm_pipeline().fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
            rows.append({"seed": s, "split": name, "roc_auc": roc_auc_score(y[te], p),
                         "test_patients_also_in_train": shared})
    return pd.DataFrame(rows)


def add_subgroup_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    lo = out["age"].str.extract(r"\[(\d+)-")[0].astype(int)
    out["age_group"] = pd.cut(lo, [-1, 39, 59, 79, 200], labels=["<40", "40-59", "60-79", "80+"])
    out["race_recorded"] = out["race"].fillna("Missing")
    return out


def main(fast: bool = False) -> dict:
    for d in (MODEL_DIR, REPORT_DIR, FIG_DIR):
        d.mkdir(exist_ok=True)
    raw = load_raw()
    df = build_cohort(raw)
    cohort = {
        "raw_encounters": int(len(raw)),
        "raw_patients": int(raw[GROUP].nunique()),
        "excluded_death_or_hospice": int(raw["discharge_disposition_id"].isin([11, 13, 14, 19, 20, 21]).sum()),
        "excluded_unknown_gender": int((raw["gender"] == "Unknown/Invalid").sum()),
        "cohort_encounters": int(len(df)),
        "cohort_patients": int(df[GROUP].nunique()),
        "prevalence_readmit_30d": float(df[TARGET].mean()),
    }
    print("cohort:", json.dumps(cohort, indent=2))

    tr_idx, te_idx = grouped_split(df)
    train, test = df.iloc[tr_idx], df.iloc[te_idx]
    assert not set(train[GROUP]) & set(test[GROUP])
    X_tr, y_tr = train[INPUT_COLUMNS], train[TARGET].to_numpy()
    X_te, y_te = test[INPUT_COLUMNS], test[TARGET].to_numpy()
    print(f"train {len(train)} encounters / test {len(test)} encounters (disjoint patients)")

    results: dict = {"cohort": cohort, "split": {
        "method": "GroupShuffleSplit by patient_nbr", "test_size": TEST_SIZE, "seed": SEED,
        "train_encounters": int(len(train)), "test_encounters": int(len(test)),
        "train_patients": int(train[GROUP].nunique()), "test_patients": int(test[GROUP].nunique())}}

    candidates = {"logistic_regression": logistic_pipeline, "gradient_boosting": gbm_pipeline}
    if not fast:
        results["cv_train_groupkfold5"] = {}
        for name, make in candidates.items():
            results["cv_train_groupkfold5"][name] = cv = grouped_cv(make, X_tr, y_tr, train[GROUP])
            print(name, "CV:", cv)

    fitted, test_probs = {}, {}
    results["test"] = {}
    for name, make in candidates.items():
        fitted[name] = make().fit(X_tr, y_tr)
        p = fitted[name].predict_proba(X_te)[:, 1]
        test_probs[name] = p
        m = core_metrics(y_te, p)
        m["roc_auc_95ci"] = bootstrap_auc_ci(y_te, p)
        m["ece"] = expected_calibration_error(y_te, p)
        m["top10"] = operating_point(y_te, p, TOP_FRACTION)
        results["test"][name] = m
        print(name, "test:", json.dumps(m, indent=1))

    # Simple heuristic baseline: rank patients by prior-year inpatient admissions
    # (ties broken by emergency visits). No model, no fitting.
    heur = (X_te["number_inpatient"] + X_te["number_emergency"] / 100).to_numpy(float)
    results["test_heuristic_prior_admissions"] = {
        "roc_auc": float(roc_auc_score(y_te, heur)),
        "pr_auc": core_metrics(y_te, np.clip(heur / (heur.max() or 1), 0, 1))["pr_auc"],
        # remaining ties broken at random (seeded) so exactly 10% are flagged
        "top10": operating_point(
            y_te, heur + np.random.default_rng(SEED).random(len(heur)) * 1e-3, TOP_FRACTION),
    }
    print("heuristic:", results["test_heuristic_prior_admissions"])

    best = max(results["test"], key=lambda k: results["test"][k]["roc_auc"])
    model, p_best = fitted[best], test_probs[best]
    results["selected_model"] = best
    op = results["test"][best]["top10"]

    # Calibration table + subgroup analysis for the selected model
    calibration_table(y_te, p_best).to_csv(REPORT_DIR / "calibration_test.csv", index=False)
    sub_df = add_subgroup_columns(test).assign(p=p_best)
    subgroups = pd.concat([
        subgroup_table(sub_df, col, TARGET, "p", op["threshold"])
        for col in ["age_group", "gender", "race_recorded"]
    ])
    subgroups.to_csv(REPORT_DIR / "subgroups_test.csv", index=False, float_format="%.4f")

    if not fast:
        leak = leakage_experiment(df)
        leak.to_csv(REPORT_DIR / "leakage_random_vs_grouped.csv", index=False, float_format="%.4f")
        summary = leak.groupby("split").agg(roc_auc_mean=("roc_auc", "mean"),
                                            roc_auc_std=("roc_auc", "std"),
                                            shared_patients_mean=("test_patients_also_in_train", "mean"))
        results["leakage_experiment"] = summary.reset_index().to_dict(orient="records")
        print(summary)

    # Figures
    plots.roc_pr(y_te, test_probs, FIG_DIR / "roc_pr.png")
    plots.calibration(y_te, test_probs, FIG_DIR / "calibration.png")
    plots.subgroups(subgroups, FIG_DIR / "subgroup_auc.png", results["test"][best]["roc_auc"],
                    plots.LABELS[best].lower())

    # Artefacts
    joblib.dump(model, MODEL_DIR / "model.joblib", compress=3)
    profile = build_profile(engineer_features(X_tr))
    (MODEL_DIR / "reference_profile.json").write_text(json.dumps(profile, indent=1))
    t = results["test"][best]
    metadata = {
        "model_name": "readmission-30d",
        "model_version": "1.0.0",
        "estimator": best,
        "trained_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "data": {"file": str(RAW_PATH.relative_to(ROOT)), "sha256": file_sha256(RAW_PATH),
                 "source": "UCI ML Repository id 296 (CC BY 4.0)"},
        "training_rows": int(len(train)),
        "seed": SEED,
        "input_columns": INPUT_COLUMNS,
        "model_features": FEATURES,
        "excluded_discharge_dispositions": [11, 13, 14, 19, 20, 21],
        "metrics_test": {k: t[k] for k in ["n", "prevalence", "roc_auc", "pr_auc", "brier",
                                           "brier_baseline", "ece"]},
        "operating_point": {"rule": f"flag top {int(TOP_FRACTION * 100)}% of risk",
                            "threshold": op["threshold"], "recall": op["recall"],
                            "precision": op["precision"]},
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (REPORT_DIR / "metrics.json").write_text(json.dumps(results, indent=2))

    # Example payload for the README / smoke test: 3 real test-set encounters
    recs = json.loads(test[INPUT_COLUMNS].head(3).to_json(orient="records"))
    (ROOT / "examples").mkdir(exist_ok=True)
    (ROOT / "examples" / "single.json").write_text(json.dumps(recs[0], indent=2) + "\n")
    (ROOT / "examples" / "batch.json").write_text(json.dumps({"encounters": recs}, indent=2) + "\n")
    print("selected:", best, "| saved", MODEL_DIR / "model.joblib",
          f"({(MODEL_DIR / 'model.joblib').stat().st_size / 1e6:.2f} MB)")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    main(**vars(ap.parse_args()))
