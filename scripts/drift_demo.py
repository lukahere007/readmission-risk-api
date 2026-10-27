"""Demonstrate the PSI monitor on three batches, scored against the training profile:

1. the held-out test patients (same population -> expect no drift)
2. the 20% of test encounters with the highest encounter_id (a proxy for "later"
   admissions; the dataset does not document that ids are chronological)
3. a deliberately perturbed copy of the test set (older patients, more medications,
   more prior admissions, A1c tested more often) -> expect drift flags

Writes reports/drift_demo.csv and data/demo/*.csv.gz batches usable with
`python -m readmission.monitor`.

    python scripts/drift_demo.py
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

from readmission.data import TARGET, AGE_BANDS, INPUT_COLUMNS, ROOT, build_cohort, engineer_features, load_raw
from readmission.monitoring import drift_report
from readmission.train import grouped_split

SEED = 7


def perturb(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    shift = rng.random(len(out)) < 0.5  # half the patients one age band older
    idx = out["age"].map(AGE_BANDS.index)
    out.loc[shift, "age"] = [AGE_BANDS[min(i + 1, 9)] for i in idx[shift]]
    out["num_medications"] = (out["num_medications"] * 1.3).round().astype(int)
    more = rng.random(len(out)) < 0.3
    out.loc[more, "number_inpatient"] += 1
    tested = (out["A1Cresult"] == "None") & (rng.random(len(out)) < 0.3)
    out.loc[tested, "A1Cresult"] = rng.choice([">8", ">7", "Norm"], tested.sum())
    return out


def main() -> None:
    profile = json.loads((ROOT / "models" / "reference_profile.json").read_text())
    df = build_cohort(load_raw())
    _, te = grouped_split(df)
    test = df.iloc[te]
    late = test.loc[test["encounter_id"] >= test["encounter_id"].quantile(0.8)]
    batches = {"test": test, "late_encounter_ids": late, "perturbed": perturb(test)}

    demo_dir = ROOT / "data" / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    cols = {}
    model = joblib.load(ROOT / "models" / "model.joblib")
    perf = {}
    for name, b in batches.items():
        r = drift_report(profile, engineer_features(b[INPUT_COLUMNS]))
        cols[name] = r.set_index("feature")["psi"]
        p = model.predict_proba(b[INPUT_COLUMNS])[:, 1]
        perf[name] = {"n": len(b), "prevalence": round(float(b[TARGET].mean()), 4),
                      "mean_predicted": round(float(p.mean()), 4)}
        if name != "perturbed":  # perturbed labels are not meaningful
            perf[name]["roc_auc"] = round(float(roc_auc_score(b[TARGET], p)), 4)
        print(f"\n== {name} ({len(b)} rows): top features by PSI")
        print(r.head(6).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    for name in ("late_encounter_ids", "perturbed"):
        b = batches[name][INPUT_COLUMNS].head(2000)
        b.to_csv(demo_dir / f"{name}_sample.csv.gz", index=False,
                 compression={"method": "gzip", "mtime": 0})
    out = pd.DataFrame(cols).sort_values("perturbed", ascending=False)
    out.to_csv(ROOT / "reports" / "drift_demo.csv", float_format="%.4f")
    (ROOT / "reports" / "drift_demo_performance.json").write_text(json.dumps(perf, indent=2) + "\n")
    print("\nperformance:", json.dumps(perf, indent=1))
    print("\nfeatures with PSI >= 0.25:", (out >= 0.25).sum().to_dict())
    print("features with PSI >= 0.10:", (out >= 0.10).sum().to_dict())


if __name__ == "__main__":
    main()
