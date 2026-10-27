"""Loading, cohort definition and feature engineering for the UCI diabetes readmission data.

Everything here is a pure function of a DataFrame so it can be unit tested and reused
identically at training time and inside the served pipeline.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "diabetic_data.csv.gz"

TARGET = "readmitted_30d"
GROUP = "patient_nbr"

# discharge_disposition_id codes from IDs_mapping.csv that mean the patient died or went to
# hospice. These patients cannot (or are very unlikely to) be readmitted, so they are outside
# the population the model is meant for and are removed before training and rejected by the API.
#   11 Expired; 13 Hospice / home; 14 Hospice / medical facility;
#   19 Expired at home (Medicaid only, hospice); 20 Expired in a medical facility (hospice);
#   21 Expired, place unknown (hospice)
DEATH_OR_HOSPICE_DISPOSITIONS = frozenset({11, 13, 14, 19, 20, 21})

AGE_BANDS = [f"[{lo}-{lo + 10})" for lo in range(0, 100, 10)]

# Diabetes drugs kept as features: those with non-trivial variation in the data. The other
# 15 drug columns in the raw file are "No" for >99% of encounters.
MED_COLUMNS = [
    "metformin", "repaglinide", "glimepiride", "glipizide",
    "glyburide", "pioglitazone", "rosiglitazone", "insulin",
]

NUMERIC_RAW = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
]
CATEGORICAL_RAW = [
    "race", "gender", "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "medical_specialty", "max_glu_serum", "A1Cresult", "change", "diabetesMed", *MED_COLUMNS,
]
DIAG_COLUMNS = ["diag_1", "diag_2", "diag_3"]
# Columns the model (and therefore the API) needs as input.
INPUT_COLUMNS = ["age", *NUMERIC_RAW, *CATEGORICAL_RAW, *DIAG_COLUMNS]

# Columns produced by `engineer_features`, which is the first step of the model pipeline.
NUMERIC_FEATURES = [
    "age_years", *NUMERIC_RAW, "prior_visits", "n_meds_changed", "n_meds_active",
]
CATEGORICAL_FEATURES = [
    *[c for c in CATEGORICAL_RAW], "diag_1_group", "diag_2_group", "diag_3_group",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_raw(path: Path | str = RAW_PATH) -> pd.DataFrame:
    """Read the raw UCI file. '?' marks missing; the literal string 'None' is a real level
    (e.g. max_glu_serum 'None' = test not done) so pandas' default NA parsing is disabled."""
    return pd.read_csv(path, na_values=["?"], keep_default_na=False, low_memory=False)


def build_cohort(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the study population rules and create the binary target.

    * drop encounters discharged to death or hospice (see DEATH_OR_HOSPICE_DISPOSITIONS)
    * drop the 3 rows with gender 'Unknown/Invalid'
    * target = readmitted within 30 days ("<30"); ">30" and "NO" are negatives
    """
    df = raw.loc[~raw["discharge_disposition_id"].isin(DEATH_OR_HOSPICE_DISPOSITIONS)]
    df = df.loc[df["gender"] != "Unknown/Invalid"].copy()
    df[TARGET] = (df["readmitted"] == "<30").astype(int)
    return df.reset_index(drop=True)


def icd9_group(code: object) -> str:
    """Map an ICD-9 code string to the broad categories used by Strack et al. (2014), Table 2.

    Diabetes is 250.xx; V and E codes (supplementary classification) go to "Other".
    """
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return "Missing"
    s = str(code).strip()
    if not s or s == "?":
        return "Missing"
    if s[0] in "VvEe":
        return "Other"
    try:
        x = float(s)
    except ValueError:
        return "Other"
    if 250 <= x < 251:
        return "Diabetes"
    if 390 <= x <= 459 or int(x) == 785:
        return "Circulatory"
    if 460 <= x <= 519 or int(x) == 786:
        return "Respiratory"
    if 520 <= x <= 579 or int(x) == 787:
        return "Digestive"
    if 800 <= x <= 999:
        return "Injury"
    if 710 <= x <= 739:
        return "Musculoskeletal"
    if 580 <= x <= 629 or int(x) == 788:
        return "Genitourinary"
    if 140 <= x <= 239:
        return "Neoplasms"
    return "Other"


def age_band_to_years(band: object) -> float:
    """'[70-80)' -> 75.0 (band midpoint). Unknown -> NaN."""
    try:
        lo, hi = str(band).strip("[)").split("-")
        return (float(lo) + float(hi)) / 2
    except (ValueError, AttributeError):
        return np.nan


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw encounter columns (INPUT_COLUMNS) into model features (FEATURES).

    Stateless, so it is safe to run inside the sklearn pipeline at both fit and predict time.
    """
    out = pd.DataFrame(index=df.index)
    out["age_years"] = df["age"].map(age_band_to_years).astype(float)
    for c in NUMERIC_RAW:
        out[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    out["prior_visits"] = out[["number_outpatient", "number_emergency", "number_inpatient"]].sum(axis=1)
    meds = df[MED_COLUMNS].astype("string")
    out["n_meds_changed"] = meds.isin(["Up", "Down"]).sum(axis=1).astype(float)
    out["n_meds_active"] = meds.isin(["Up", "Down", "Steady"]).sum(axis=1).astype(float)
    for c in CATEGORICAL_RAW:
        out[c] = df[c].astype("string").fillna("Missing").astype(object)
    for c in DIAG_COLUMNS:
        out[f"{c}_group"] = df[c].map(icd9_group).astype(object)
    return out[FEATURES]
