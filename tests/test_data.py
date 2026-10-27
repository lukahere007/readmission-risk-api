import numpy as np
import pandas as pd
import pytest

from readmission.data import (FEATURES, INPUT_COLUMNS, TARGET, age_band_to_years, build_cohort,
                              engineer_features, icd9_group)


@pytest.mark.parametrize("code,expected", [
    ("250.83", "Diabetes"), ("250", "Diabetes"), ("428", "Circulatory"), ("785", "Circulatory"),
    ("486", "Respiratory"), ("786", "Respiratory"), ("578", "Digestive"), ("996", "Injury"),
    ("715", "Musculoskeletal"), ("599", "Genitourinary"), ("162", "Neoplasms"),
    ("V57", "Other"), ("E909", "Other"), ("276", "Other"), (None, "Missing"),
    (np.nan, "Missing"), ("?", "Missing"), ("", "Missing"),
])
def test_icd9_group(code, expected):
    assert icd9_group(code) == expected


def test_age_band_midpoint():
    assert age_band_to_years("[70-80)") == 75
    assert np.isnan(age_band_to_years("unknown"))


def _raw(**overrides) -> pd.DataFrame:
    row = {"encounter_id": 1, "patient_nbr": 10, "race": "Caucasian", "gender": "Female",
           "age": "[60-70)", "admission_type_id": 1, "discharge_disposition_id": 1,
           "admission_source_id": 7, "time_in_hospital": 3, "medical_specialty": np.nan,
           "num_lab_procedures": 40, "num_procedures": 1, "num_medications": 12,
           "number_outpatient": 1, "number_emergency": 2, "number_inpatient": 3,
           "diag_1": "428", "diag_2": "250.01", "diag_3": "V45", "number_diagnoses": 9,
           "max_glu_serum": "None", "A1Cresult": ">8", "change": "Ch", "diabetesMed": "Yes",
           "metformin": "Steady", "repaglinide": "No", "glimepiride": "No", "glipizide": "No",
           "glyburide": "No", "pioglitazone": "No", "rosiglitazone": "No", "insulin": "Up",
           "readmitted": "<30"}
    row.update(overrides)
    return pd.DataFrame([row])


def test_build_cohort_excludes_death_hospice_and_sets_target():
    raw = pd.concat([
        _raw(),                                            # kept, positive
        _raw(readmitted=">30"),                            # kept, negative (>30 is not <30)
        _raw(readmitted="NO"),                             # kept, negative
        _raw(discharge_disposition_id=11),                 # expired
        _raw(discharge_disposition_id=13),                 # hospice / home
        _raw(discharge_disposition_id=21),                 # expired, hospice
        _raw(gender="Unknown/Invalid"),
    ], ignore_index=True)
    cohort = build_cohort(raw)
    assert len(cohort) == 3
    assert cohort[TARGET].tolist() == [1, 0, 0]


def test_engineer_features_columns_and_values():
    f = engineer_features(_raw()[INPUT_COLUMNS])
    assert list(f.columns) == FEATURES
    r = f.iloc[0]
    assert r.age_years == 65
    assert r.prior_visits == 6
    assert r.n_meds_changed == 1 and r.n_meds_active == 2
    assert r.medical_specialty == "Missing"
    assert (r.diag_1_group, r.diag_2_group, r.diag_3_group) == ("Circulatory", "Diabetes", "Other")
    assert r.admission_type_id == "1"
