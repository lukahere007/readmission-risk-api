import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from readmission.api import app, get_bundle
from readmission.data import INPUT_COLUMNS


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_model_loads_and_outputs_probabilities(encounter):
    bundle = get_bundle()
    frame = pd.DataFrame([encounter] * 5, columns=INPUT_COLUMNS)
    p = bundle.model.predict_proba(frame)
    assert p.shape == (5, 2)
    assert np.all((p >= 0) & (p <= 1)) and np.allclose(p.sum(axis=1), 1)


def test_metadata_has_required_fields(client):
    info = client.get("/model-info").json()
    for key in ["trained_at_utc", "data", "metrics_test", "model_features", "sklearn_version",
                "operating_point"]:
        assert key in info
    assert len(info["data"]["sha256"]) == 64


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_single(client, encounter):
    r = client.post("/predict", json=encounter)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0 <= body["risk"] <= 1
    assert body["flag_for_follow_up"] == (body["risk"] >= body["threshold"])


def test_predict_batch_matches_single(client, encounter):
    other = {**encounter, "number_inpatient": 6, "age": "[80-90)"}
    r = client.post("/predict/batch", json={"encounters": [encounter, other]})
    assert r.status_code == 200, r.text
    preds = r.json()["predictions"]
    assert len(preds) == 2
    single = client.post("/predict", json=encounter).json()
    assert preds[0]["risk"] == single["risk"]


def test_more_prior_admissions_raises_risk(client, encounter):
    low = client.post("/predict", json={**encounter, "number_inpatient": 0}).json()["risk"]
    high = client.post("/predict", json={**encounter, "number_inpatient": 8}).json()["risk"]
    assert high > low


@pytest.mark.parametrize("patch", [
    {"age": "75"},                          # not a valid band
    {"gender": "X"},
    {"time_in_hospital": 0},                # below range
    {"num_medications": -1},
    {"discharge_disposition_id": 11},       # expired: out of scope
    {"discharge_disposition_id": 14},       # hospice: out of scope
    {"diag_1": "heart failure"},            # not an ICD-9 code
    {"insulin": "Maybe"},
    {"unexpected_field": 1},                # extra fields forbidden
])
def test_invalid_input_returns_422(client, encounter, patch):
    r = client.post("/predict", json={**encounter, **patch})
    assert r.status_code == 422


def test_missing_required_field_returns_422(client, encounter):
    bad = dict(encounter)
    bad.pop("age")
    assert client.post("/predict", json=bad).status_code == 422


def test_empty_batch_returns_422(client):
    assert client.post("/predict/batch", json={"encounters": []}).status_code == 422
