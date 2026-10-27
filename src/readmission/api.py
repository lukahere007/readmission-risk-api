"""FastAPI service for the readmission model.

    uvicorn readmission.api:app --host 0.0.0.0 --port 8000

Set MODEL_DIR to load the artefact from somewhere other than ./models.
"""
from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from fastapi import FastAPI

from .data import INPUT_COLUMNS, ROOT
from .schemas import BatchRequest, BatchResponse, Encounter, Prediction, PredictionResponse

log = logging.getLogger("readmission.api")


class ModelBundle:
    def __init__(self, model_dir: Path):
        self.metadata = json.loads((model_dir / "metadata.json").read_text())
        self.model = joblib.load(model_dir / "model.joblib")
        self.threshold = float(self.metadata["operating_point"]["threshold"])
        self.version = self.metadata["model_version"]
        if self.metadata["sklearn_version"] != sklearn.__version__:
            log.warning("model trained with scikit-learn %s, running %s",
                        self.metadata["sklearn_version"], sklearn.__version__)

    def predict(self, encounters: list[Encounter]) -> list[Prediction]:
        frame = pd.DataFrame([e.model_dump() for e in encounters], columns=INPUT_COLUMNS)
        risks = self.model.predict_proba(frame)[:, 1]
        return [Prediction(risk=round(float(r), 4), flag_for_follow_up=bool(r >= self.threshold),
                           threshold=round(self.threshold, 4)) for r in risks]


@lru_cache(maxsize=1)
def get_bundle() -> ModelBundle:
    return ModelBundle(Path(os.environ.get("MODEL_DIR", ROOT / "models")))


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_bundle()  # load the model once at startup so a bad artefact fails fast
    yield


app = FastAPI(
    lifespan=lifespan,
    title="30-day readmission risk API",
    version="1.0.0",
    description="Scores diabetic inpatient encounters for 30-day readmission risk. "
                "Portfolio project trained on 1999-2008 US data; not for clinical use.",
)


@app.get("/health")
def health() -> dict:
    b = get_bundle()
    return {"status": "ok", "model_version": b.version}


@app.get("/model-info")
def model_info() -> dict:
    return get_bundle().metadata


@app.post("/predict", response_model=PredictionResponse)
def predict(encounter: Encounter) -> PredictionResponse:
    b = get_bundle()
    p = b.predict([encounter])[0]
    return PredictionResponse(**p.model_dump(), model_version=b.version)


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest) -> BatchResponse:
    b = get_bundle()
    return BatchResponse(model_version=b.version, predictions=b.predict(req.encounters))
