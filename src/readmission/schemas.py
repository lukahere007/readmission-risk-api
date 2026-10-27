"""Request / response schemas for the prediction API (pydantic v2)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .data import AGE_BANDS, DEATH_OR_HOSPICE_DISPOSITIONS

AgeBand = Literal[tuple(AGE_BANDS)]  # type: ignore[valid-type]
MedStatus = Literal["No", "Steady", "Up", "Down"]


class Encounter(BaseModel):
    """One inpatient diabetic encounter, using the field names and codes of the UCI dataset.

    Values are known at discharge; the model scores risk of readmission within 30 days.
    """

    model_config = ConfigDict(extra="forbid")

    race: Optional[Literal["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"]] = None
    gender: Literal["Female", "Male"]
    age: AgeBand  # type: ignore[valid-type]
    admission_type_id: int = Field(ge=1, le=8)
    discharge_disposition_id: int = Field(ge=1, le=30)
    admission_source_id: int = Field(ge=1, le=26)
    time_in_hospital: int = Field(ge=1, le=14, description="Length of stay in days (1-14)")
    medical_specialty: Optional[str] = Field(default=None, max_length=64)
    num_lab_procedures: int = Field(ge=0, le=200)
    num_procedures: int = Field(ge=0, le=20)
    num_medications: int = Field(ge=0, le=150)
    number_outpatient: int = Field(ge=0, le=100, description="Outpatient visits in prior year")
    number_emergency: int = Field(ge=0, le=100, description="Emergency visits in prior year")
    number_inpatient: int = Field(ge=0, le=100, description="Inpatient admissions in prior year")
    diag_1: Optional[str] = Field(default=None, max_length=8, description="Primary ICD-9 code")
    diag_2: Optional[str] = Field(default=None, max_length=8)
    diag_3: Optional[str] = Field(default=None, max_length=8)
    number_diagnoses: int = Field(ge=1, le=20)
    max_glu_serum: Literal["None", "Norm", ">200", ">300"] = "None"
    A1Cresult: Literal["None", "Norm", ">7", ">8"] = "None"
    change: Literal["No", "Ch"] = "No"
    diabetesMed: Literal["No", "Yes"] = "No"
    metformin: MedStatus = "No"
    repaglinide: MedStatus = "No"
    glimepiride: MedStatus = "No"
    glipizide: MedStatus = "No"
    glyburide: MedStatus = "No"
    pioglitazone: MedStatus = "No"
    rosiglitazone: MedStatus = "No"
    insulin: MedStatus = "No"

    @field_validator("discharge_disposition_id")
    @classmethod
    def in_scope_disposition(cls, v: int) -> int:
        if v in DEATH_OR_HOSPICE_DISPOSITIONS:
            raise ValueError(
                "encounters discharged to death or hospice are outside the model's intended use"
            )
        return v

    @field_validator("diag_1", "diag_2", "diag_3")
    @classmethod
    def icd9_like(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        s = v.strip()
        if not s:
            return None
        body = s[1:] if s[0] in "VvEe" else s
        try:
            float(body)
        except ValueError as exc:
            raise ValueError(f"'{v}' does not look like an ICD-9 code") from exc
        return s


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    encounters: list[Encounter] = Field(min_length=1, max_length=1000)


class Prediction(BaseModel):
    risk: float = Field(ge=0, le=1, description="Predicted probability of readmission <30 days")
    flag_for_follow_up: bool = Field(
        description="True if risk is at or above the top-10% threshold chosen on the test set")
    threshold: float


class PredictionResponse(Prediction):
    model_version: str


class BatchResponse(BaseModel):
    model_version: str
    predictions: list[Prediction]
