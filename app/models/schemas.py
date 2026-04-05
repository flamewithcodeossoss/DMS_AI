from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class InputType(str, Enum):
    TEXT = "text"
    PDF = "pdf"
    IMAGE = "image"


# ── Request Models ──────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=5000)
    session_id: str = Field(default="default")


# ── Clinical Insight Sub-models ─────────────────────────────

class PatientInfo(BaseModel):
    name: Optional[str] = None
    age: Optional[str] = None
    gender: Optional[str] = None
    id: Optional[str] = None
    date_of_report: Optional[str] = None
    additional: Optional[dict] = None


class Symptom(BaseModel):
    name: str
    severity: Optional[str] = None
    duration: Optional[str] = None
    notes: Optional[str] = None


class MedicalAssessment(BaseModel):
    diagnosis: Optional[str] = None
    findings: Optional[list[str]] = None
    lab_results: Optional[dict] = None
    imaging_results: Optional[str] = None
    medications: Optional[list[str]] = None
    notes: Optional[str] = None


class SuggestedNextStep(BaseModel):
    action: str
    priority: Optional[str] = None
    reason: Optional[str] = None


# ── Response Models ─────────────────────────────────────────

class ClinicalInsights(BaseModel):
    patient_info: PatientInfo
    symptoms: list[Symptom]
    medical_assessment: MedicalAssessment
    suggested_next_steps: list[SuggestedNextStep]
    human_summary: str


class AnalyzeResponse(BaseModel):
    success: bool = True
    input_type: InputType
    insights: ClinicalInsights
    chunks_indexed: int


class QueryResponse(BaseModel):
    success: bool = True
    answer: str
    sources_used: int


class HealthResponse(BaseModel):
    status: str = "ok"
    model: str
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
