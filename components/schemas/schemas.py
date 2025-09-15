from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Literal
from typing_extensions import Annotated

class ConvoExtract(BaseModel):
    engagement_rating : Literal["Low", "Medium", "High"]
    clarity_rating : Annotated[int, Field(ge=1, le=10)]
    resolution_rating : Annotated[int, Field(ge=1, le=10)]
    sentiment_rating: Literal["Negative", "Neutral", "Positive"]
    service_category: Optional[
        Literal[
            "Preventive Maintenance Services",
            "Car-buying Assistance",
            "Parts Replacement",
            "Diagnosis",
            "PMS",
        ]
    ]  
    location: Optional[str]
    schedule_date: Optional[str]
    schedule_time: Optional[str]
    car: Optional[str]
    contact_num: Optional[str]
    payment: Optional[str]
    inspection: Optional[str]
    quotation: Optional[str]
    model: str
    summary: str


class SchemaIntent(BaseModel):
    intent_rating: Literal["No Intent", "Low Intent", "Moderate Intent", "High Intent", "Hot Intent"]


class FieldSpec(BaseModel):
    name: str
    py_type: str
    description: str
    default: Optional[str]
    enum_values: List[str] = Field(default_factory=list)


class SchemaSpec(BaseModel):
    class_name: str
    fields: List[FieldSpec]


class ScoreItem(BaseModel):
    intent: str = Field(..., description="Intent intent name")
    score: Annotated[
        float, Field(ge=0.0, le=1.0)
    ] = Field(..., description="Confidence for this intent in [0,1]")
    model_config = {"extra": "forbid"}


class IntentEvaluation(BaseModel):
    intents: Annotated[List[str], Field(min_length=1)] = Field(..., description="All intent levels from the rubric, ordered low+high")
    scorecard: Annotated[List[ScoreItem], Field(min_length=1)] = Field(..., description="Per-intent confidence scores")
    top_intent: str = Field(..., description="intent with the highest confidence")
    top_confidence: Annotated[float, Field(ge=0.0, le=1.0)] = Field(..., description="Confidence for top_intent")
    rationale: str = Field(..., description="Short explanation (≤5 lines)")
    evidence: List[str] = Field(default_factory=list, description="Concrete items citing fields/timestamps/snippets")

    class Config:
        extra = "forbid"

    @model_validator(mode="after")
    def _enforce_consistency(self):
        if self.top_intent not in self.intents:
            raise ValueError("top_intent must appear in intents")

        scores = {item.intent: float(item.score) for item in self.scorecard}
        missing = [lbl for lbl in self.intents if lbl not in scores]
        if missing:
            raise ValueError(f"scorecard missing intents: {missing}")

        extras = [lbl for lbl in scores if lbl not in self.intents]
        if extras:
            raise ValueError(f"scorecard has unknown intents not present in intens: {extras}")

        if scores.get(self.top_intent) is None:
            raise ValueError("scorecard must include top_intent")
        if abs(scores[self.top_intent] - float(self.top_confidence)) > 1e-9:
            raise ValueError("top_confidence must equal the score assigned to top_intent")
        
        return self


class RubricIssues(BaseModel):
    section: str
    problem: str