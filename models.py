from pydantic import BaseModel


class ScoreRequest(BaseModel):
    reference: str
    candidate: str


class Score(BaseModel):
    precision: float
    recall: float
    f1: float


class RougeScore(BaseModel):
    rouge1: Score
    rouge2: Score
    rougeL: Score
