"""
RAGAS evaluation endpoint — /api/v1/evaluation
"""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Any

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


class EvalResponse(BaseModel):
    run_at: str
    metrics: dict[str, float]
    num_questions: int
    per_question: list[dict[str, Any]]


@router.post("/ragas", response_model=EvalResponse)
async def run_evaluation() -> EvalResponse:
    """
    Run RAGAS evaluation over 10 ground-truth style queries.
    Returns faithfulness, answer_relevancy, context_precision, context_recall.
    """
    from app.evaluation.ragas_eval import run_ragas_evaluation
    report = await run_ragas_evaluation()
    d = report.to_dict()
    return EvalResponse(
        run_at=d["run_at"],
        metrics=d["metrics"],
        num_questions=d["num_questions"],
        per_question=d["per_question"],
    )
