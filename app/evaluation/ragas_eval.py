"""
RAGAS evaluation for Bonobos fashion RAG pipeline.
Measures: faithfulness, answer_relevancy, context_precision, context_recall.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from app.config import get_settings
from app.rag.chain import BonobosRAGChain

settings = get_settings()

# Ground-truth evaluation dataset — 10 representative style queries
EVAL_DATASET = [
    {
        "question": "What slim fit chinos do you have for business casual?",
        "ground_truth": "Bonobos offers the Weekday Warrior Slim Chino in khaki and navy, made with 97% cotton and 3% elastane with 4-way stretch. Perfect for business casual occasions.",
    },
    {
        "question": "I need a complete outfit for a summer wedding as a guest",
        "ground_truth": "For a summer wedding guest, consider the Italian Stretch Suit in Charcoal with a white dress shirt and Oxford shoes. For a garden wedding, the Seersucker Stripe Shirt with white chinos and loafers works well.",
    },
    {
        "question": "What jackets work for cold weather weekends?",
        "ground_truth": "The Glacier Park Parka in Navy is ideal for cold weather weekends, featuring 650-fill down insulation and a water-resistant shell. Layer over a Merino Wool Crew Neck Sweater.",
    },
    {
        "question": "What fits do you offer for athletic builds?",
        "ground_truth": "Bonobos offers the Athletic Fit designed specifically for men with more muscular builds. Available in chinos and jeans, providing extra room through the seat and thighs while maintaining a tailored look.",
    },
    {
        "question": "Suggest a smart casual date night outfit in navy",
        "ground_truth": "For date night, pair the Italian Stretch Blazer in Navy with a white Oxford shirt and Athletic Stretch Jeans in indigo, finished with Chelsea boots. This blazer-over-jeans combination is polished yet relaxed.",
    },
    {
        "question": "What polo shirts do you have for summer?",
        "ground_truth": "The Riviera Polo in slim fit features moisture-wicking performance fabric and UPF 50+ sun protection. Available in white and multiple colors, it pairs well with khaki chinos, navy shorts, white sneakers, or loafers.",
    },
    {
        "question": "I need pants for a business formal meeting",
        "ground_truth": "The Dress Chino in Tailored fit and grey is ideal for business formal, made from a 70% wool blend with a higher rise. Also consider the Italian Stretch Suit trousers in charcoal for the most formal look.",
    },
    {
        "question": "What linen options do you have for summer?",
        "ground_truth": "The Linen Shirt in slim fit and white is perfect for summer — 100% linen, lightweight and breathable. Wear half-tucked with khaki chinos for smart casual or with shorts for a casual look.",
    },
    {
        "question": "What sweaters work for layering under a blazer?",
        "ground_truth": "The Merino Wool Crew Neck Sweater in grey is ideal for layering under a blazer. Made from 100% fine-gauge merino wool, it sits slim under jackets without bulking. Perfect for business casual and smart casual.",
    },
    {
        "question": "What navy blazers are available for smart casual?",
        "ground_truth": "The Italian Stretch Blazer in Navy and Slim fit features unstructured construction from stretch wool blend, making it comfortable and versatile. Wear over a white tee with chinos for smart casual or with dress trousers for business.",
    },
]


@dataclass
class RAGASReport:
    run_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    num_questions: int = 0
    per_question: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_at": self.run_at,
            "metrics": {
                "faithfulness": round(self.faithfulness, 4),
                "answer_relevancy": round(self.answer_relevancy, 4),
                "context_precision": round(self.context_precision, 4),
                "context_recall": round(self.context_recall, 4),
            },
            "num_questions": self.num_questions,
            "per_question": self.per_question,
        }


async def run_ragas_evaluation(
    questions: list[dict] | None = None,
    top_k: int = 8,
) -> RAGASReport:
    """
    Run RAGAS evaluation over the ground-truth dataset.
    Returns a RAGASReport with all four metrics.
    """
    eval_set = questions or EVAL_DATASET
    chain = BonobosRAGChain()

    queries = []
    answers = []
    contexts_list = []
    ground_truths = []

    for item in eval_set:
        result = await chain.style(query=item["question"], top_k=top_k)
        queries.append(item["question"])
        answers.append(result.answer)
        contexts_list.append([s["name"] + ": " + str(s.get("score", "")) for s in result.sources])
        ground_truths.append(item["ground_truth"])

    dataset = Dataset.from_dict(
        {
            "question": queries,
            "answer": answers,
            "contexts": contexts_list,
            "ground_truth": ground_truths,
        }
    )

    result = evaluate(
        dataset,
        metrics=[
            answer_faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    report = RAGASReport(
        faithfulness=float(result["faithfulness"]),
        answer_relevancy=float(result["answer_relevancy"]),
        context_precision=float(result["context_precision"]),
        context_recall=float(result["context_recall"]),
        num_questions=len(eval_set),
    )

    for i, q in enumerate(queries):
        report.per_question.append(
            {
                "question": q,
                "answer_preview": answers[i][:200],
                "sources_count": len(contexts_list[i]),
            }
        )

    return report
