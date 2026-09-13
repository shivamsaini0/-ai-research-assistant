"""
Evaluation API Endpoint
=========================
Exposes the RAGAS evaluation pipeline via REST.
Supports two modes:
  1. Custom  — user provides their own Q&A pairs + contexts
  2. Auto    — system runs a quick self-test using stored documents
"""
import logging
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional

from app.core.evaluation import evaluate_with_ragas
from app.core.rag_chain import run_rag_query
from app.core.retriever import hybrid_retrieve
from app.core.embeddings import embed_query
from app.core.auth import get_current_user

router = APIRouter()
logger = logging.getLogger(__name__)


# ─── Request / Response Models ────────────────────────────────────────────────

class EvalSample(BaseModel):
    question: str
    ground_truth: Optional[str] = None


class EvalRequest(BaseModel):
    """
    Run evaluation on a list of question-answer pairs.
    If ground_truth is provided, context recall is also computed.
    """
    samples: list[EvalSample]
    session_id: str = "eval_session"
    document_ids: Optional[list[str]] = None


class QuickEvalRequest(BaseModel):
    """Auto-evaluate using built-in sample questions."""
    document_ids: Optional[list[str]] = None


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/evaluate")
async def run_evaluation(request: EvalRequest, _current_user: dict = Depends(get_current_user)):
    """
    Evaluate the RAG pipeline on provided question samples.

    For each question:
    1. Runs hybrid retrieval to get contexts
    2. Generates an answer with Gemini
    3. Scores using RAGAS metrics

    Returns aggregate scores and per-question breakdown.
    """
    if not request.samples:
        raise HTTPException(status_code=400, detail="Provide at least 1 sample.")
    if len(request.samples) > 20:
        raise HTTPException(status_code=400, detail="Max 20 samples per evaluation run.")

    questions, answers, contexts, ground_truths = [], [], [], []

    for sample in request.samples:
        # Retrieve context for this question
        chunks = hybrid_retrieve(
            sample.question,
            n_results=4,
            doc_ids=request.document_ids,
        )

        if not chunks:
            # Skip questions with no context (no docs uploaded)
            continue

        # Generate answer through the RAG chain
        result = run_rag_query(
            session_id=f"{request.session_id}_{len(questions)}",
            query=sample.question,
            doc_ids=request.document_ids,
        )

        questions.append(sample.question)
        answers.append(result["answer"])
        contexts.append([c["content"] for c in chunks])

        if sample.ground_truth:
            ground_truths.append(sample.ground_truth)

    if not questions:
        raise HTTPException(
            status_code=422,
            detail="No documents found. Upload documents before running evaluation."
        )

    result = evaluate_with_ragas(
        questions=questions,
        answers=answers,
        contexts=contexts,
        ground_truths=ground_truths if ground_truths else None,
    )

    return result


@router.post("/evaluate/quick")
async def quick_evaluation(request: QuickEvalRequest, _current_user: dict = Depends(get_current_user)):
    """
    Run a quick self-evaluation using 5 generic research questions.
    Useful for testing the pipeline without preparing custom Q&A pairs.
    """
    sample_questions = [
        EvalSample(question="What is the main topic of the uploaded documents?"),
        EvalSample(question="What methodology or approach is described?"),
        EvalSample(question="What are the key findings or conclusions?"),
        EvalSample(question="What limitations are mentioned in the documents?"),
        EvalSample(question="Who are the authors or contributors mentioned?"),
    ]

    return await run_evaluation(EvalRequest(
        samples=sample_questions,
        session_id="quick_eval",
        document_ids=request.document_ids,
    ))
