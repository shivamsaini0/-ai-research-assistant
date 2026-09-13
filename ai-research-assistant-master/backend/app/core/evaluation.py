"""
Evaluation Module — RAGAS-based RAG Pipeline Assessment
=========================================================
Measures the quality of the RAG pipeline using 4 key metrics:

🧠 AI CONCEPT — RAG Evaluation with RAGAS:

    Traditional NLP metrics (BLEU, ROUGE) compare word overlap.
    They don't measure whether an answer is *factually correct*
    or *grounded in the retrieved documents*.

    RAGAS (RAG Assessment) solves this with LLM-as-judge metrics:

    1. FAITHFULNESS (0–1):
       "Is the answer supported by the retrieved context?"
       Measures hallucination — a faithful answer uses only
       facts from the retrieved chunks, not the LLM's memory.

    2. ANSWER RELEVANCY (0–1):
       "Does the answer actually address the question?"
       Low relevancy = verbose/off-topic responses.

    3. CONTEXT PRECISION (0–1):
       "Are the retrieved chunks actually useful for answering?"
       Measures retrieval quality — are we fetching relevant chunks?

    4. CONTEXT RECALL (0–1):
       "Did we retrieve all the info needed to answer?"
       Requires a ground-truth answer to compare against.

    An ideal RAG system scores high on all four.
    In practice, there's a tension between precision and recall.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def evaluate_with_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: Optional[list[str]] = None,
) -> dict:
    """
    Run RAGAS evaluation on a batch of QA pairs.

    Args:
        questions:     List of user questions
        answers:       List of LLM-generated answers
        contexts:      List of retrieved chunk lists (one per question)
        ground_truths: Optional list of reference answers (for recall)

    Returns:
        dict with metric scores (0.0–1.0) and per-question breakdown
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
        )

        # Build RAGAS dataset
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }

        metrics = [faithfulness, answer_relevancy, context_precision]

        # context_recall requires ground truth
        if ground_truths and len(ground_truths) == len(questions):
            from ragas.metrics import context_recall
            data["ground_truth"] = ground_truths
            metrics.append(context_recall)

        dataset = Dataset.from_dict(data)

        # Configure RAGAS to use Gemini
        _configure_ragas_llm()

        result = evaluate(dataset, metrics=metrics)

        scores = result.to_pandas().mean(numeric_only=True).to_dict()

        # Round for display
        rounded = {k: round(float(v), 4) for k, v in scores.items()}

        # Per-question breakdown
        df = result.to_pandas()
        per_question = []
        for i, row in df.iterrows():
            entry = {"question": questions[i], "answer_preview": answers[i][:120] + "..."}
            for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
                if metric in row:
                    entry[metric] = round(float(row[metric]), 4) if not __import__("math").isnan(row[metric]) else None
            per_question.append(entry)

        return {
            "status": "success",
            "aggregate": rounded,
            "per_question": per_question,
            "num_samples": len(questions),
        }

    except ImportError as e:
        logger.warning(f"RAGAS not available: {e}. Using fallback evaluation.")
        return _fallback_evaluation(questions, answers, contexts)
    except Exception as e:
        logger.error(f"RAGAS evaluation error: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "aggregate": {},
            "per_question": [],
            "num_samples": 0,
        }


def _configure_ragas_llm():
    """Configure RAGAS to use Gemini as its judge LLM."""
    try:
        from ragas.llms import LangchainLLMWrapper
        from langchain_google_genai import ChatGoogleGenerativeAI
        from ragas import evaluate
        from ragas.metrics import faithfulness

        from app.core.config import get_settings
        settings = get_settings()

        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=settings.gemini_api_key,
            temperature=0,
        )
        wrapped = LangchainLLMWrapper(llm)
        faithfulness.llm = wrapped
    except Exception:
        pass  # Will use default LLM if available


def _fallback_evaluation(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
) -> dict:
    """
    Lightweight fallback evaluation when RAGAS is unavailable.
    Uses heuristic scoring based on:
    - Answer length (relevancy proxy)
    - Context word overlap with answer (faithfulness proxy)
    - Citation presence
    """
    import re

    per_question = []
    all_faithfulness = []
    all_relevancy = []

    for q, a, ctx_list in zip(questions, answers, contexts):
        # Heuristic faithfulness: word overlap between answer and contexts
        ctx_words = set(" ".join(ctx_list).lower().split())
        ans_words = set(a.lower().split())
        overlap = len(ans_words & ctx_words) / max(len(ans_words), 1)
        faith = min(overlap * 2, 1.0)  # scale up

        # Heuristic relevancy: penalize very short or very generic answers
        ans_len = len(a.split())
        relevancy = min(ans_len / 50, 1.0) if ans_len < 50 else 1.0
        # Bonus for having citations
        if "[Source:" in a or "according to" in a.lower():
            relevancy = min(relevancy + 0.1, 1.0)

        # Heuristic context precision: check if context words appear in answer
        precision = min(len(ctx_words & ans_words) / max(len(ctx_words), 1) * 5, 1.0)

        all_faithfulness.append(faith)
        all_relevancy.append(relevancy)

        per_question.append({
            "question": q,
            "answer_preview": a[:120] + "...",
            "faithfulness": round(faith, 4),
            "answer_relevancy": round(relevancy, 4),
            "context_precision": round(precision, 4),
            "context_recall": None,
        })

    return {
        "status": "success (heuristic fallback)",
        "aggregate": {
            "faithfulness": round(sum(all_faithfulness) / len(all_faithfulness), 4),
            "answer_relevancy": round(sum(all_relevancy) / len(all_relevancy), 4),
            "context_precision": round(
                sum(e["context_precision"] for e in per_question) / len(per_question), 4
            ),
        },
        "per_question": per_question,
        "num_samples": len(questions),
        "note": "Heuristic evaluation (RAGAS/langchain-google-genai not installed). Scores are approximations."
    }
