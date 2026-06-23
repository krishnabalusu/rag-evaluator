"""
RAGAS Evaluator (RAGAS 0.4.x compatible with Groq).
Scores RAG pipeline on Faithfulness, AnswerRelevancy, ContextRecall.
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import EMBEDDING_MODEL, OPENAI_API_KEY

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY


def _build_ragas_llm():
    ragas_llm = LangchainLLMWrapper(ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        temperature=0.0,
    ))
    ragas_emb = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    )
    # Inject LLM into old-style metrics
    faithfulness.llm = ragas_llm
    answer_relevancy.llm = ragas_llm
    answer_relevancy.embeddings = ragas_emb
    context_recall.llm = ragas_llm
    return ragas_llm, ragas_emb


def run_evaluation(
    pipeline,
    testset: list,
    run_name: str = "run_1",
    config: dict = None,
) -> dict:
    """
    Run RAGAS evaluation on a RAG pipeline.

    Args:
        pipeline:  LangChainRAGPipeline instance (already built)
        testset:   List of QA pairs from testset_generator
        run_name:  Name for this eval run
        config:    Pipeline config dict

    Returns:
        Dict with faithfulness, answer_relevancy, context_recall scores
    """
    print(f"\n{'='*60}")
    print(f"  Running RAGAS Evaluation: {run_name}")
    print(f"  Questions: {len(testset)}")
    if config:
        print(f"  Config: {config}")
    print(f"{'='*60}\n")

    questions, answers, contexts, ground_truths = [], [], [], []

    for i, item in enumerate(testset, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]
        print(f"  [{i}/{len(testset)}] {question[:70]}...")
        result = pipeline.query(question)
        questions.append(question)
        answers.append(result["answer"])
        contexts.append(result["contexts"])
        ground_truths.append(ground_truth)

    # Build RAGAS dataset with correct column names
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    # Configure RAGAS metrics with Groq LLM
    ragas_llm, ragas_emb = _build_ragas_llm()

    print(f"\n  Scoring with RAGAS metrics (faithfulness, answer_relevancy, context_recall)...")
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall],
        raise_exceptions=False,
    )

    import math

    def _score(key):
        val = result[key]
        if isinstance(val, (int, float)):
            return round(float(val), 4) if not math.isnan(float(val)) else 0.0
        # list of per-sample scores — average, skip NaN
        nums = [float(v) for v in val if v is not None and not math.isnan(float(v))]
        return round(sum(nums) / len(nums), 4) if nums else 0.0

    f  = _score("faithfulness")
    ar = _score("answer_relevancy")
    cr = _score("context_recall")

    scores = {
        "run_name": run_name,
        "timestamp": datetime.now().isoformat(),
        "config": config or {},
        "total_questions": len(testset),
        "faithfulness": f,
        "answer_relevancy": ar,
        "context_recall": cr,
        "overall": round((f + ar + cr) / 3, 4),
    }

    path = RESULTS_DIR / f"{run_name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  RESULTS: {run_name}")
    print(f"{'='*60}")
    print(f"  Faithfulness:     {scores['faithfulness']:.4f}")
    print(f"  Answer Relevancy: {scores['answer_relevancy']:.4f}")
    print(f"  Context Recall:   {scores['context_recall']:.4f}")
    print(f"  Overall Score:    {scores['overall']:.4f}")
    print(f"{'='*60}")
    print(f"  Saved → {path}\n")

    return scores
