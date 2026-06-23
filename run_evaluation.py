"""
Enterprise RAG Evaluation — Main Entry Point

Runs two evaluation rounds with different configs and compares results.

Usage:
    python run_evaluation.py                    # full eval (generates testset + 2 runs)
    python run_evaluation.py --testset-only     # only generate testset
    python run_evaluation.py --eval-only        # only run eval (reuse existing testset)
    python run_evaluation.py --compare          # only compare existing runs
"""

import sys
import os
import json
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
from evaluation.langchain_pipeline import LangChainRAGPipeline
from evaluation.testset_generator import generate_testset
from evaluation.evaluator import run_evaluation
from evaluation.report import compare_runs, print_all_runs

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
TESTSET_PATH = RESULTS_DIR / "testset.json"

# ── Config for Run 1 (baseline) ────────────────────────────────────────────
RUN1_CONFIG = {
    "run_name": "run_1_chunk512",
    "chunk_size": 512,
    "chunk_overlap": 64,
    "top_k": 5,
}

# ── Config for Run 2 (experiment — smaller chunks) ─────────────────────────
RUN2_CONFIG = {
    "run_name": "run_2_chunk256",
    "chunk_size": 256,
    "chunk_overlap": 32,
    "top_k": 5,
}

TEST_SIZE = 10  # number of synthetic questions (increase for more thorough eval)


def load_or_generate_testset(force_regenerate: bool = False) -> list:
    if TESTSET_PATH.exists() and not force_regenerate:
        print(f"\nLoading existing testset from {TESTSET_PATH}")
        with open(TESTSET_PATH) as f:
            testset = json.load(f)
        print(f"Loaded {len(testset)} test questions")
        return testset

    # Generate fresh testset using run1 pipeline docs
    pipeline = LangChainRAGPipeline(
        chunk_size=RUN1_CONFIG["chunk_size"],
        chunk_overlap=RUN1_CONFIG["chunk_overlap"],
    )
    chunks = pipeline.chunks or __import__(
        "evaluation.langchain_pipeline", fromlist=["load_wiki_documents"]
    ).load_wiki_documents(RUN1_CONFIG["chunk_size"], RUN1_CONFIG["chunk_overlap"])

    testset = generate_testset(chunks, test_size=TEST_SIZE, save=True)
    return testset


def run_full_evaluation():
    print("\n" + "=" * 65)
    print("  ENTERPRISE RAG EVALUATION FRAMEWORK")
    print("=" * 65)

    # ── Step 1: Build Run 1 pipeline ──────────────────────────────────────
    print("\n[PHASE 1] Building baseline RAG pipeline (chunk_size=512)...")
    pipeline1 = LangChainRAGPipeline(
        chunk_size=RUN1_CONFIG["chunk_size"],
        chunk_overlap=RUN1_CONFIG["chunk_overlap"],
        top_k=RUN1_CONFIG["top_k"],
    ).build(collection_name="eval_run1")

    # ── Step 2: Generate testset ──────────────────────────────────────────
    print("\n[PHASE 2] Generating synthetic testset with RAGAS...")
    if TESTSET_PATH.exists():
        print(f"Reusing existing testset: {TESTSET_PATH}")
        with open(TESTSET_PATH) as f:
            testset = json.load(f)
        print(f"Loaded {len(testset)} questions")
    else:
        testset = generate_testset(num_questions=TEST_SIZE, save=True)

    # ── Step 3: Evaluate Run 1 ────────────────────────────────────────────
    print("\n[PHASE 3] Evaluating baseline pipeline...")
    scores1 = run_evaluation(
        pipeline=pipeline1,
        testset=testset,
        run_name=RUN1_CONFIG["run_name"],
        config=RUN1_CONFIG,
    )

    # ── Step 4: Build Run 2 pipeline ──────────────────────────────────────
    print("\n[PHASE 4] Building experiment RAG pipeline (chunk_size=256)...")
    pipeline2 = LangChainRAGPipeline(
        chunk_size=RUN2_CONFIG["chunk_size"],
        chunk_overlap=RUN2_CONFIG["chunk_overlap"],
        top_k=RUN2_CONFIG["top_k"],
    ).build(collection_name="eval_run2")

    # ── Step 5: Evaluate Run 2 ────────────────────────────────────────────
    print("\n[PHASE 5] Evaluating experiment pipeline...")
    scores2 = run_evaluation(
        pipeline=pipeline2,
        testset=testset,
        run_name=RUN2_CONFIG["run_name"],
        config=RUN2_CONFIG,
    )

    # ── Step 6: Compare ───────────────────────────────────────────────────
    print("\n[PHASE 6] Generating comparison report...")
    compare_runs(RUN1_CONFIG["run_name"], RUN2_CONFIG["run_name"])
    print_all_runs()

    print("EVALUATION COMPLETE.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enterprise RAG Evaluation")
    parser.add_argument("--testset-only", action="store_true", help="Only generate testset")
    parser.add_argument("--eval-only", action="store_true", help="Only run eval (reuse testset)")
    parser.add_argument("--compare", action="store_true", help="Only compare existing runs")
    args = parser.parse_args()

    if args.compare:
        compare_runs(RUN1_CONFIG["run_name"], RUN2_CONFIG["run_name"])
        print_all_runs()
    elif args.testset_only:
        generate_testset(num_questions=TEST_SIZE, save=True)
    else:
        run_full_evaluation()
