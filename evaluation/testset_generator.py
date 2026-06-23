"""
Custom Testset Generator — no RAGAS dependency.
Extracts factual QA pairs from wiki documents using NLP patterns.
Saves to results/testset.json for reuse across eval runs.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import json
import random
from pathlib import Path
from ingestion.document_loader import load_directory
from preprocessing.text_cleaner import clean_documents

WIKI_DOCS_DIR = Path(__file__).parent.parent / "wiki_docs"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

random.seed(42)

# Question templates mapped to sentence patterns
TEMPLATES = [
    ("What is {subject}?",          r"^([A-Z][^.]{5,40}) (?:is|are|refers to|can be defined as) (.{30,200})\.$"),
    ("How does {subject} work?",    r"^([A-Z][^.]{5,40}) (?:works|operates|functions) by (.{30,200})\.$"),
    ("What are the {subject}?",     r"^(?:The )?([a-z][^.]{5,40}) (?:include|are|consist of) (.{30,200})\.$"),
    ("Why is {subject} important?", r"^([A-Z][^.]{5,40}) (?:is important|plays a key|enables|allows) (.{30,200})\.$"),
    ("What does {subject} use?",    r"^([A-Z][^.]{5,40}) uses? (.{30,200})\.$"),
]


def _extract_qa_from_text(text: str, source: str, max_per_doc: int = 5) -> list:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if 60 < len(s.strip()) < 400]

    qa_pairs = []
    for sentence in sentences:
        for template, pattern in TEMPLATES:
            match = re.match(pattern, sentence)
            if match:
                subject = match.group(1).strip().rstrip(",;:")
                question = template.format(subject=subject)
                qa_pairs.append({
                    "question": question,
                    "ground_truth": sentence,
                    "source": source,
                    "evolution_type": "simple",
                })
                break

        if len(qa_pairs) >= max_per_doc:
            break

    return qa_pairs


def generate_testset(num_questions: int = 20, save: bool = True) -> list:
    """
    Generate QA pairs from Wikipedia docs.

    Args:
        num_questions: Total number of QA pairs to generate
        save:          Save to results/testset.json

    Returns:
        List of {question, ground_truth, source, evolution_type}
    """
    print(f"\n{'='*60}")
    print(f"  Testset Generation")
    print(f"  Target: {num_questions} QA pairs from wiki docs")
    print(f"{'='*60}\n")

    docs = load_directory(WIKI_DOCS_DIR)
    docs = clean_documents(docs)

    all_qa = []
    per_doc = max(3, num_questions // len(docs) + 1)

    for doc in docs:
        source = doc.metadata.get("file_name", "unknown")
        qa = _extract_qa_from_text(doc.text, source, max_per_doc=per_doc)
        all_qa.extend(qa)
        print(f"  {source}: {len(qa)} QA pairs")

    # Shuffle and cap
    random.shuffle(all_qa)
    final = all_qa[:num_questions]

    print(f"\n  Total generated: {len(final)} QA pairs")

    if save:
        path = RESULTS_DIR / "testset.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(final, f, indent=2, ensure_ascii=False)
        print(f"  Saved → {path}")

    # Print sample
    if final:
        print(f"\n  Sample QA pair:")
        print(f"  Q: {final[0]['question']}")
        print(f"  A: {final[0]['ground_truth'][:100]}...")

    return final


if __name__ == "__main__":
    qa_pairs = generate_testset(num_questions=20)
    print(f"\nDone: {len(qa_pairs)} QA pairs generated")
