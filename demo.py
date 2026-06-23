import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.rag_pipeline import RAGPipeline


def print_result(result: dict):
    print("\n" + "=" * 60)
    print(f"Q: {result['question']}")
    print(f"\nA: {result['answer']}")
    print(f"\nSources:")
    for s in result["sources"]:
        print(f"   - {s['file']} | chunk {s['chunk']} | score {s['score']}")
    print(f"\nModel: {result.get('model', 'unknown')} | Tokens: {result.get('tokens_used', '?')}")
    print("=" * 60)


if __name__ == "__main__":
    pipeline = RAGPipeline()

    # Index documents (run once — comment out after first run)
    pipeline.index()

    # Ask questions
    questions = [
        "What is RAG and what are its benefits?",
        "How many days do employees have to submit expenses?",
        "What is the price of AcmeAnalytics Pro?",
        "What is the capital of France?",  # should say don't know
    ]

    for question in questions:
        result = pipeline.query(question)
        print_result(result)
