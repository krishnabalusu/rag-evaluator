import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generation.generator import generate_answer


def test_answer_generated():
    context = """RAG stands for Retrieval Augmented Generation. 
    It retrieves relevant documents and passes them to an LLM to generate grounded answers.
    RAG reduces hallucinations by grounding answers in retrieved facts."""

    query = "What is RAG and what are its benefits?"
    result = generate_answer(query, context)

    assert "answer" in result
    assert len(result["answer"]) > 10
    assert "model" in result
    assert "tokens_used" in result
    print(f"  [PASS] Answer generated")
    print(f"         Model: {result['model']}")
    print(f"         Tokens used: {result['tokens_used']}")
    print(f"         Answer: {result['answer'][:200]}...")


def test_unknown_context_says_dont_know():
    context = "The sky is blue. Water is wet."
    query = "What is the capital of France?"
    result = generate_answer(query, context)

    answer_lower = result["answer"].lower()
    assert (
        "don't have" in answer_lower or
        "not" in answer_lower or
        "cannot" in answer_lower or
        "no information" in answer_lower or
        "context" in answer_lower
    )
    print(f"  [PASS] Model correctly refused to hallucinate")
    print(f"         Answer: {result['answer'][:150]}...")


if __name__ == "__main__":
    tests = [
        test_answer_generated,
        test_unknown_context_says_dont_know,
    ]

    print("\n" + "=" * 55)
    print("  Step 7: Generator Tests")
    print("=" * 55)

    passed, failed = 0, 0
    for test in tests:
        print(f"\n[RUN] {test.__name__}")
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            failed += 1

    print("\n" + "=" * 55)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 55 + "\n")