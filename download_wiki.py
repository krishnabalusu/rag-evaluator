"""
Downloads 15 Wikipedia articles on AI/ML topics.
Saves each as a .txt file in wiki_docs/ folder.
Run once: python download_wiki.py
"""

import wikipediaapi
from pathlib import Path

WIKI_DOCS_DIR = Path(__file__).parent / "wiki_docs"
WIKI_DOCS_DIR.mkdir(exist_ok=True)

TOPICS = [
    "Artificial intelligence",
    "Machine learning",
    "Deep learning",
    "Artificial neural network",
    "Natural language processing",
    "Large language model",
    "Retrieval-augmented generation",
    "Vector database",
    "Transformer (deep learning architecture)",
    "BERT (language model)",
    "Generative pre-trained transformer",
    "Computer vision",
    "Reinforcement learning",
    "Transfer learning",
    "Word embedding",
]


def download_articles():
    wiki = wikipediaapi.Wikipedia(
        language="en",
        user_agent="enterprise-rag-eval/1.0"
    )

    print(f"\nDownloading {len(TOPICS)} Wikipedia articles...\n")
    success, failed = 0, 0

    for topic in TOPICS:
        try:
            page = wiki.page(topic)
            if not page.exists():
                print(f"  [SKIP] {topic}: page not found")
                failed += 1
                continue

            filename = topic.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_") + ".txt"
            filepath = WIKI_DOCS_DIR / filename
            filepath.write_text(page.text, encoding="utf-8", errors="replace")
            print(f"  [OK] {topic} → {filename} ({len(page.text):,} chars)")
            success += 1
        except Exception as e:
            print(f"  [SKIP] {topic}: {e}")
            failed += 1

    print(f"\nDone: {success} downloaded, {failed} skipped")
    print(f"Saved to: {WIKI_DOCS_DIR}\n")


if __name__ == "__main__":
    download_articles()
