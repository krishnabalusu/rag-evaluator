"""
Comparison Report Generator.
Compares two eval runs and shows improvement/degradation per metric.
This is the before/after analysis that builds the resume bullet.
"""

import json
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path(__file__).parent.parent / "results"


def compare_runs(run1_name: str, run2_name: str) -> dict:
    """
    Compare two evaluation runs side by side.
    Shows which metrics improved and which degraded.
    """
    path1 = RESULTS_DIR / f"{run1_name}.json"
    path2 = RESULTS_DIR / f"{run2_name}.json"

    if not path1.exists():
        raise FileNotFoundError(f"Run not found: {path1}")
    if not path2.exists():
        raise FileNotFoundError(f"Run not found: {path2}")

    with open(path1) as f:
        r1 = json.load(f)
    with open(path2) as f:
        r2 = json.load(f)

    metrics = ["faithfulness", "answer_relevancy", "context_recall", "overall"]

    print(f"\n{'='*65}")
    print(f"  BEFORE/AFTER COMPARISON")
    print(f"{'='*65}")
    print(f"  {'Metric':<22} {'Before':>10} {'After':>10} {'Change':>10}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10}")

    comparison = {}
    for metric in metrics:
        before = r1.get(metric, 0)
        after = r2.get(metric, 0)
        delta = after - before
        direction = "▲" if delta > 0 else "▼" if delta < 0 else "="
        flag = "✅" if delta > 0 else "❌" if delta < 0.01 else "➡"

        print(f"  {metric:<22} {before:>10.4f} {after:>10.4f} {direction}{abs(delta):>8.4f} {flag}")
        comparison[metric] = {"before": before, "after": after, "delta": round(delta, 4)}

    print(f"{'='*65}")
    print(f"\n  Config change:")
    print(f"  Before: {r1.get('config', {})}")
    print(f"  After:  {r2.get('config', {})}")

    # Generate resume bullet
    bottleneck = min(["faithfulness", "answer_relevancy", "context_recall"], key=lambda m: r1.get(m, 1))
    overall_delta = comparison["overall"]["delta"]
    winner = run1_name if overall_delta < 0 else run2_name
    winner_cfg = r1.get("config", {}) if overall_delta < 0 else r2.get("config", {})
    best_scores = r1 if overall_delta < 0 else r2

    chunk_key = "chunk_size"
    best_chunk = winner_cfg.get(chunk_key, "?")

    print(f"\n{'='*65}")
    print(f"  RESUME BULLET:")
    print(f"{'='*65}")
    print(f"""
  Built enterprise RAG evaluation framework using RAGAS on {r1['total_questions']} synthetic
  QA pairs from 21 Wikipedia AI/ML articles. Benchmarked two chunk-size
  configurations: chunk={r1['config'].get('chunk_size')} scored
  Faithfulness={r1['faithfulness']:.2f}, AnswerRelevancy={r1['answer_relevancy']:.2f},
  ContextRecall={r1['context_recall']:.2f} (overall {r1['overall']:.2f}); chunk={r2['config'].get('chunk_size')}
  scored overall {r2['overall']:.2f}. Identified chunk_size={best_chunk} as optimal,
  achieving {abs(overall_delta):.2f} higher overall RAGAS score.
""")
    print(f"{'='*65}\n")

    # Save comparison report
    report = {
        "generated_at": datetime.now().isoformat(),
        "run1": run1_name,
        "run2": run2_name,
        "comparison": comparison,
    }
    report_path = RESULTS_DIR / f"comparison_{run1_name}_vs_{run2_name}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Comparison saved → {report_path}\n")

    return report


def print_all_runs():
    """Print summary of all evaluation runs."""
    runs = list(RESULTS_DIR.glob("run_*.json"))
    if not runs:
        print("No evaluation runs found.")
        return

    print(f"\n{'='*65}")
    print(f"  ALL EVALUATION RUNS")
    print(f"{'='*65}")
    print(f"  {'Run':<20} {'Faith':>8} {'Relev':>8} {'Recall':>8} {'Overall':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for run_path in sorted(runs):
        with open(run_path) as f:
            r = json.load(f)
        print(f"  {r['run_name']:<20} {r['faithfulness']:>8.4f} {r['answer_relevancy']:>8.4f} {r['context_recall']:>8.4f} {r['overall']:>8.4f}")

    print(f"{'='*65}\n")
