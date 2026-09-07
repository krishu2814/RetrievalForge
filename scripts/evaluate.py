"""
RetrievalForge - Automated Retrieval Benchmark Runner.

Runs the curated evaluation dataset against multiple retrieval strategies
(Dense, BM25, MMR, Hybrid RRF, Cross-Encoder Reranking, and Contextual Compression),
computes IR metrics (Recall@K, Hit@K, MRR@K, NDCG@K, Precision@K),
renders comparative benchmark tables, and exports results to JSON and Markdown.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --strategies dense bm25 hybrid hybrid_reranked --top-k 5
"""

import sys
import time
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.pipeline.rag_pipeline import RAGPipeline, SUPPORTED_STRATEGIES
from app.evaluation.metrics import evaluate_query_retrieval, aggregate_metrics, QueryMetrics

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None

logger = logging.getLogger("evaluate")


def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Loads the evaluation dataset from a JSON file."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at: {dataset_path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def benchmark_strategy(
    strategy: str,
    dataset: List[Dict[str, Any]],
    pipeline: RAGPipeline,
    top_k: int = 5,
    compress: bool = False,
) -> Dict[str, Any]:
    """
    Evaluates a single retrieval strategy across all queries in the dataset.
    """
    query_metrics: List[QueryMetrics] = []

    for item in dataset:
        q_id = item["id"]
        query = item["query"]
        category = item["category"]
        expected_targets = item.get("expected_sources", []) + item.get("expected_doc_ids", [])

        # Measure retrieval latency
        t0 = time.perf_counter()
        trace = pipeline.run(
            query=query,
            strategy=strategy,
            top_k=top_k,
            compress=compress,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        # Extract retrieved document identifiers
        retrieved_items = []
        for doc in trace.candidates:
            src = doc.metadata.get("source")
            if src:
                retrieved_items.append(src)
            retrieved_items.append(doc.document_id)

        m = evaluate_query_retrieval(
            query_id=q_id,
            query=query,
            category=category,
            retrieved_items=retrieved_items,
            expected_targets=expected_targets,
            latency_ms=latency_ms,
        )
        query_metrics.append(m)

    aggregation = aggregate_metrics(query_metrics)
    return {
        "strategy": strategy,
        "top_k": top_k,
        "compress": compress,
        "overall": aggregation.get("overall", {}),
        "by_category": aggregation.get("by_category", {}),
        "queries": [m.model_dump() for m in query_metrics],
    }


def render_terminal_results(benchmark_results: Dict[str, Dict[str, Any]]) -> None:
    """
    Renders formatted terminal tables comparing retrieval strategies.
    """
    if not RICH_AVAILABLE:
        print_ascii_results(benchmark_results)
        return

    # 1. Overall Summary Table
    table = Table(
        title="🏆 RetrievalForge Strategy Benchmark (Overall Metrics)",
        box=box.ROUNDED,
        header_style="bold cyan",
        show_lines=True
    )
    table.add_column("Strategy", style="bold white", width=22)
    table.add_column("Hit@1", justify="right", style="green", width=9)
    table.add_column("Hit@3", justify="right", style="green", width=9)
    table.add_column("Hit@5", justify="right", style="bold green", width=9)
    table.add_column("Recall@5", justify="right", style="yellow", width=10)
    table.add_column("MRR@5", justify="right", style="magenta", width=9)
    table.add_column("NDCG@5", justify="right", style="bold magenta", width=10)
    table.add_column("Latency", justify="right", style="cyan", width=11)

    for strat, res in benchmark_results.items():
        o = res["overall"]
        table.add_row(
            strat,
            f"{o.get('mean_hit_at_1', 0.0) * 100:.1f}%",
            f"{o.get('mean_hit_at_3', 0.0) * 100:.1f}%",
            f"{o.get('mean_hit_at_5', 0.0) * 100:.1f}%",
            f"{o.get('mean_recall_at_5', 0.0) * 100:.1f}%",
            f"{o.get('mean_mrr_at_5', 0.0):.3f}",
            f"{o.get('mean_ndcg_at_5', 0.0):.3f}",
            f"{o.get('mean_latency_ms', 0.0):.1f} ms",
        )

    console.print(table)

    # 2. Category Breakdown Table
    cat_table = Table(
        title="📊 Performance by Query Challenge Category (MRR@5)",
        box=box.SIMPLE_HEAVY,
        header_style="bold magenta",
        show_lines=True
    )
    cat_table.add_column("Challenge Category", style="bold white", width=22)
    for strat in benchmark_results.keys():
        cat_table.add_column(strat, justify="center", style="cyan")

    # Collect categories
    first_res = next(iter(benchmark_results.values()))
    categories = list(first_res.get("by_category", {}).keys())

    for cat in categories:
        row = [cat]
        for strat, res in benchmark_results.items():
            cat_mrr = res.get("by_category", {}).get(cat, {}).get("mrr_at_5", 0.0)
            row.append(f"{cat_mrr:.3f}")
        cat_table.add_row(*row)

    console.print(cat_table)


def print_ascii_results(benchmark_results: Dict[str, Dict[str, Any]]) -> None:
    """Fallback plain text printer for headless environments."""
    print("\n" + "=" * 95)
    print("RETRIEVALFORGE BENCHMARK RESULTS (OVERALL)")
    print("=" * 95)
    header = f"{'Strategy':<24} | {'Hit@1':<8} | {'Hit@3':<8} | {'Hit@5':<8} | {'Recall@5':<9} | {'MRR@5':<8} | {'NDCG@5':<8} | {'Latency':<10}"
    print(header)
    print("-" * len(header))

    for strat, res in benchmark_results.items():
        o = res["overall"]
        row = (
            f"{strat:<24} | "
            f"{o.get('mean_hit_at_1', 0.0) * 100:>6.1f}% | "
            f"{o.get('mean_hit_at_3', 0.0) * 100:>6.1f}% | "
            f"{o.get('mean_hit_at_5', 0.0) * 100:>6.1f}% | "
            f"{o.get('mean_recall_at_5', 0.0) * 100:>7.1f}% | "
            f"{o.get('mean_mrr_at_5', 0.0):>8.3f} | "
            f"{o.get('mean_ndcg_at_5', 0.0):>8.3f} | "
            f"{o.get('mean_latency_ms', 0.0):>7.1f} ms"
        )
        print(row)
    print("=" * 95 + "\n")


def generate_markdown_summary(benchmark_results: Dict[str, Dict[str, Any]]) -> str:
    """Generates a GitHub-flavored markdown summary table."""
    md = []
    md.append("## 🏆 RetrievalForge Benchmark Results\n")
    md.append("| Strategy | Hit@1 | Hit@3 | Hit@5 | Recall@5 | MRR@5 | NDCG@5 | Latency (ms) |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for strat, res in benchmark_results.items():
        o = res["overall"]
        md.append(
            f"| **{strat}** | "
            f"{o.get('mean_hit_at_1', 0.0) * 100:.1f}% | "
            f"{o.get('mean_hit_at_3', 0.0) * 100:.1f}% | "
            f"{o.get('mean_hit_at_5', 0.0) * 100:.1f}% | "
            f"{o.get('mean_recall_at_5', 0.0) * 100:.1f}% | "
            f"{o.get('mean_mrr_at_5', 0.0):.3f} | "
            f"{o.get('mean_ndcg_at_5', 0.0):.3f} | "
            f"{o.get('mean_latency_ms', 0.0):.1f} |"
        )

    md.append("\n### 📊 Category Breakdown (MRR@5)\n")
    first_res = next(iter(benchmark_results.values()))
    categories = list(first_res.get("by_category", {}).keys())

    cat_header = "| Category | " + " | ".join(f"**{s}**" for s in benchmark_results.keys()) + " |"
    cat_sep = "| :--- | " + " | ".join(":---:" for _ in benchmark_results.keys()) + " |"
    md.append(cat_header)
    md.append(cat_sep)

    for cat in categories:
        scores = [f"{res.get('by_category', {}).get(cat, {}).get('mrr_at_5', 0.0):.3f}" for res in benchmark_results.values()]
        md.append(f"| `{cat}` | " + " | ".join(scores) + " |")

    return "\n".join(md)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run automated benchmarks across retrieval strategies in RetrievalForge.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=settings.EVAL_DATASET_PATH,
        help="Path to evaluation dataset JSON",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["dense", "bm25", "mmr", "hybrid", "hybrid_reranked"],
        choices=SUPPORTED_STRATEGIES,
        help="Retrieval strategies to benchmark",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=settings.DEFAULT_TOP_K,
        help="Number of candidates to retrieve",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Enable contextual compression during retrieval",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=settings.EVAL_RESULTS_PATH,
        help="Path to export benchmark results JSON",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=str(Path(settings.EVAL_RESULTS_PATH).parent / "benchmark_summary.md"),
        help="Path to export markdown summary",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    print("\n" + "=" * 75)
    print("🔥 RetrievalForge - Automated Retrieval Benchmark Engine")
    print("=" * 75)
    print(f"Dataset:    {args.dataset}")
    print(f"Strategies: {', '.join(args.strategies)}")
    print(f"Top-K:      {args.top_k}")
    print(f"Compress:   {args.compress}")
    print("=" * 75 + "\n")

    dataset = load_dataset(args.dataset)
    print(f"Loaded {len(dataset)} evaluation questions from dataset.\n")

    pipeline = RAGPipeline()
    benchmark_results: Dict[str, Dict[str, Any]] = {}

    for strat in args.strategies:
        print(f"Benchmarking strategy: '{strat}'...")
        res = benchmark_strategy(
            strategy=strat,
            dataset=dataset,
            pipeline=pipeline,
            top_k=args.top_k,
            compress=args.compress,
        )
        benchmark_results[strat] = res
        print(f"  -> MRR@5: {res['overall']['mean_mrr_at_5']:.3f} | Hit@5: {res['overall']['mean_hit_at_5']*100:.1f}%\n")

    # Render results
    render_terminal_results(benchmark_results)

    # Export JSON results
    out_json_path = Path(args.output_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"\nSaved benchmark results JSON to: {out_json_path}")

    # Export Markdown summary
    out_md_path = Path(args.output_md)
    md_content = generate_markdown_summary(benchmark_results)
    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Saved benchmark markdown summary to: {out_md_path}\n")


if __name__ == "__main__":
    main()
