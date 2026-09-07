"""
RetrievalForge - Advanced Retrieval RAG Lab CLI.

Interactive CLI and batch query interface for evaluating, visualizing,
and benchmarking advanced retrieval strategies (Dense, BM25, MMR, Hybrid,
Multi-Query, Cross-Encoder Reranking, and Contextual Compression).
"""

import sys
import argparse
from typing import Optional, List

from app.config import settings
from app.models.schemas import MetadataFilter, RetrievalTrace, RAGResponse
from app.pipeline.rag_pipeline import RAGPipeline, SUPPORTED_STRATEGIES
from app.debugging.retrieval_debugger import RetrievalDebugger
from app.generation.generator import RAGGenerator

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


def print_banner() -> None:
    """Prints the RetrievalForge welcome banner."""
    title = "🔥 RetrievalForge — Advanced Retrieval RAG Lab"
    subtitle = "Dense | BM25 | MMR | Hybrid RRF | Multi-Query | Cross-Encoder Reranking | Compression"
    if RICH_AVAILABLE:
        panel_text = Text()
        panel_text.append(f"{title}\n", style="bold cyan")
        panel_text.append(subtitle, style="dim white")
        console.print(Panel(panel_text, border_style="cyan"))
    else:
        print("=" * 80)
        print(title)
        print(subtitle)
        print("=" * 80)


def print_answer(response: RAGResponse) -> None:
    """Displays the synthesized answer and cited sources in a styled panel."""
    if RICH_AVAILABLE:
        ans_text = Text()
        ans_text.append(f"{response.answer}\n\n", style="bold white")
        if response.sources:
            ans_text.append("Cited Sources: ", style="bold cyan")
            ans_text.append(", ".join(f"[{s}]" for s in response.sources), style="green")
        else:
            ans_text.append("Cited Sources: None", style="dim")
        console.print(Panel(ans_text, title=f"💡 [bold white]Generated Answer ({response.strategy})[/bold white]", border_style="green"))
    else:
        print("\n" + "=" * 80)
        print(f"ANSWER ({response.strategy}):")
        print(response.answer)
        if response.sources:
            print("\nCited Sources:", ", ".join(f"[{s}]" for s in response.sources))
        print("=" * 80 + "\n")


def execute_query(
    query: str,
    pipeline: RAGPipeline,
    debugger: RetrievalDebugger,
    generator: RAGGenerator,
    strategy: str = "hybrid_reranked",
    top_k: int = 5,
    filters: Optional[MetadataFilter] = None,
    compress: Optional[bool] = None,
    generate_answer: bool = True,
) -> RAGResponse:
    """
    Executes retrieval and optional answer synthesis for a single query.
    """
    trace = pipeline.run(
        query=query,
        strategy=strategy,
        top_k=top_k,
        filters=filters,
        compress=compress,
    )

    debugger.print_trace(trace)

    if generate_answer:
        response = generator.generate_from_trace(trace)
        print_answer(response)
        return response

    return RAGResponse(
        answer="[Generation skipped by user request]",
        query=query,
        sources=[doc.metadata.get("source", doc.document_id) for doc in trace.candidates],
        strategy=strategy,
        trace=trace,
    )


def execute_comparison(
    query: str,
    pipeline: RAGPipeline,
    debugger: RetrievalDebugger,
    generator: RAGGenerator,
    strategies: Optional[List[str]] = None,
    top_k: int = 5,
    filters: Optional[MetadataFilter] = None,
    generate_answer: bool = True,
) -> None:
    """
    Executes multiple retrieval strategies side-by-side for comparison.
    """
    to_compare = strategies or ["dense", "bm25", "hybrid", "hybrid_reranked"]
    traces = []

    if RICH_AVAILABLE:
        console.print(f"\n[bold cyan]Running side-by-side comparison across {len(to_compare)} strategies...[/bold cyan]")
    else:
        print(f"\nRunning comparison across {len(to_compare)} strategies...")

    for strat in to_compare:
        trace = pipeline.run(
            query=query,
            strategy=strat,
            top_k=top_k,
            filters=filters,
        )
        traces.append(trace)

    debugger.compare_traces(traces)

    if generate_answer and traces:
        # Generate answer using the best reranked trace
        best_trace = traces[-1]
        response = generator.generate_from_trace(best_trace)
        print_answer(response)


def interactive_repl(
    pipeline: RAGPipeline,
    debugger: RetrievalDebugger,
    generator: RAGGenerator,
    default_strategy: str = "hybrid_reranked",
) -> None:
    """
    Runs an interactive terminal REPL session.
    """
    current_strategy = default_strategy
    compare_mode = False

    print_banner()
    print("Commands:")
    print("  :s <strategy>  - Switch retrieval strategy")
    print("  :compare       - Toggle side-by-side strategy comparison mode")
    print("  :list          - List available retrieval strategies")
    print("  :exit, :quit   - Exit the lab\n")

    while True:
        try:
            prompt_str = f"RetrievalForge ({current_strategy}{' [COMPARE]' if compare_mode else ''}) > "
            user_input = input(prompt_str).strip()

            if not user_input:
                continue

            if user_input.lower() in (":exit", ":quit", "exit", "quit"):
                print("Exiting RetrievalForge. Goodbye!")
                break

            if user_input.lower() == ":list":
                print("\nAvailable Strategies:")
                for s in SUPPORTED_STRATEGIES:
                    print(f"  - {s}")
                print()
                continue

            if user_input.lower() == ":compare":
                compare_mode = not compare_mode
                status = "ENABLED" if compare_mode else "DISABLED"
                print(f"Comparison mode {status}.\n")
                continue

            if user_input.startswith(":s "):
                new_strat = user_input[3:].strip().lower()
                if new_strat in SUPPORTED_STRATEGIES:
                    current_strategy = new_strat
                    print(f"Active strategy set to: '{current_strategy}'\n")
                else:
                    print(f"Unknown strategy '{new_strat}'. Supported: {SUPPORTED_STRATEGIES}\n")
                continue

            # Execute search
            if compare_mode:
                execute_comparison(
                    query=user_input,
                    pipeline=pipeline,
                    debugger=debugger,
                    generator=generator,
                    top_k=settings.DEFAULT_TOP_K,
                )
            else:
                execute_query(
                    query=user_input,
                    pipeline=pipeline,
                    debugger=debugger,
                    generator=generator,
                    strategy=current_strategy,
                    top_k=settings.DEFAULT_TOP_K,
                )

        except (KeyboardInterrupt, EOFError):
            print("\nExiting RetrievalForge. Goodbye!")
            break
        except Exception as e:
            print(f"\n[Error] {e}\n")


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="RetrievalForge — Advanced Retrieval RAG Lab CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-q", "--query",
        type=str,
        default=None,
        help="Query string to search. If omitted, starts interactive REPL mode.",
    )
    parser.add_argument(
        "-s", "--strategy",
        type=str,
        default="hybrid_reranked",
        choices=SUPPORTED_STRATEGIES,
        help="Retrieval strategy to execute.",
    )
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=settings.DEFAULT_TOP_K,
        help="Number of final context documents to retrieve.",
    )
    parser.add_argument(
        "-c", "--compare",
        action="store_true",
        help="Run side-by-side comparison across dense, bm25, hybrid, and reranked strategies.",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Force contextual compression on retrieved documents.",
    )
    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="Skip LLM answer generation and only display retrieval trace.",
    )
    parser.add_argument(
        "--department",
        type=str,
        default=None,
        help="Filter candidates by metadata department (e.g. HR, Billing, Security).",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Filter candidates by document version (e.g. 1.0, 2.0).",
    )
    parser.add_argument(
        "--list-strategies",
        action="store_true",
        help="List all supported retrieval strategies and exit.",
    )
    return parser


def main() -> None:
    """Main CLI entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.list_strategies:
        print("Supported Retrieval Strategies in RetrievalForge:")
        for s in SUPPORTED_STRATEGIES:
            print(f"  - {s}")
        sys.exit(0)

    pipeline = RAGPipeline(strategy=args.strategy)
    debugger = RetrievalDebugger()
    generator = RAGGenerator()

    # Build metadata filter if specified
    filters = None
    if args.department or args.version:
        filters = MetadataFilter(
            department=args.department,
            version=args.version,
        )

    # If query was passed, run once and exit
    if args.query:
        if args.compare:
            execute_comparison(
                query=args.query,
                pipeline=pipeline,
                debugger=debugger,
                generator=generator,
                top_k=args.top_k,
                filters=filters,
                generate_answer=not args.no_generate,
            )
        else:
            execute_query(
                query=args.query,
                pipeline=pipeline,
                debugger=debugger,
                generator=generator,
                strategy=args.strategy,
                top_k=args.top_k,
                filters=filters,
                compress=args.compress if args.compress else None,
                generate_answer=not args.no_generate,
            )
    else:
        # Start interactive REPL
        interactive_repl(
            pipeline=pipeline,
            debugger=debugger,
            generator=generator,
            default_strategy=args.strategy,
        )


if __name__ == "__main__":
    main()
