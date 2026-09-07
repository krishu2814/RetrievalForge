"""
Retrieval Debugger module.
Provides rich visual diagnostic inspection and side-by-side comparison
of retrieval traces, scoring breakdowns, ranking shifts, and context compression.
"""

import logging
from typing import List, Optional, Dict, Any

from app.models.schemas import RetrievalResult, RetrievalTrace

logger = logging.getLogger(__name__)

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class RetrievalDebugger:
    """
    Diagnostic tool to inspect, visualize, and compare retrieval traces.
    Helps engineers understand exactly why documents were selected,
    how scores changed across pipeline stages, and how compression impacted context.
    """

    def __init__(self, console: Optional[Any] = None):
        if RICH_AVAILABLE:
            self.console = console or Console()
        else:
            self.console = None

    # =========================================================================
    # Terminal Display Methods (Rich-enhanced)
    # =========================================================================

    def print_trace(
        self,
        trace: RetrievalTrace,
        show_content: bool = True,
        max_content_length: int = 150
    ) -> None:
        """
        Prints a detailed, colorized diagnostic dashboard for a single RetrievalTrace.
        """
        if not RICH_AVAILABLE:
            print(self.format_trace_text(trace, show_content=show_content, max_content_length=max_content_length))
            return

        # 1. Header Panel
        header_text = Text()
        header_text.append("Query: ", style="bold cyan")
        header_text.append(f'"{trace.query}"\n', style="bold white")
        header_text.append("Strategy: ", style="bold cyan")
        header_text.append(f"{trace.strategy}\n", style="bold green")

        if trace.query_variants:
            header_text.append("Query Variants: ", style="bold cyan")
            header_text.append(f"{' | '.join(trace.query_variants)}\n", style="italic yellow")

        if trace.filters:
            header_text.append("Filters: ", style="bold cyan")
            header_text.append(f"{trace.filters.model_dump(exclude_defaults=True)}\n", style="magenta")

        header_text.append("Candidates Retrieved: ", style="bold cyan")
        header_text.append(f"{len(trace.candidates)}", style="bold yellow")

        self.console.print(Panel(header_text, title="🔍 [bold white]RetrievalForge Trace[/bold white]", border_style="cyan"))

        if not trace.candidates:
            self.console.print("[bold red]No candidate documents retrieved.[/bold red]")
            return

        # 2. Candidate Scores Table
        table = Table(
            title="Candidate Ranking & Diagnostic Scores",
            box=box.ROUNDED,
            header_style="bold magenta",
            show_lines=True
        )

        table.add_column("Rank", justify="center", style="bold cyan", width=6)
        table.add_column("Orig", justify="center", style="dim", width=6)
        table.add_column("Source / Doc ID", style="bold white", width=26)
        table.add_column("Dept", justify="center", style="green", width=10)
        table.add_column("Dense", justify="right", style="cyan", width=8)
        table.add_column("BM25", justify="right", style="yellow", width=8)
        table.add_column("RRF", justify="right", style="magenta", width=9)
        table.add_column("Rerank", justify="right", style="bold green", width=9)
        table.add_column("Ratio", justify="center", style="blue", width=7)

        for doc in trace.candidates:
            rank_str = f"#{doc.rank}"
            orig_str = f"#{doc.original_rank}" if doc.original_rank is not None else "-"
            source_name = doc.metadata.get("source") or doc.document_id
            dept = doc.metadata.get("department", "-")

            dense_str = f"{doc.dense_score:.3f}" if doc.dense_score is not None else "-"
            bm25_str = f"{doc.bm25_score:.2f}" if doc.bm25_score is not None else "-"
            rrf_str = f"{doc.fusion_score:.4f}" if doc.fusion_score is not None else "-"
            rerank_str = f"{doc.rerank_score:+.3f}" if doc.rerank_score is not None else "-"

            ratio_val = doc.metadata.get("compression_ratio")
            ratio_str = f"{int(ratio_val * 100)}%" if ratio_val is not None else "100%"

            table.add_row(
                rank_str,
                orig_str,
                source_name[:25],
                dept,
                dense_str,
                bm25_str,
                rrf_str,
                rerank_str,
                ratio_str
            )

        self.console.print(table)

        # 3. Content Snippets Panel
        if show_content:
            snippets_text = Text()
            for i, doc in enumerate(trace.candidates, start=1):
                source = doc.metadata.get("source", doc.document_id)
                snippets_text.append(f"[{i}] {source} (Rank #{doc.rank})\n", style="bold cyan")
                
                content_preview = doc.content.strip().replace("\n", " ")
                if len(content_preview) > max_content_length:
                    content_preview = content_preview[:max_content_length] + "..."
                snippets_text.append(f"    {content_preview}\n", style="white")

                if "original_content" in doc.metadata:
                    orig_len = len(doc.metadata["original_content"])
                    comp_len = len(doc.content)
                    snippets_text.append(
                        f"    [Compressed: {comp_len}/{orig_len} chars ({doc.metadata.get('compression_ratio', 1.0) * 100:.0f}% retained)]\n",
                        style="dim italic green"
                    )

            self.console.print(Panel(snippets_text, title="📄 [bold white]Context Snippets[/bold white]", border_style="dim"))

    def compare_traces(self, traces: List[RetrievalTrace]) -> None:
        """
        Prints a side-by-side comparison of multiple retrieval traces for the same query.
        Shows how different strategies rank different documents.
        """
        if not traces:
            return

        if not RICH_AVAILABLE:
            print(self.format_comparison_text(traces))
            return

        query = traces[0].query
        table = Table(
            title=f'Comparison for Query: "{query}"',
            box=box.HEAVY_EDGE,
            header_style="bold cyan",
            show_lines=True
        )

        table.add_column("Rank", justify="center", style="bold white", width=6)
        for trace in traces:
            table.add_column(f"{trace.strategy}\n({len(trace.candidates)} docs)", style="white")

        # Determine maximum candidates to compare across strategies
        max_k = max(len(t.candidates) for t in traces) if traces else 0

        for r in range(max_k):
            row = [f"#{r + 1}"]
            for trace in traces:
                if r < len(trace.candidates):
                    doc = trace.candidates[r]
                    src = doc.metadata.get("source", doc.document_id)
                    score_info = f" ({doc.score:.3f})" if doc.score is not None else ""
                    cell = f"{src}{score_info}"
                else:
                    cell = "-"
                row.append(cell)
            table.add_row(*row)

        self.console.print(table)

    # =========================================================================
    # Plain Text / Markdown Formatting Methods
    # =========================================================================

    def format_trace_text(
        self,
        trace: RetrievalTrace,
        show_content: bool = True,
        max_content_length: int = 150
    ) -> str:
        """
        Formats a retrieval trace into plain text with an ASCII table.
        Safe for logging, headless environments, or unit test assertions.
        """
        lines = []
        lines.append("=" * 80)
        lines.append(f"RETRIEVAL TRACE: '{trace.query}'")
        lines.append(f"Strategy: {trace.strategy} | Candidates: {len(trace.candidates)}")
        if trace.query_variants:
            lines.append(f"Variants: {' | '.join(trace.query_variants)}")
        if trace.filters:
            lines.append(f"Filters: {trace.filters.model_dump(exclude_defaults=True)}")
        lines.append("-" * 80)

        # Table Header
        header = f"{'Rank':<5} | {'Orig':<5} | {'Source / ID':<26} | {'Dept':<8} | {'Dense':<7} | {'BM25':<7} | {'RRF':<8} | {'Rerank':<8}"
        lines.append(header)
        lines.append("-" * len(header))

        for doc in trace.candidates:
            rank_str = f"#{doc.rank}"
            orig_str = f"#{doc.original_rank}" if doc.original_rank is not None else "-"
            source_name = (doc.metadata.get("source") or doc.document_id)[:25]
            dept = doc.metadata.get("department", "-")[:8]
            dense_str = f"{doc.dense_score:.3f}" if doc.dense_score is not None else "-"
            bm25_str = f"{doc.bm25_score:.2f}" if doc.bm25_score is not None else "-"
            rrf_str = f"{doc.fusion_score:.4f}" if doc.fusion_score is not None else "-"
            rerank_str = f"{doc.rerank_score:+.3f}" if doc.rerank_score is not None else "-"

            row = f"{rank_str:<5} | {orig_str:<5} | {source_name:<26} | {dept:<8} | {dense_str:<7} | {bm25_str:<7} | {rrf_str:<8} | {rerank_str:<8}"
            lines.append(row)

        if show_content and trace.candidates:
            lines.append("-" * 80)
            lines.append("CONTENT SNIPPETS:")
            for i, doc in enumerate(trace.candidates, start=1):
                src = doc.metadata.get("source", doc.document_id)
                snippet = doc.content.strip().replace("\n", " ")[:max_content_length]
                lines.append(f"[{i}] {src} (#{doc.rank}): {snippet}...")

        lines.append("=" * 80)
        return "\n".join(lines)

    def format_comparison_text(self, traces: List[RetrievalTrace]) -> str:
        """
        Formats a comparison of multiple traces into plain text.
        """
        if not traces:
            return "No traces to compare."

        lines = []
        lines.append("=" * 80)
        lines.append(f"STRATEGY COMPARISON: '{traces[0].query}'")
        lines.append("-" * 80)

        strategies = [t.strategy for t in traces]
        header = f"{'Rank':<6} | " + " | ".join(f"{s:<22}" for s in strategies)
        lines.append(header)
        lines.append("-" * len(header))

        max_k = max(len(t.candidates) for t in traces) if traces else 0
        for r in range(max_k):
            cells = [f"#{r + 1:<4}"]
            for t in traces:
                if r < len(t.candidates):
                    doc = t.candidates[r]
                    src = (doc.metadata.get("source") or doc.document_id)[:16]
                    score_info = f"{doc.score:.2f}" if doc.score is not None else ""
                    cells.append(f"{src} ({score_info})")
                else:
                    cells.append("-")
            lines.append(" | ".join(f"{c:<22}" if i > 0 else f"{c:<6}" for i, c in enumerate(cells)))

        lines.append("=" * 80)
        return "\n".join(lines)

    def format_markdown(self, trace: RetrievalTrace) -> str:
        """
        Formats a retrieval trace into GitHub-flavored Markdown.
        Ideal for creating PR comments, evaluation reports, and documentation.
        """
        md = []
        md.append(f"### Retrieval Trace: `{trace.query}`\n")
        md.append(f"- **Strategy**: `{trace.strategy}`")
        md.append(f"- **Candidates**: {len(trace.candidates)}")
        if trace.query_variants:
            md.append(f"- **Query Variants**: {', '.join(f'`{v}`' for v in trace.query_variants)}")
        if trace.filters:
            md.append(f"- **Filters**: `{trace.filters.model_dump(exclude_defaults=True)}`")

        md.append("\n| Rank | Orig | Source / Doc ID | Dept | Dense | BM25 | RRF | Rerank |")
        md.append("| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: |")

        for doc in trace.candidates:
            rank_str = f"#{doc.rank}"
            orig_str = f"#{doc.original_rank}" if doc.original_rank is not None else "-"
            source_name = doc.metadata.get("source") or doc.document_id
            dept = doc.metadata.get("department", "-")
            dense_str = f"{doc.dense_score:.3f}" if doc.dense_score is not None else "-"
            bm25_str = f"{doc.bm25_score:.2f}" if doc.bm25_score is not None else "-"
            rrf_str = f"{doc.fusion_score:.4f}" if doc.fusion_score is not None else "-"
            rerank_str = f"{doc.rerank_score:+.3f}" if doc.rerank_score is not None else "-"

            md.append(
                f"| {rank_str} | {orig_str} | `{source_name}` | {dept} | {dense_str} | {bm25_str} | {rrf_str} | {rerank_str} |"
            )

        return "\n".join(md)
