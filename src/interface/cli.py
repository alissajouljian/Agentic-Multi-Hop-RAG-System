"""
Rich CLI interface for interactive Q&A with the agentic RAG system.
"""

from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from src.agent.agent import AgenticRAG, AgentTrace
from src.agent.static_rag import StaticRAG
from src.retrieval.retriever import Retriever
from src.indexing.indexer import HybridIndex
from src.config import cfg
from src.utils import get_logger

log = get_logger(__name__)
console = Console()


def _render_trace(trace: AgentTrace, method: str) -> None:
    """Pretty-print an agent trace to the console."""
    if len(trace.sub_questions) > 1 or method == "agentic":
        sq_text = "\n".join(
            f"  {i+1}. {q}" for i, q in enumerate(trace.sub_questions)
        )
        console.print(
            Panel(sq_text, title="🔍 Sub-Questions", border_style="cyan")
        )

    for it in trace.iterations:
        table = Table(
            title=f"📚 Retrieval (Iteration {it['iteration']})",
            box=box.ROUNDED,
            show_lines=True,
        )
        table.add_column("Sub-Question", style="white", max_width=50)
        table.add_column("Retrieved", justify="center")
        table.add_column("Re-ranked", justify="center")
        table.add_column("Top Sources", style="dim")

        for sr in it.get("sub_results", []):
            table.add_row(
                sr["sub_question"][:50],
                str(sr["num_retrieved"]),
                str(sr["num_reranked"]),
                ", ".join(sr["top_titles"][:3]),
            )
        console.print(table)

    if trace.verification:
        v = trace.verification
        status = "✅ PASSED" if v["passed"] else "❌ FAILED"
        vtext = (
            f"Status: {status}\n"
            f"Confidence: {v['confidence']:.2%}\n"
            f"Claims: {v['supported']}/{v['total_claims']} supported"
        )
        if v["unsupported"]:
            vtext += "\n\nUnsupported claims:\n" + "\n".join(
                f"  ⚠ {c}" for c in v["unsupported"]
            )
        color = "green" if v["passed"] else "red"
        console.print(
            Panel(vtext, title="🛡️ Self-Verification", border_style=color)
        )

    stats = (
        f"⏱ {trace.elapsed_seconds:.1f}s  │  "
        f"🔍 {trace.total_retrievals} retrievals  │  "
        f"🤖 {trace.total_llm_calls} LLM calls  │  "
        f"{trace.token_summary}"
    )
    console.print(Text(stats, style="dim"))


def run_cli() -> None:
    """Interactive CLI Q&A loop."""
    console.print(
        Panel.fit(
            "[bold cyan]Agentic Multi-Hop RAG System[/bold cyan]\n"
            "[dim]Type a question and press Enter. Type 'quit' to exit.\n"
            "Type 'mode static' or 'mode agentic' to switch modes.[/dim]",
            border_style="cyan",
        )
    )

    console.print("[yellow]Loading indices …[/yellow]")
    try:
        index = HybridIndex.load_all()
        retriever = Retriever(index)
        agentic = AgenticRAG(retriever=retriever)
        static = StaticRAG(retriever=retriever)
        console.print("[green]✓ Indices loaded successfully![/green]\n")
    except Exception as e:
        console.print(f"[red]Failed to load indices: {e}[/red]")
        console.print("[yellow]Run 'python main.py index' first.[/yellow]")
        return

    mode = "agentic"

    while True:
        try:
            query = console.input(f"\n[bold {'cyan' if mode == 'agentic' else 'magenta'}][{mode}][/] 💬 > ")
        except (KeyboardInterrupt, EOFError):
            break

        query = query.strip()
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            break
        if query.lower().startswith("mode "):
            new_mode = query.split(" ", 1)[1].strip().lower()
            if new_mode in ("agentic", "static"):
                mode = new_mode
                console.print(f"[green]Switched to {mode} mode[/green]")
            else:
                console.print("[red]Unknown mode. Use 'agentic' or 'static'.[/red]")
            continue

        console.print(f"\n[dim]Processing with {mode} RAG …[/dim]\n")

        try:
            if mode == "agentic":
                answer, trace = agentic.query(query)
            else:
                answer, trace = static.query(query)

            _render_trace(trace, mode)

            console.print(
                Panel(
                    Markdown(answer),
                    title="💡 Answer",
                    border_style="green",
                    padding=(1, 2),
                )
            )

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

    console.print("\n[dim]Goodbye! 👋[/dim]")
