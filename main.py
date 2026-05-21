#!/usr/bin/env python3
"""
Main entry point for the Agentic Multi-Hop RAG System (v2).

Usage:
    python main.py index       [--subset N]     Build corpus indices
    python main.py query       "question"       Single query (agentic)
    python main.py static      "question"       Single query (static)
    python main.py evaluate    [--sample N]     Run evaluation pipeline
    python main.py plots                        Generate ALL plots
    python main.py chat                         Interactive Rich CLI
    python main.py serve                        Launch Gradio web UI
    python main.py streamlit                    Launch Streamlit web UI
"""

from __future__ import annotations

import sys
import click

from src.config import cfg
from src.utils import get_logger

log = get_logger("main")


@click.group()
def cli():
    """Agentic Multi-Hop RAG System (v2 — with dynamic budgeting & ROUGE-L)."""
    cfg.ensure_dirs()


@cli.command()
@click.option("--subset", default=cfg.corpus_subset_size, type=int,
              help="Number of documents to index (default: 50000)")
def index(subset: int):
    """Build BM25 + Dense indices from HotpotQA Wikipedia corpus."""
    from src.indexing.corpus_loader import load_hotpotqa_corpus
    from src.indexing.chunker import chunk_documents
    from src.indexing.indexer import HybridIndex

    log.info("═══ INDEXING PIPELINE ═══")
    log.info("Subset size: %d documents", subset)

    docs   = load_hotpotqa_corpus(subset_size=subset)
    chunks = chunk_documents(docs)
    hybrid = HybridIndex.build_all(chunks)
    hybrid.save_all()

    log.info("═══ INDEXING COMPLETE ═══")
    log.info("Indexed %d documents → %d chunks", len(docs), len(chunks))
    log.info("Indices saved to: %s", cfg.index_dir)


@cli.command()
@click.argument("question")
def query(question: str):
    """Run a single question through the Agentic RAG pipeline."""
    from src.indexing.indexer import HybridIndex
    from src.retrieval.retriever import Retriever
    from src.agent.agent import AgenticRAG

    index     = HybridIndex.load_all()
    retriever = Retriever(index)
    agent     = AgenticRAG(retriever=retriever)

    answer, trace = agent.query(question)

    print("\n" + "=" * 70)
    print(f"Question:   {question}")
    print(f"Answer:     {answer}")
    print(f"Complexity: {trace.complexity}  (budget: {trace.dynamic_budget} calls)")
    print(f"Sub-Qs:     {trace.sub_questions}")
    print(f"Iterations: {len(trace.iterations)}")
    if trace.verification:
        v = trace.verification
        print(f"Verification: confidence={v['confidence']:.2%}, passed={v['passed']}")
    print(f"Stats:      {trace.token_summary}")
    print("=" * 70)


@cli.command()
@click.argument("question")
def static(question: str):
    """Run a single question through the Static RAG baseline."""
    from src.indexing.indexer import HybridIndex
    from src.retrieval.retriever import Retriever
    from src.agent.static_rag import StaticRAG

    index     = HybridIndex.load_all()
    retriever = Retriever(index)
    rag       = StaticRAG(retriever=retriever)

    answer, trace = rag.query(question)

    print("\n" + "=" * 70)
    print(f"Question: {question}")
    print(f"Answer:   {answer}")
    print(f"Stats:    {trace.token_summary}")
    print("=" * 70)


@cli.command()
@click.option("--sample", default=cfg.eval_sample_size, type=int,
              help="Number of questions to evaluate")
def evaluate(sample: int):
    """Run comparative evaluation: Agentic vs Static RAG."""
    from src.indexing.indexer import HybridIndex
    from src.retrieval.retriever import Retriever
    from src.evaluation.evaluator import Evaluator

    log.info("═══ EVALUATION PIPELINE (v2) ═══")
    log.info("Sample size: %d questions", sample)

    index     = HybridIndex.load_all()
    retriever = Retriever(index)
    evaluator = Evaluator(retriever=retriever)

    results = evaluator.run(sample_size=sample)

    log.info("═══ EVALUATION COMPLETE ═══")
    log.info("Results saved to: %s", cfg.metrics_dir)


@cli.command()
def plots():
    """Generate all evaluation visualisation plots (including stretch goals)."""
    from src.evaluation.visualize import generate_all_plots
    generate_all_plots()
    log.info("All plots saved to: %s", cfg.plots_dir)


@cli.command()
def chat():
    """Launch interactive Rich CLI chatbot."""
    from src.interface.cli import run_cli
    run_cli()


@cli.command()
@click.option("--port",  default=7860, type=int, help="Server port")
@click.option("--share", is_flag=True,           help="Create public share link")
def serve(port: int, share: bool):
    """Launch Gradio web interface."""
    from src.interface.gradio_app import launch_app
    launch_app(server_port=port, share=share)


@cli.command()
@click.option("--port", default=8501, type=int, help="Server port")
def streamlit(port: int):
    """Launch Streamlit web interface."""
    import subprocess
    import os

    app_path = os.path.join("src", "interface", "streamlit_app.py")
    log.info("Launching Streamlit UI on port %d …", port)
    try:
        subprocess.run([
            "streamlit", "run", app_path,
            "--server.port",    str(port),
            "--server.address", "0.0.0.0",
        ], check=True)
    except KeyboardInterrupt:
        log.info("Streamlit server stopped.")


if __name__ == "__main__":
    cli()
