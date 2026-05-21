"""
Gradio web interface for the Agentic Multi-Hop RAG system.
"""

from __future__ import annotations

import json
from typing import Optional

import gradio as gr

from src.agent.agent import AgenticRAG, AgentTrace
from src.agent.static_rag import StaticRAG
from src.retrieval.retriever import Retriever
from src.indexing.indexer import HybridIndex
from src.config import cfg
from src.utils import get_logger

log = get_logger(__name__)

_retriever: Optional[Retriever] = None
_agentic: Optional[AgenticRAG] = None
_static: Optional[StaticRAG] = None


def _ensure_loaded():
    """Lazy-load indices and models."""
    global _retriever, _agentic, _static
    if _retriever is None:
        log.info("Loading indices for Gradio app …")
        index = HybridIndex.load_all()
        _retriever = Retriever(index)
        _agentic = AgenticRAG(retriever=_retriever)
        _static = StaticRAG(retriever=_retriever)
        log.info("Indices loaded!")


def _format_trace(trace: AgentTrace) -> str:
    """Format agent trace as readable markdown."""
    parts = []

    if trace.sub_questions:
        parts.append("### 🔍 Sub-Questions")
        for i, q in enumerate(trace.sub_questions):
            parts.append(f"{i+1}. {q}")

    for it in trace.iterations:
        parts.append(f"\n### 📚 Retrieval (Iteration {it['iteration']})")
        for sr in it.get("sub_results", []):
            parts.append(f"- **{sr['sub_question'][:60]}**")
            parts.append(f"  - Retrieved: {sr['num_retrieved']} → Re-ranked: {sr['num_reranked']}")
            parts.append(f"  - Top sources: {', '.join(sr['top_titles'][:3])}")

    if trace.verification:
        v = trace.verification
        status = "✅ PASSED" if v["passed"] else "❌ FAILED"
        parts.append(f"\n### 🛡️ Verification: {status}")
        parts.append(f"- Confidence: {v['confidence']:.1%}")
        parts.append(f"- Claims supported: {v['supported']}/{v['total_claims']}")
        if v["unsupported"]:
            parts.append("- Unsupported claims:")
            for c in v["unsupported"]:
                parts.append(f"  - ⚠️ {c}")

    parts.append(f"\n---\n⏱ {trace.elapsed_seconds:.1f}s | "
                 f"🔍 {trace.total_retrievals} retrievals | "
                 f"🤖 {trace.total_llm_calls} LLM calls | "
                 f"{trace.token_summary}")

    return "\n".join(parts)


def process_query(question: str, mode: str) -> tuple[str, str, float]:
    """
    Process a question and return (answer, trace, confidence).
    """
    _ensure_loaded()

    if not question.strip():
        return "Please enter a question.", "", 0.0

    try:
        if mode == "Agentic Multi-Hop RAG":
            answer, trace = _agentic.query(question)
        else:
            answer, trace = _static.query(question)

        trace_md = _format_trace(trace)
        confidence = trace.verification["confidence"] if trace.verification else 0.0

        return answer, trace_md, confidence

    except Exception as e:
        log.error("Query failed: %s", e)
        return f"Error: {e}", "", 0.0


def create_app() -> gr.Blocks:
    """Build the Gradio interface."""
    with gr.Blocks(
        title="Agentic Multi-Hop RAG",
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="rose",
        ),
        css="""
        .main-title { text-align: center; margin-bottom: 0.5em; }
        .subtitle { text-align: center; color: #666; margin-bottom: 1.5em; }
        """,
    ) as app:
        gr.Markdown(
            "# 🧠 Agentic Multi-Hop RAG System\n"
            "### Ask complex questions that require reasoning across multiple documents",
            elem_classes=["main-title"],
        )

        with gr.Row():
            with gr.Column(scale=3):
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="e.g., Were Scott Derrickson and Ed Wood of the same nationality?",
                    lines=2,
                )
            with gr.Column(scale=1):
                mode_selector = gr.Radio(
                    choices=["Agentic Multi-Hop RAG", "Static Single-Shot RAG"],
                    value="Agentic Multi-Hop RAG",
                    label="Mode",
                )

        submit_btn = gr.Button("🚀 Ask", variant="primary", size="lg")

        with gr.Row():
            with gr.Column(scale=2):
                answer_output = gr.Markdown(label="Answer")
            with gr.Column(scale=1):
                confidence_output = gr.Number(
                    label="Verification Confidence",
                    precision=2,
                )

        with gr.Accordion("📋 Agent Trace (Click to expand)", open=False):
            trace_output = gr.Markdown()

        gr.Examples(
            examples=[
                ["Were Scott Derrickson and Ed Wood of the same nationality?"],
                ["What government position was held by the woman who portrayed Velma combust on Scooby-Doo?"],
                ["Which magazine was started first, Arthur's Magazine or First for Women?"],
                ["The arena where the Weights Roadrunners play opened in what year?"],
            ],
            inputs=question_input,
        )

        submit_btn.click(
            fn=process_query,
            inputs=[question_input, mode_selector],
            outputs=[answer_output, trace_output, confidence_output],
        )

    return app


def launch_app(**kwargs) -> None:
    """Launch the Gradio app."""
    app = create_app()
    port = kwargs.pop("server_port", 7860)
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=kwargs.pop("share", False),
        **kwargs,
    )
