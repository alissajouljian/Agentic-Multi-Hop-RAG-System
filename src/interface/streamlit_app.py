"""
Professional Streamlit Chat Interface for Agentic Multi-Hop RAG.
"""

import streamlit as st
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.agent.agent import AgenticRAG
from src.agent.static_rag import StaticRAG
from src.retrieval.retriever import Retriever
from src.indexing.indexer import HybridIndex
from src.config import cfg
from src.utils import get_logger

st.set_page_config(
    page_title="Agentic RAG Chat",
    page_icon="🧠",
    layout="wide",
)

@st.cache_resource
def load_system():
    """Load indices and agents once."""
    index = HybridIndex.load_all()
    retriever = Retriever(index)
    agentic = AgenticRAG(retriever=retriever)
    static = StaticRAG(retriever=retriever)
    return agentic, static

st.markdown("""
    <style>
    .stChatMessage { border-radius: 15px; padding: 15px; margin-bottom: 10px; }
    .stChatMessage.user { background-color: #f0f2f6; }
    .stChatMessage.assistant { background-color: #ffffff; border: 1px solid #e0e0e0; }
    .agent-trace { font-size: 0.85em; color: #555; background: #f9f9f9; padding: 10px; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("🧠 Agentic RAG Settings")
    mode = st.radio("Retrieval Mode", ["Agentic Multi-Hop", "Static Single-Shot"])
    st.divider()
    st.markdown("""
    **System Capabilities:**
    - Query Decomposition
    - Hybrid BM25/Vector Search
    - Iterative Self-Verification
    - Fact-Grounded Synthesis
    """)
    if st.button("Clear History"):
        st.session_state.messages = []
        st.rerun()

st.title("Research Assistant")
st.caption("NLP Project: Agentic Multi-Hop Retrieval-Augmented Generation")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "trace" in msg:
            with st.expander("🔍 Agent Thought Trace"):
                st.markdown(msg["trace"])

if prompt := st.chat_input("Ask a research question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    agentic, static = load_system()

    with st.chat_message("assistant"):
        with st.status("Thinking...", expanded=True) as status:
            if mode == "Agentic Multi-Hop":
                answer, trace = agentic.query(prompt)
            else:
                answer, trace = static.query(prompt)
            status.update(label="Complete!", state="complete", expanded=False)

        trace_md = f"**Decomposed Sub-questions:**\n" + "\n".join([f"- {q}" for q in trace.sub_questions])
        trace_md += f"\n\n**Total LLM Calls:** {trace.total_llm_calls}"
        trace_md += f"\n**Total Retrievals:** {trace.total_retrievals}"
        trace_md += f"\n**Elapsed Time:** {trace.elapsed_seconds:.1f}s"
        
        if trace.verification:
            v = trace.verification
            trace_md += f"\n\n**Self-Verification:** {'✅ Passed' if v['passed'] else '❌ Failed'}"
            trace_md += f"\n- Confidence: {v['confidence']:.1%}"
            if v['unsupported']:
                trace_md += "\n- Unsupported Claims:\n  " + "\n  ".join([f"⚠️ {c}" for c in v['unsupported']])

        st.markdown(answer)
        with st.expander("🔍 Agent Thought Trace"):
            st.markdown(trace_md)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "trace": trace_md
        })
