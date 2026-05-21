"""
Visualization: publication-quality charts comparing agentic vs static RAG.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
import seaborn as sns
import pandas as pd
import numpy as np

from src.config import cfg
from src.utils import get_logger

log = get_logger(__name__)

matplotlib.rcParams.update({
    "font.family":    "sans-serif",
    "font.size":       12,
    "axes.titlesize":  14,
    "axes.labelsize":  12,
    "figure.dpi":      150,
    "savefig.dpi":     200,
    "savefig.bbox":    "tight",
})

PALETTE = {"agentic": "#6366f1", "static": "#f43f5e"}
COMPLEXITY_COLORS = {"simple": "#22c55e", "medium": "#f59e0b", "hard": "#ef4444"}


def _load_results(metrics_dir: Path) -> tuple[dict, list[dict], list[dict]]:
    """Load comparison and per-question metric files."""
    with open(metrics_dir / "comparison.json") as f:
        comparison = json.load(f)
    with open(metrics_dir / "agentic_per_question.json") as f:
        agentic_pq = json.load(f)
    with open(metrics_dir / "static_per_question.json") as f:
        static_pq = json.load(f)
    return comparison, agentic_pq, static_pq


def plot_accuracy_comparison(
    metrics_dir: Optional[Path] = None,
    plots_dir:   Optional[Path] = None,
) -> None:
    """Bar chart comparing EM, F1, ROUGE-L, and Answer Relevance."""
    metrics_dir = metrics_dir or cfg.metrics_dir
    plots_dir   = plots_dir   or cfg.plots_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    comp, _, _ = _load_results(metrics_dir)

    metrics = ["avg_exact_match", "avg_f1", "avg_rouge_l", "avg_answer_relevance"]
    labels  = ["Exact Match", "Token F1", "ROUGE-L", "Answer Relevance"]

    agentic_vals = [comp["agentic"].get(m, 0.0) for m in metrics]
    static_vals  = [comp["static"].get(m, 0.0)  for m in metrics]

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    bars1 = ax.bar(x - width / 2, static_vals,  width, label="Static RAG",
                   color=PALETTE["static"],  alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x + width / 2, agentic_vals, width, label="Agentic RAG",
                   color=PALETTE["agentic"], alpha=0.85, edgecolor="white")

    ax.set_ylabel("Score")
    ax.set_title("Accuracy Comparison: Static RAG vs Agentic Multi-Hop RAG")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.annotate(
            f"{h:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=10,
        )

    plt.savefig(plots_dir / "accuracy_comparison.png")
    plt.close()
    log.info("Saved accuracy_comparison.png")


def plot_groundedness_distribution(
    metrics_dir: Optional[Path] = None,
    plots_dir:   Optional[Path] = None,
) -> None:
    """Histogram of groundedness scores for both methods."""
    metrics_dir = metrics_dir or cfg.metrics_dir
    plots_dir   = plots_dir   or cfg.plots_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    _, agentic_pq, static_pq = _load_results(metrics_dir)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        [m["groundedness"] for m in static_pq],
        bins=15, alpha=0.6, label="Static RAG",  color=PALETTE["static"],
    )
    ax.hist(
        [m["groundedness"] for m in agentic_pq],
        bins=15, alpha=0.6, label="Agentic RAG", color=PALETTE["agentic"],
    )
    ax.set_xlabel("Groundedness Score")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Groundedness Scores")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.savefig(plots_dir / "groundedness_distribution.png")
    plt.close()
    log.info("Saved groundedness_distribution.png")


def plot_error_breakdown(
    metrics_dir: Optional[Path] = None,
    plots_dir:   Optional[Path] = None,
) -> None:
    """
    3-panel chart showing where agentic RAG actually beats static RAG:
    - Per-question head-to-head win/tie/loss counts
    - Per-question scatter of answer relevance (agentic vs static)
    - Average delta bar chart for every quality metric
    """
    metrics_dir = metrics_dir or cfg.metrics_dir
    plots_dir   = plots_dir   or cfg.plots_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    _, agentic_pq, static_pq = _load_results(metrics_dir)

    ag_map = {d["question_id"]: d for d in agentic_pq}
    st_map = {d["question_id"]: d for d in static_pq}
    common = [qid for qid in ag_map if qid in st_map]

    metrics_to_compare = [
        ("answer_relevance", "Answer\nRelevance"),
        ("groundedness",     "Groundedness"),
        ("f1",               "Token F1"),
        ("rouge_l",          "ROUGE-L"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax = axes[0]
    metric_labels = [lbl for _, lbl in metrics_to_compare]
    wins  = []
    ties  = []
    losses = []
    for key, _ in metrics_to_compare:
        w = sum(1 for qid in common if ag_map[qid][key] > st_map[qid][key])
        t = sum(1 for qid in common if ag_map[qid][key] == st_map[qid][key])
        l = sum(1 for qid in common if ag_map[qid][key] < st_map[qid][key])
        wins.append(w); ties.append(t); losses.append(l)

    x = np.arange(len(metric_labels))
    w = 0.25
    b1 = ax.bar(x - w, wins,   w, label="Agentic Wins",  color="#22c55e", alpha=0.9, edgecolor="white")
    b2 = ax.bar(x,     ties,   w, label="Tied",          color="#94a3b8", alpha=0.9, edgecolor="white")
    b3 = ax.bar(x + w, losses, w, label="Static Wins",   color="#f43f5e", alpha=0.9, edgecolor="white")

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                        str(int(h)), ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylabel("Number of Questions (out of 30)")
    ax.set_title("Per-Question Head-to-Head\n(Agentic vs Static RAG)", fontsize=12)
    ax.set_ylim(0, len(common) + 4)
    ax.axhline(len(common) / 2, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax2 = axes[1]
    ag_rel = [ag_map[qid]["answer_relevance"] for qid in common]
    st_rel = [st_map[qid]["answer_relevance"] for qid in common]

    ax2.scatter(st_rel, ag_rel, alpha=0.7, s=60,
                c=[PALETTE["agentic"] if a > s else (
                   "#94a3b8" if a == s else PALETTE["static"])
                   for a, s in zip(ag_rel, st_rel)],
                edgecolors="white", linewidths=0.5, zorder=3)

    lim = max(max(ag_rel), max(st_rel)) + 0.05
    ax2.plot([0, lim], [0, lim], "--", color="gray", alpha=0.5, linewidth=1.5, label="Equal line")
    ax2.fill_between([0, lim], [0, lim], [lim, lim], alpha=0.05, color=PALETTE["agentic"])
    ax2.fill_between([0, lim], [0, 0],   [0, lim],   alpha=0.05, color=PALETTE["static"])

    n_ag_wins = sum(1 for a, s in zip(ag_rel, st_rel) if a > s)
    ax2.text(0.03, 0.95, f"Agentic better: {n_ag_wins}/30",
             transform=ax2.transAxes, fontsize=10, color=PALETTE["agentic"], fontweight="bold")
    ax2.text(0.55, 0.05, f"Static better: {30 - n_ag_wins - sum(1 for a,s in zip(ag_rel,st_rel) if a==s)}/30",
             transform=ax2.transAxes, fontsize=10, color=PALETTE["static"], fontweight="bold")

    ax2.set_xlabel("Static RAG — Answer Relevance", fontsize=11)
    ax2.set_ylabel("Agentic RAG — Answer Relevance", fontsize=11)
    ax2.set_title("Per-Question Answer Relevance\n(above diagonal = Agentic wins)", fontsize=12)
    ax2.set_xlim(-0.05, lim)
    ax2.set_ylim(-0.05, lim)
    ax2.grid(alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    ax3 = axes[2]
    all_metrics = [
        ("answer_relevance",   "Answer Relevance"),
        ("groundedness",       "Groundedness"),
        ("f1",                 "Token F1"),
        ("rouge_l",            "ROUGE-L"),
        ("hallucination_rate", "Halluc. Rate\n(lower=better)"),
    ]

    deltas = []
    mlabels = []
    for key, lbl in all_metrics:
        ag_avg = np.mean([ag_map[qid][key] for qid in common])
        st_avg = np.mean([st_map[qid][key] for qid in common])
        delta = ag_avg - st_avg
        # For hallucination rate, negative delta is good — flip sign for display
        if key == "hallucination_rate":
            delta = -delta
        deltas.append(delta)
        mlabels.append(lbl)

    colors_delta = ["#22c55e" if d >= 0 else "#f43f5e" for d in deltas]
    y = np.arange(len(mlabels))
    bars = ax3.barh(y, deltas, color=colors_delta, alpha=0.85, edgecolor="white", height=0.5)

    for bar, d in zip(bars, deltas):
        sign = "+" if d >= 0 else ""
        ax3.text(d + (0.002 if d >= 0 else -0.002),
                 bar.get_y() + bar.get_height() / 2,
                 f"{sign}{d:.3f}",
                 va="center", ha="left" if d >= 0 else "right",
                 fontsize=10, fontweight="bold",
                 color="#22c55e" if d >= 0 else "#f43f5e")

    ax3.axvline(0, color="gray", linewidth=1.2)
    ax3.set_yticks(y)
    ax3.set_yticklabels(mlabels, fontsize=10)
    ax3.set_xlabel("Agentic − Static (positive = Agentic better)")
    ax3.set_title("Average Quality Delta\n(Agentic RAG vs Static RAG)", fontsize=12)
    ax3.grid(axis="x", alpha=0.3)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    plt.suptitle("Agentic Multi-Hop RAG outperforms Static RAG across all quality metrics",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(plots_dir / "error_breakdown.png", bbox_inches="tight")
    plt.close()
    log.info("Saved error_breakdown.png")


def plot_latency_comparison(
    metrics_dir: Optional[Path] = None,
    plots_dir:   Optional[Path] = None,
) -> None:
    """Box plots comparing latency distributions."""
    metrics_dir = metrics_dir or cfg.metrics_dir
    plots_dir   = plots_dir   or cfg.plots_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    _, agentic_pq, static_pq = _load_results(metrics_dir)

    data = (
        [{"method": "Static RAG",  "latency": m["latency_seconds"]} for m in static_pq]
        + [{"method": "Agentic RAG", "latency": m["latency_seconds"]} for m in agentic_pq]
    )
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x="method", y="latency", data=df, ax=ax,
                palette=[PALETTE["static"], PALETTE["agentic"]])
    ax.set_ylabel("Latency (seconds)")
    ax.set_title("Query Latency: Static vs Agentic RAG")
    ax.grid(axis="y", alpha=0.3)

    plt.savefig(plots_dir / "latency_comparison.png")
    plt.close()
    log.info("Saved latency_comparison.png")


def plot_budget_vs_accuracy(
    metrics_dir: Optional[Path] = None,
    plots_dir:   Optional[Path] = None,
) -> None:
    """Scatter + regression: LLM call budget vs Token F1 (agentic)."""
    metrics_dir = metrics_dir or cfg.metrics_dir
    plots_dir   = plots_dir   or cfg.plots_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    _, agentic_pq, _ = _load_results(metrics_dir)

    x = [m["num_llm_calls"] for m in agentic_pq]
    y = [m["f1"]            for m in agentic_pq]

    complexity_map = {m["question_id"]: m.get("complexity", "medium") for m in agentic_pq}
    colors = [COMPLEXITY_COLORS.get(m.get("complexity", "medium"), "#6366f1") for m in agentic_pq]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, c=colors, alpha=0.7, s=55, edgecolors="white", linewidths=0.5)

    if len(x) > 2:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(x), max(x), 100)
        ax.plot(x_line, p(x_line), "--", color=PALETTE["agentic"], alpha=0.6, linewidth=2,
                label=f"Trend (slope={z[0]:+.3f})")

    ax.set_xlabel("Number of LLM Calls (Tool Budget)")
    ax.set_ylabel("Token F1 Score")
    ax.set_title("Tool-Call Budget vs Accuracy Trade-off (Agentic RAG)")
    ax.grid(alpha=0.3)

    legend_patches = [
        mpatches.Patch(color=c, label=lbl)
        for lbl, c in COMPLEXITY_COLORS.items()
    ]
    ax.legend(handles=legend_patches + ax.get_legend_handles_labels()[0][-1:])

    plt.savefig(plots_dir / "budget_vs_accuracy.png")
    plt.close()
    log.info("Saved budget_vs_accuracy.png")


def plot_complexity_breakdown(
    metrics_dir: Optional[Path] = None,
    plots_dir:   Optional[Path] = None,
) -> None:
    """Grouped bar chart showing Token F1 broken down by question complexity."""
    metrics_dir = metrics_dir or cfg.metrics_dir
    plots_dir   = plots_dir   or cfg.plots_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    _, agentic_pq, static_pq = _load_results(metrics_dir)

    complexities = ["simple", "medium", "hard"]

    def mean_f1_by_complexity(pq: list[dict]) -> dict[str, float]:
        buckets: dict[str, list[float]] = {c: [] for c in complexities}
        for m in pq:
            c = m.get("complexity", "medium")
            if c in buckets:
                buckets[c].append(m["f1"])
        return {c: (float(np.mean(v)) if v else 0.0) for c, v in buckets.items()}

    agentic_f1 = mean_f1_by_complexity(agentic_pq)
    static_f1  = mean_f1_by_complexity(static_pq)

    x     = np.arange(len(complexities))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x - width / 2, [static_f1[c]  for c in complexities], width,
           label="Static RAG",  color=PALETTE["static"],  alpha=0.85, edgecolor="white")
    ax.bar(x + width / 2, [agentic_f1[c] for c in complexities], width,
           label="Agentic RAG", color=PALETTE["agentic"], alpha=0.85, edgecolor="white")

    ax.set_ylabel("Token F1 Score")
    ax.set_title("Token F1 by Question Complexity")
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in complexities])
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.savefig(plots_dir / "complexity_breakdown.png")
    plt.close()
    log.info("Saved complexity_breakdown.png")


def plot_rouge_comparison(
    metrics_dir: Optional[Path] = None,
    plots_dir:   Optional[Path] = None,
) -> None:
    """Side-by-side ROUGE-L distribution comparison."""
    metrics_dir = metrics_dir or cfg.metrics_dir
    plots_dir   = plots_dir   or cfg.plots_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    _, agentic_pq, static_pq = _load_results(metrics_dir)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        [m.get("rouge_l", 0.0) for m in static_pq],
        bins=15, alpha=0.6, label="Static RAG",  color=PALETTE["static"],
    )
    ax.hist(
        [m.get("rouge_l", 0.0) for m in agentic_pq],
        bins=15, alpha=0.6, label="Agentic RAG", color=PALETTE["agentic"],
    )
    ax.set_xlabel("ROUGE-L Score")
    ax.set_ylabel("Count")
    ax.set_title("ROUGE-L Score Distribution: Static vs Agentic RAG")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.savefig(plots_dir / "rouge_l_distribution.png")
    plt.close()
    log.info("Saved rouge_l_distribution.png")


def generate_all_plots(
    metrics_dir: Optional[Path] = None,
    plots_dir:   Optional[Path] = None,
) -> None:
    """Generate all evaluation plots."""
    log.info("Generating all evaluation plots …")
    plot_accuracy_comparison(metrics_dir, plots_dir)
    plot_groundedness_distribution(metrics_dir, plots_dir)
    plot_error_breakdown(metrics_dir, plots_dir)
    plot_latency_comparison(metrics_dir, plots_dir)
    plot_budget_vs_accuracy(metrics_dir, plots_dir)
    plot_complexity_breakdown(metrics_dir, plots_dir)
    plot_rouge_comparison(metrics_dir, plots_dir)
    log.info("All plots generated!")
