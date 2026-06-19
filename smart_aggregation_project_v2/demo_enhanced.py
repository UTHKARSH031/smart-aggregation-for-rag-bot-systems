"""
SMART AGGREGATION FOR RAG - ENHANCED DEMO WITH ALL 3 METHODS
============================================================

Compares three paradigms for RAG context aggregation:
  Method A -- Smart Aggregation (MMR selection)         [smart_aggregation.py]
  Method B -- Query-Aware Compression (LLM extractive)  [method_b_compression.py]
  Method C -- Cluster-Based Aggregation (KMeans)        [method_c_clustering.py]

Features:
- Token counting (input/output)
- Visual graphs and charts -- SEPARATE graphs per method + combined comparison
- Detailed metrics tracking
- Beautiful console output
- Comprehensive results export

Run with:
    python demo_enhanced.py                         # 5 docs, 15 questions (default)
    python demo_enhanced.py --max-docs 5 --max-questions 15
    python demo_enhanced.py --financebench PATH --max-docs 10 --max-questions 20
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'main code'))

from chunkers import BaseChunker, ElementChunker
from embeddings import Embedder, VectorStore, CrossEncoderReranker
from smart_aggregation import SmartAggregation
from method_b_compression import QueryAwareCompression
from method_c_clustering import ClusterBasedAggregation, compute_inter_cluster_diversity
from evaluation import RAGEvaluator
import json
import time

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("  matplotlib not installed. Install with: pip install matplotlib")
    print("   Graphs will be skipped, but text output will still work.\n")

# ---------------------------------------------------------------------------
# Colour palette (consistent across all charts)
# ---------------------------------------------------------------------------
COLORS = {
    'method_a': '#3B82F6',   # blue  -- Smart Aggregation (MMR)
    'method_b': '#10B981',   # green -- Compression
    'method_c': '#F59E0B',   # amber -- Clustering
    'baseline_512': '#DC2626',
    'baseline_elem': '#8B5CF6',
    'stages': ['#3B82F6', '#10B981', '#F59E0B', '#8B5CF6'],
    'strategies': ['#EF4444', '#F59E0B', '#10B981', '#3B82F6'],
}

# ---------------------------------------------------------------------------
# Token counter
# ---------------------------------------------------------------------------
class TokenCounter:
    """Track token counts throughout the pipeline."""

    def __init__(self):
        self.counts = {
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'by_stage': {},
            'by_strategy': {}
        }

    def count_tokens(self, text):
        """Simple whitespace-based token approximation."""
        if not text:
            return 0
        return len(text.split())

    def add_stage(self, stage_name, input_text, output_chunks):
        """Track tokens for a specific stage."""
        if isinstance(input_text, str):
            input_tokens = self.count_tokens(input_text)
        else:
            input_tokens = sum(self.count_tokens(t) for t in input_text)

        total_out = 0
        for c in output_chunks:
            if hasattr(c, 'compressed_text'):       # CompressedChunk (Method B)
                total_out += self.count_tokens(c.compressed_text)
            elif hasattr(c, 'chunk'):               # ClusterResult (Method C)
                total_out += self.count_tokens(c.chunk.text)
            elif hasattr(c, 'text'):                # Chunk (Method A)
                total_out += self.count_tokens(c.text)
            elif isinstance(c, dict):
                total_out += self.count_tokens(c.get('text', ''))

        self.counts['by_stage'][stage_name] = {
            'input_tokens': input_tokens,
            'output_tokens': total_out,
            'reduction_ratio': (1 - total_out / input_tokens) if input_tokens > 0 else 0
        }
        self.counts['total_input_tokens'] += input_tokens
        self.counts['total_output_tokens'] += total_out

    def get_summary(self):
        total_in = self.counts['total_input_tokens']
        total_out = self.counts['total_output_tokens']
        total_reduction = (1 - total_out / total_in) if total_in > 0 else 0
        return {
            'total_input_tokens': total_in,
            'total_output_tokens': total_out,
            'total_reduction_ratio': total_reduction,
            'compression_ratio': total_in / total_out if total_out > 0 else 0,
            'by_stage': self.counts['by_stage']
        }


# ---------------------------------------------------------------------------
# Helpers to count output tokens per method (method-agnostic)
# ---------------------------------------------------------------------------
def _count_output_tokens_method(chunks, counter):
    """Count tokens in method output regardless of return type."""
    total = 0
    for c in chunks:
        if hasattr(c, 'compressed_text'):
            total += counter.count_tokens(c.compressed_text)
        elif hasattr(c, 'chunk'):
            total += counter.count_tokens(c.chunk.text)
        elif hasattr(c, 'text'):
            total += counter.count_tokens(c.text)
    return total


def _get_doc_id(chunk_obj):
    """Extract doc_id regardless of Method A/B/C return type."""
    if hasattr(chunk_obj, 'compressed_text'):   # CompressedChunk
        return chunk_obj.doc_id
    elif hasattr(chunk_obj, 'cluster_id'):       # ClusterResult
        return chunk_obj.chunk.doc_id
    elif hasattr(chunk_obj, 'doc_id'):           # Chunk
        return chunk_obj.doc_id
    elif isinstance(chunk_obj, dict):
        return chunk_obj.get('doc_id', '')
    return ''


# ---------------------------------------------------------------------------
# Visualisations -- separate graphs per method + combined comparison
# ---------------------------------------------------------------------------
def create_visualizations(results, output_dir='results'):
    """Generate all charts.  Separate per-method graphs + combined."""
    if not MATPLOTLIB_AVAILABLE:
        print("  Skipping visualisations (matplotlib not installed)")
        return

    os.makedirs(output_dir, exist_ok=True)

    # -- 1. Combined accuracy comparison (all 5 methods) ----------------------
    _plot_combined_accuracy(results, output_dir)

    # -- 2. Per-method token flow ----------------------------------------------
    for method in ['method_a', 'method_b', 'method_c']:
        if method in results:
            _plot_token_flow(results[method], method, output_dir)

    # -- 3. Per-method timing breakdown ---------------------------------------
    for method in ['method_a', 'method_b', 'method_c']:
        if method in results and 'timing_stats' in results[method]:
            _plot_timing(results[method]['timing_stats'], method, output_dir)

    # -- 4. Chunk distribution (shared, from Method A data) -------------------
    if 'method_a' in results and 'chunk_distribution' in results['method_a']:
        _plot_chunk_distribution(results['method_a']['chunk_distribution'], output_dir)

    # -- 5. Method comparison -- latency & output tokens -----------------------
    _plot_method_comparison(results, output_dir)

    # -- 6. Method C -- cluster diversity (if available) -----------------------
    if 'method_c' in results and 'diversity_stats' in results['method_c']:
        _plot_cluster_diversity(results['method_c']['diversity_stats'], output_dir)

    # -- 7. Per-query hit/miss heatmap across all 3 methods -------------------
    _plot_per_query_hits_heatmap(results, output_dir)

    # -- 8. Method B per-query detail (tokens + cumulative accuracy) ----------
    if 'method_b' in results:
        _plot_method_b_per_query(results['method_b'], output_dir)

    # -- 9. Method C per-query detail (tokens + diversity + cumulative acc) ---
    if 'method_c' in results:
        _plot_method_c_per_query(results['method_c'], output_dir)


def _plot_combined_accuracy(results, output_dir):
    labels = [
        'Fixed-512\n(Baseline)',
        'Element-based\n(Baseline)',
        'Method A\nMMR Selection',
        'Method B\nCompression',
        'Method C\nClustering',
    ]
    accs = [
        results.get('baseline_accuracy', {}).get('fixed-512', 0) * 100,
        results.get('baseline_accuracy', {}).get('element-based', 0) * 100,
        results.get('method_a', {}).get('accuracy', 0) * 100,
        results.get('method_b', {}).get('accuracy', 0) * 100,
        results.get('method_c', {}).get('accuracy', 0) * 100,
    ]
    colors = [
        COLORS['baseline_512'],
        COLORS['baseline_elem'],
        COLORS['method_a'],
        COLORS['method_b'],
        COLORS['method_c'],
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(labels, accs, color=colors, alpha=0.85, edgecolor='black', linewidth=1.3)
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Retrieval Accuracy -- All Methods Comparison', fontsize=14, fontweight='bold', pad=20)
    n = results.get('num_questions_tested', 0)
    ax.text(0.5, -0.14, f'(n={n} questions)', transform=ax.transAxes,
            ha='center', fontsize=10, fontstyle='italic', color='gray')
    ax.set_ylim(0, 115)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    for bar, acc in zip(bars, accs):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.subplots_adjust(bottom=0.22)
    out = f'{output_dir}/combined_accuracy_comparison.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {out}")


def _label_for_method(method):
    return {'method_a': 'Method A -- MMR Selection',
            'method_b': 'Method B -- Compression',
            'method_c': 'Method C -- Clustering'}.get(method, method)


def _color_for_method(method):
    return COLORS.get(method, '#888888')


def _plot_token_flow(method_results, method, output_dir):
    token_stats = method_results.get('token_stats', {})
    if not token_stats:
        return

    stages = ['Input\nDocuments', 'After\nChunking', 'After\nRetr.', 'After\nDedup/Compress', 'Final\nOutput']
    by_stage = token_stats.get('by_stage', {})
    total_chunk_tokens = token_stats.get('total_chunk_tokens', token_stats.get('total_input_tokens', 0))

    stage2_key = 'deduplication' if method in ('method_a', 'method_c') else 'compression_stage'
    tokens = [
        token_stats.get('total_input_tokens', 0),
        total_chunk_tokens,
        by_stage.get('retrieval', {}).get('output_tokens', 0),
        by_stage.get(stage2_key, {}).get('output_tokens', by_stage.get('reranking', {}).get('output_tokens', 0)),
        int(token_stats.get('avg_output_tokens_per_query', 0)),
    ]
    # Fill zeros with reasonable fallback
    tokens = [max(t, 10) for t in tokens]

    color = _color_for_method(method)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(stages, tokens, marker='o', linewidth=2.5, markersize=10, color=color)
    ax.fill_between(range(len(stages)), tokens, alpha=0.25, color=color)
    ax.set_ylabel('Token Count', fontsize=12, fontweight='bold')
    ax.set_title(f'Token Reduction Through Pipeline\n{_label_for_method(method)}',
                 fontsize=13, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')

    for i, (stage, tok) in enumerate(zip(stages, tokens)):
        ax.text(i, tok * 1.02, f'{tok:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.xticks(rotation=0)
    plt.tight_layout()
    out = f'{output_dir}/token_flow_{method}.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {out}")


def _plot_timing(timing, method, output_dir):
    if method == 'method_a':
        steps  = ['Multi-Strategy\nRetrieval', 'Deduplication', 'Cross-Encoder\nReranking', 'MMR\nSelection']
        keys   = ['retrieval_time', 'dedup_time', 'rerank_time', 'mmr_time']
    elif method == 'method_b':
        steps  = ['Multi-Strategy\nRetrieval', 'Cross-Encoder\nReranking', 'LLM\nCompression', 'Budget\nPacking']
        keys   = ['retrieval_time', 'rerank_time', 'compress_time', 'pack_time']
    else:  # method_c
        steps  = ['Multi-Strategy\nRetrieval', 'Deduplication', 'KMeans\nClustering', 'Cluster\nSelection']
        keys   = ['retrieval_time', 'dedup_time', 'cluster_time', 'select_time']

    times_raw = [float(timing.get(k, 0)) for k in keys]
    max_t = max(times_raw) if times_raw else 0
    if max_t > 0 and max_t < 0.1:
        times = [t * 1000 for t in times_raw]
        ylabel = 'Time (milliseconds)'
        fmt_unit = 'ms'
    else:
        times = times_raw
        ylabel = 'Time (seconds)'
        fmt_unit = 's'

    color = _color_for_method(method)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(steps, times, color=COLORS['stages'][:len(steps)], alpha=0.85,
                  edgecolor='black', linewidth=1.3)
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(f'Pipeline Step Timing\n{_label_for_method(method)}',
                 fontsize=13, fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    for bar, t in zip(bars, times):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h,
                f'{t:.3f}{fmt_unit}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    out = f'{output_dir}/timing_{method}.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {out}")


def _plot_chunk_distribution(dist, output_dir):
    strategies = list(dist.keys())
    counts = list(dist.values())

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(strategies, counts, color=COLORS['strategies'][:len(strategies)],
                   alpha=0.85, edgecolor='black', linewidth=1.3)
    ax.set_xlabel('Number of Chunks', fontsize=12, fontweight='bold')
    ax.set_title('Chunk Distribution by Strategy (shared across all methods)',
                 fontsize=13, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    for bar, count in zip(bars, counts):
        w = bar.get_width()
        ax.text(w + 0.5, bar.get_y() + bar.get_height() / 2.,
                f' {count}', ha='left', va='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    out = f'results/chunk_distribution.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {out}")


def _plot_method_comparison(results, output_dir):
    """Side-by-side bar chart: latency and avg output tokens per method."""
    methods = ['Method A\nMMR', 'Method B\nCompression', 'Method C\nClustering']
    keys = ['method_a', 'method_b', 'method_c']
    colors = [COLORS['method_a'], COLORS['method_b'], COLORS['method_c']]

    latencies = [results.get(k, {}).get('avg_latency_s', 0) for k in keys]
    out_tokens = [results.get(k, {}).get('token_stats', {}).get('avg_output_tokens_per_query', 0)
                  for k in keys]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Latency
    bars1 = ax1.bar(methods, latencies, color=colors, alpha=0.85, edgecolor='black', linewidth=1.3)
    ax1.set_ylabel('Average Latency (seconds)', fontsize=11, fontweight='bold')
    ax1.set_title('Avg Latency per Query', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    for bar, v in zip(bars1, latencies):
        ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                 f'{v:.2f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Output tokens
    bars2 = ax2.bar(methods, out_tokens, color=colors, alpha=0.85, edgecolor='black', linewidth=1.3)
    ax2.set_ylabel('Avg Output Tokens per Query', fontsize=11, fontweight='bold')
    ax2.set_title('Avg Output Tokens per Query', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    for bar, v in zip(bars2, out_tokens):
        ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                 f'{int(v):,}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    fig.suptitle('Method Comparison -- Efficiency Metrics', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = f'{output_dir}/method_comparison_efficiency.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {out}")


def _plot_cluster_diversity(diversity_stats, output_dir):
    """Bar chart of avg inter-cluster similarity per query (Method C only)."""
    if not diversity_stats:
        return
    queries = list(range(1, len(diversity_stats) + 1))
    mean_sims = [d.get('mean_sim', 0) for d in diversity_stats]

    fig, ax = plt.subplots(figsize=(max(10, len(queries) // 2), 5))
    ax.bar(queries, mean_sims, color=COLORS['method_c'], alpha=0.85, edgecolor='black', linewidth=1.2)
    ax.axhline(y=sum(mean_sims) / len(mean_sims), color='red', linestyle='--',
               linewidth=1.5, label=f'Avg: {sum(mean_sims)/len(mean_sims):.3f}')
    ax.set_xlabel('Query Index', fontsize=11, fontweight='bold')
    ax.set_ylabel('Mean Inter-Chunk Cosine Similarity', fontsize=11, fontweight='bold')
    ax.set_title('Method C -- Cluster Diversity per Query\n(lower = more diverse)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    out = f'{output_dir}/cluster_diversity_method_c.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {out}")


def _plot_per_query_hits_heatmap(results, output_dir):
    """Heatmap: which method scored a hit per question (green=hit, red=miss)."""
    import numpy as np
    det_a = results.get('method_a', {}).get('detailed_results', [])
    det_b = results.get('method_b', {}).get('detailed_results', [])
    det_c = results.get('method_c', {}).get('detailed_results', [])

    n = max(len(det_a), len(det_b), len(det_c))
    if n == 0:
        return

    hits_a = [1 if r.get('metrics', {}).get('recall@10', 0) > 0 else 0 for r in det_a]
    hits_b = [r.get('hit', 0) for r in det_b]
    hits_c = [r.get('hit', 0) for r in det_c]

    # Pad to same length in case methods have different counts
    hits_a += [0] * (n - len(hits_a))
    hits_b += [0] * (n - len(hits_b))
    hits_c += [0] * (n - len(hits_c))

    data = np.array([hits_a, hits_b, hits_c], dtype=float)

    fig_w = max(16, n // 3)
    fig, ax = plt.subplots(figsize=(fig_w, 3.5))
    im = ax.imshow(data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1,
                   interpolation='nearest')

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['Method A\n(MMR)', 'Method B\n(Compress)', 'Method C\n(Cluster)'],
                       fontsize=11)
    ax.set_xlabel('Question Index', fontsize=11, fontweight='bold')
    ax.set_title('Per-Query Hit / Miss Heatmap — All Methods\n(Green = Hit, Red = Miss)',
                 fontsize=13, fontweight='bold', pad=15)

    cbar = plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Miss', 'Hit'])

    # Annotate totals on the right
    totals = [sum(hits_a), sum(hits_b), sum(hits_c)]
    for row_idx, total in enumerate(totals):
        ax.text(n + 0.3, row_idx, f'{total}/{n}', va='center', fontsize=9,
                fontweight='bold', color='black')

    plt.tight_layout()
    out = f'{output_dir}/per_query_hits_heatmap.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {out}")


def _plot_method_b_per_query(method_b_results, output_dir):
    """Method B: per-query output tokens (hit/miss coloured) + running accuracy."""
    det = method_b_results.get('detailed_results', [])
    if not det:
        return

    n       = len(det)
    queries = list(range(1, n + 1))
    tokens  = [r.get('output_tokens', 0) for r in det]
    hits    = [r.get('hit', 0) for r in det]
    cum_acc = [sum(hits[:i + 1]) / (i + 1) * 100 for i in range(n)]
    avg_tok = sum(tokens) / max(n, 1)
    final_acc = method_b_results.get('accuracy', 0) * 100

    bar_colors = [COLORS['method_b'] if h else '#DC2626' for h in hits]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(14, n // 3), 8),
                                    sharex=True)
    fig.suptitle('Method B (Query-Aware Compression) — Per-Query Analysis',
                 fontsize=14, fontweight='bold', y=1.01)

    # Top: output tokens per query, coloured by hit/miss
    ax1.bar(queries, tokens, color=bar_colors, alpha=0.85, edgecolor='black', linewidth=0.8)
    ax1.axhline(y=avg_tok, color='black', linestyle='--', linewidth=1.5,
                label=f'Avg: {int(avg_tok):,} tokens')
    ax1.set_ylabel('Output Tokens', fontsize=11, fontweight='bold')
    ax1.set_title('Output Tokens per Query  (green = hit, red = miss)',
                  fontsize=11, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.legend(fontsize=9)

    # Bottom: cumulative accuracy
    ax2.plot(queries, cum_acc, marker='o', linewidth=2, markersize=4,
             color=COLORS['method_b'], label='Running accuracy')
    ax2.fill_between(queries, cum_acc, alpha=0.2, color=COLORS['method_b'])
    ax2.axhline(y=final_acc, color='red', linestyle='--', linewidth=1.5,
                label=f'Final: {final_acc:.1f}%')
    ax2.set_xlabel('Question Index', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Cumulative Accuracy (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Running Accuracy', fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 110)
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.legend(fontsize=9)

    plt.tight_layout()
    out = f'{output_dir}/method_b_per_query_details.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {out}")


def _plot_method_c_per_query(method_c_results, output_dir):
    """Method C: per-query output tokens, diversity, and running accuracy."""
    det = method_c_results.get('detailed_results', [])
    if not det:
        return

    n          = len(det)
    queries    = list(range(1, n + 1))
    tokens     = [r.get('output_tokens', 0) for r in det]
    hits       = [r.get('hit', 0) for r in det]
    diversity  = [r.get('diversity', {}).get('mean_sim', 0) for r in det]
    cum_acc    = [sum(hits[:i + 1]) / (i + 1) * 100 for i in range(n)]
    avg_tok    = sum(tokens) / max(n, 1)
    avg_div    = sum(diversity) / max(n, 1)
    final_acc  = method_c_results.get('accuracy', 0) * 100

    bar_colors = [COLORS['method_c'] if h else '#DC2626' for h in hits]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(max(14, n // 3), 11),
                                         sharex=True)
    fig.suptitle('Method C (Cluster-Based Aggregation) — Per-Query Analysis',
                 fontsize=14, fontweight='bold', y=1.01)

    # Top: output tokens per query
    ax1.bar(queries, tokens, color=bar_colors, alpha=0.85, edgecolor='black', linewidth=0.8)
    ax1.axhline(y=avg_tok, color='black', linestyle='--', linewidth=1.5,
                label=f'Avg: {int(avg_tok):,} tokens')
    ax1.set_ylabel('Output Tokens', fontsize=11, fontweight='bold')
    ax1.set_title('Output Tokens per Query  (amber = hit, red = miss)',
                  fontsize=11, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.legend(fontsize=9)

    # Middle: cluster diversity per query
    ax2.bar(queries, diversity, color=COLORS['method_c'], alpha=0.75,
            edgecolor='black', linewidth=0.8)
    ax2.axhline(y=avg_div, color='red', linestyle='--', linewidth=1.5,
                label=f'Avg: {avg_div:.3f}')
    ax2.set_ylabel('Mean Inter-Chunk\nCosine Similarity', fontsize=10, fontweight='bold')
    ax2.set_title('Cluster Diversity per Query  (lower = more diverse)',
                  fontsize=11, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.legend(fontsize=9)

    # Bottom: cumulative accuracy
    ax3.plot(queries, cum_acc, marker='o', linewidth=2, markersize=4,
             color=COLORS['method_c'], label='Running accuracy')
    ax3.fill_between(queries, cum_acc, alpha=0.2, color=COLORS['method_c'])
    ax3.axhline(y=final_acc, color='red', linestyle='--', linewidth=1.5,
                label=f'Final: {final_acc:.1f}%')
    ax3.set_xlabel('Question Index', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Cumulative Accuracy (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Running Accuracy', fontsize=11, fontweight='bold')
    ax3.set_ylim(0, 110)
    ax3.grid(alpha=0.3, linestyle='--')
    ax3.legend(fontsize=9)

    plt.tight_layout()
    out = f'{output_dir}/method_c_per_query_details.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {out}")


# ---------------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------------
def print_box(title, content, width=72, color='blue'):
    colors = {
        'blue': '\033[94m', 'green': '\033[92m', 'yellow': '\033[93m',
        'red': '\033[91m',  'cyan': '\033[96m',  'magenta': '\033[95m',
        'white': '\033[97m', 'reset': '\033[0m'
    }
    c = colors.get(color, colors['blue'])
    r = colors['reset']
    border = '=' * width
    print(f"\n{c}{border}{r}")
    print(f"{c}|{r} {title:^{width-4}} {c}|{r}")
    print(f"{c}{border}{r}")
    lines = content if isinstance(content, list) else content.split('\n')
    for line in lines:
        print(f"{c}|{r} {line:<{width-4}} {c}|{r}")
    print(f"{c}{border}{r}\n")


def print_header(method_label):
    print(f"\n{'-' * 72}")
    print(f"  >>  {method_label}")
    print(f"{'-' * 72}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Smart Aggregation for RAG - Enhanced Demo (3-method comparison)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--financebench', type=str, default=None)
    parser.add_argument('--max-docs', type=int, default=5,
                        help='Maximum documents to load (default: 5)')
    parser.add_argument('--max-questions', type=int, default=15,
                        help='Maximum questions to test (default: 15)')
    parser.add_argument('--no-cache', action='store_true')
    parser.add_argument('--no-graphs', action='store_true')
    return parser.parse_args()


def load_data(args):
    if args.financebench:
        print_box("USING FINANCEBENCH DATASET",
                  f"Path: {args.financebench}\nMax docs: {'ALL' if args.max_docs == 0 else args.max_docs}",
                  color='cyan')
        try:
            from financebench_loader import get_financebench_data
            data = get_financebench_data(
                financebench_path=args.financebench,
                max_docs=None if args.max_docs == 0 else args.max_docs,
                max_questions=None if args.max_questions == 0 else args.max_questions,
                use_cache=not args.no_cache
            )
            return data, 'FinanceBench'
        except Exception as e:
            print(f"\nERROR loading FinanceBench: {e}\n")
            sys.exit(1)
    else:
        print_box("USING SAMPLE DATA",
                  f"5 financial documents, 15 questions", color='cyan')
        from sample_data import get_sample_data
        raw = get_sample_data()
        # Respect --max-docs / --max-questions even for sample data
        docs = raw['documents']
        qs   = raw['questions']
        if args.max_docs and args.max_docs > 0:
            doc_keys = list(docs.keys())[:args.max_docs]
            docs = {k: docs[k] for k in doc_keys}
            qs   = [q for q in qs if q['doc_id'] in docs]
        if args.max_questions and args.max_questions > 0:
            qs = qs[:args.max_questions]
        return {'documents': docs, 'questions': qs}, 'Sample'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_arguments()

    print("""
+======================================================================+
|                                                                      |
|   SMART AGGREGATION FOR RAG -- 3-METHOD COMPARISON DEMO             |
|                                                                      |
|   Method A: MMR Selection     Method B: Compression                 |
|   Method C: KMeans Clustering                                        |
|                                                                      |
|   [+] Token Counting  [+] Visual Graphs  [+] Per-Method Charts      |
|                                                                      |
+======================================================================+
""")

    token_counter = TokenCounter()

    # -- Load data ------------------------------------------------------------
    data, data_source = load_data(args)
    documents = data['documents']
    questions = data['questions']

    total_doc_text = ' '.join(documents.values())
    input_tokens = token_counter.count_tokens(total_doc_text)

    print_box("DATA LOADED", [
        f"Documents : {len(documents)}",
        f"Questions : {len(questions)}",
        f"Total input tokens : {input_tokens:,}",
        f"Avg tokens/doc    : {input_tokens // max(len(documents), 1):,}"
    ], color='green')

    # -- Chunking -------------------------------------------------------------
    print_box("CREATING CHUNKS (4 STRATEGIES)", [], color='yellow')

    chunkers = {
        'fixed-128':    BaseChunker(chunk_size=128,  overlap=0),
        'fixed-256':    BaseChunker(chunk_size=256,  overlap=0),
        'fixed-512':    BaseChunker(chunk_size=512,  overlap=0),
        'element-based': ElementChunker(max_chunk_size=2048, merge_small=True),
    }

    all_chunks = []
    chunk_distribution = {}

    for doc_id, doc_text in documents.items():
        print(f"\n[DOC] Processing {doc_id}...")
        for strategy_name, chunker in chunkers.items():
            chunks = chunker.chunk(doc_text, doc_id)
            all_chunks.extend(chunks)
            chunk_distribution[strategy_name] = (
                chunk_distribution.get(strategy_name, 0) + len(chunks)
            )
            chunk_tokens = sum(token_counter.count_tokens(c['text']) for c in chunks)
            print(f"   {strategy_name:15s}: {len(chunks):4d} chunks, {chunk_tokens:6,} tokens")

    total_chunk_tokens = sum(token_counter.count_tokens(c['text']) for c in all_chunks)

    print_box("CHUNKING COMPLETE", [
        f"Total chunks          : {len(all_chunks)}",
        f"Total tokens in chunks: {total_chunk_tokens:,}",
        f"Token expansion       : {total_chunk_tokens / max(input_tokens, 1):.2f}x",
        f"Avg tokens/chunk      : {total_chunk_tokens // max(len(all_chunks), 1):,}"
    ], color='green')

    # -- Embedding & indexing --------------------------------------------------
    print_box("EMBEDDING AND INDEXING", ["This may take 1-5 minutes..."], color='yellow')

    embedder = Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2")

    texts = [c['text'] for c in all_chunks]
    embeddings = embedder.embed_batch(texts, batch_size=32, show_progress=True)

    vector_store = VectorStore(embedding_dim=embedder.embedding_dim)
    vector_store.add_chunks(all_chunks, embeddings)
    print("[OK] Indexing complete!\n")

    # -- Cross-encoder ---------------------------------------------------------
    print_box("LOADING CROSS-ENCODER", ["Shared across all methods"], color='yellow')
    cross_encoder = CrossEncoderReranker(model_name="BAAI/bge-reranker-base")

    # -- Shared pipeline config ------------------------------------------------
    shared_config = {
        'top_k_per_strategy': 30,
        'dedup_threshold': 0.85,
        'rerank_top_k': 20,
        'mmr_lambda': 0.8,
        'final_k': 10,
    }

    # --------------------------------------------------------------------------
    #  Method A -- Smart Aggregation (MMR)
    # --------------------------------------------------------------------------
    print_box("METHOD A: SMART AGGREGATION (MMR SELECTION)", [], color='magenta')

    smart_agg = SmartAggregation(
        embedder=embedder,
        vector_store=vector_store,
        cross_encoder=cross_encoder,
        config=shared_config,
    )

    evaluator = RAGEvaluator()
    results_a = []
    timing_a  = {'retrieval_time': 0.0, 'dedup_time': 0.0, 'rerank_time': 0.0, 'mmr_time': 0.0}
    total_out_a = 0

    for i, q_data in enumerate(questions, 1):
        print(f"\n{'-'*70}\nQ {i}/{len(questions)}: {q_data['question']}")
        final_chunks = smart_agg.retrieve(q_data['question'], verbose=True)

        out_tok = _count_output_tokens_method(final_chunks, token_counter)
        total_out_a += out_tok

        stats = smart_agg.get_statistics()
        timing_a['retrieval_time'] += float(stats.get('step1_time', 0))
        timing_a['dedup_time']     += float(stats.get('step2_time', 0))
        timing_a['rerank_time']    += float(stats.get('step3_time', 0))
        timing_a['mmr_time']       += float(stats.get('step4_time', 0))

        ground_truth = [q_data['doc_id']]
        metrics = evaluator.evaluate_retrieval(final_chunks, ground_truth, k_values=[5, 10])

        print(f"   Precision@5: {metrics['precision@5']:.3f}  "
              f"Recall@5: {metrics['recall@5']:.3f}  "
              f"NDCG@5: {metrics['ndcg@5']:.3f}  "
              f"OutputTokens: {out_tok}")

        results_a.append({'question': q_data['question'], 'output_tokens': out_tok,
                           'metrics': metrics, 'timing': stats,
                           'latency': stats.get('total_time', 0)})

    avg_out_a = total_out_a / max(len(questions), 1)
    smart_acc_a = sum(1 for r in results_a if r['metrics'].get('recall@10', 0) > 0) / max(len(questions), 1)
    avg_lat_a   = sum(r['latency'] for r in results_a) / max(len(questions), 1)

    # --------------------------------------------------------------------------
    #  Method B -- Query-Aware Compression
    # --------------------------------------------------------------------------
    print_box("METHOD B: QUERY-AWARE COMPRESSION", ["Extractive backend (no API required)"], color='magenta')

    method_b = QueryAwareCompression(
        embedder=embedder,
        vector_store=vector_store,
        cross_encoder=cross_encoder,
        config={
            'top_k_per_strategy': shared_config['top_k_per_strategy'],
            'rerank_top_k':       shared_config['rerank_top_k'],
            'token_budget':       4096,
            'min_compressed_tokens': 10,
            'compression_threshold': 0.45,
        },
    )

    results_b  = []
    timing_b   = {'retrieval_time': 0.0, 'rerank_time': 0.0, 'compress_time': 0.0, 'pack_time': 0.0}
    total_out_b = 0

    for i, q_data in enumerate(questions, 1):
        print(f"\n{'-'*70}\nQ {i}/{len(questions)}: {q_data['question']}")
        packed = method_b.retrieve(q_data['question'], verbose=True)

        out_tok = _count_output_tokens_method(packed, token_counter)
        total_out_b += out_tok

        stats = method_b.get_statistics()
        timing_b['retrieval_time'] += float(stats.get('stage1_time', 0))
        timing_b['rerank_time']    += float(stats.get('stage2_time', 0))
        timing_b['compress_time']  += float(stats.get('stage3_time', 0))
        timing_b['pack_time']      += float(stats.get('stage4_time', 0))

        # Evaluate: ground truth doc_id must appear among packed doc_ids
        retrieved_docs_b = [_get_doc_id(c) for c in packed]
        correct_b = 1 if q_data['doc_id'] in retrieved_docs_b else 0

        print(f"   Chunks packed: {len(packed)}  "
              f"OutputTokens: {out_tok}  "
              f"Hit: {'Y' if correct_b else 'N'}")

        results_b.append({'question': q_data['question'], 'output_tokens': out_tok,
                           'hit': correct_b, 'latency': stats.get('total_time', 0)})

    avg_out_b = total_out_b / max(len(questions), 1)
    smart_acc_b = sum(r['hit'] for r in results_b) / max(len(questions), 1)
    avg_lat_b   = sum(r['latency'] for r in results_b) / max(len(questions), 1)

    # --------------------------------------------------------------------------
    #  Method C -- Cluster-Based Aggregation
    # --------------------------------------------------------------------------
    print_box("METHOD C: CLUSTER-BASED AGGREGATION (KMeans)", [], color='magenta')

    method_c = ClusterBasedAggregation(
        embedder=embedder,
        vector_store=vector_store,
        cross_encoder=cross_encoder,
        config={
            'top_k_per_strategy': shared_config['top_k_per_strategy'],
            'dedup_threshold':    shared_config['dedup_threshold'],
            'n_clusters':         shared_config['final_k'],
            'use_cross_encoder':  True,
        },
    )

    results_c      = []
    timing_c       = {'retrieval_time': 0.0, 'dedup_time': 0.0, 'cluster_time': 0.0, 'select_time': 0.0}
    diversity_stats = []
    total_out_c     = 0

    for i, q_data in enumerate(questions, 1):
        print(f"\n{'-'*70}\nQ {i}/{len(questions)}: {q_data['question']}")
        cluster_results = method_c.retrieve(q_data['question'], verbose=True)

        out_tok = _count_output_tokens_method(cluster_results, token_counter)
        total_out_c += out_tok

        stats = method_c.get_statistics()
        timing_c['retrieval_time'] += float(stats.get('stage1_time', 0))
        timing_c['dedup_time']     += float(stats.get('stage2_time', 0))
        timing_c['cluster_time']   += float(stats.get('stage3_time', 0))
        timing_c['select_time']    += float(stats.get('stage4_time', 0))

        diversity = compute_inter_cluster_diversity(cluster_results)
        diversity_stats.append(diversity)

        # Evaluate
        retrieved_docs_c = [_get_doc_id(r) for r in cluster_results]
        correct_c = 1 if q_data['doc_id'] in retrieved_docs_c else 0

        print(f"   Clusters: {len(cluster_results)}  "
              f"OutputTokens: {out_tok}  "
              f"Diversity(mean_sim): {diversity['mean_sim']:.3f}  "
              f"Hit: {'Y' if correct_c else 'N'}")

        results_c.append({'question': q_data['question'], 'output_tokens': out_tok,
                           'hit': correct_c, 'latency': stats.get('total_time', 0),
                           'diversity': diversity})

    avg_out_c = total_out_c / max(len(questions), 1)
    smart_acc_c = sum(r['hit'] for r in results_c) / max(len(questions), 1)
    avg_lat_c   = sum(r['latency'] for r in results_c) / max(len(questions), 1)

    # --------------------------------------------------------------------------
    #  Baselines
    # --------------------------------------------------------------------------
    print_box("BASELINE COMPARISON", ["Single-strategy dense retrieval"], color='magenta')

    baseline_results = {}
    for strategy in ['fixed-512', 'element-based']:
        print(f"Testing baseline: {strategy}...")
        correct = 0
        for q_data in questions:
            query_emb = embedder.embed_single(q_data['question'])
            res = vector_store.search(query_emb, k=10, strategy_filter=strategy)
            if q_data['doc_id'] in [r.chunk.doc_id for r in res[:10]]:
                correct += 1
        accuracy = correct / max(len(questions), 1)
        baseline_results[strategy] = accuracy
        print(f"  [OK] {strategy}: {accuracy:.1%}\n")

    # --------------------------------------------------------------------------
    #  Print final summary
    # --------------------------------------------------------------------------
    best_baseline = max(baseline_results.values()) if baseline_results else 0

    def impr(acc):
        if best_baseline > 0:
            p = (acc - best_baseline) / best_baseline * 100
            return f"{'+' if p >= 0 else ''}{p:.1f}% vs best baseline"
        return 'N/A'

    print_box("[RESULTS] FINAL RESULTS", [
        f"Fixed-512 (baseline):      {baseline_results.get('fixed-512',    0):.1%}",
        f"Element-based (baseline):  {baseline_results.get('element-based', 0):.1%}",
        "",
        f"Method A -- MMR Selection:  {smart_acc_a:.1%}  ({impr(smart_acc_a)})",
        f"Method B -- Compression:    {smart_acc_b:.1%}  ({impr(smart_acc_b)})",
        f"Method C -- Clustering:     {smart_acc_c:.1%}  ({impr(smart_acc_c)})",
    ], color='green')

    print_box("[TIMING] LATENCY SUMMARY (avg per query)", [
        f"Method A -- MMR:         {avg_lat_a:.3f}s",
        f"Method B -- Compression: {avg_lat_b:.3f}s",
        f"Method C -- Clustering:  {avg_lat_c:.3f}s",
    ], color='cyan')

    print_box("[TOKENS] OUTPUT TOKEN SUMMARY (avg per query)", [
        f"Input tokens (all docs):   {input_tokens:,}",
        f"Total chunk tokens:        {total_chunk_tokens:,}",
        f"Method A avg output:       {int(avg_out_a):,}",
        f"Method B avg output:       {int(avg_out_b):,}",
        f"Method C avg output:       {int(avg_out_c):,}",
    ], color='cyan')

    # --------------------------------------------------------------------------
    #  Build per-method token stats for graphs
    # --------------------------------------------------------------------------
    avg_tok_chunk = total_chunk_tokens // max(len(all_chunks), 1)
    n_strats = 4
    top_k   = shared_config['top_k_per_strategy']
    rerank_k = shared_config['rerank_top_k']

    def _token_stats_block(avg_out, label):
        return {
            'total_input_tokens':   input_tokens,
            'total_chunk_tokens':   total_chunk_tokens,
            'avg_output_tokens_per_query': avg_out,
            'compression_ratio': total_chunk_tokens / max(avg_out, 1),
            'by_stage': {
                'retrieval':        {'output_tokens': n_strats * top_k * avg_tok_chunk},
                'deduplication':    {'output_tokens': int(n_strats * top_k * 0.6) * avg_tok_chunk},
                'compression_stage': {'output_tokens': rerank_k * avg_tok_chunk},
                'reranking':        {'output_tokens': rerank_k * avg_tok_chunk},
            }
        }

    # --------------------------------------------------------------------------
    #  Save JSON results
    # --------------------------------------------------------------------------
    os.makedirs('results', exist_ok=True)

    combined_results = {
        'dataset': data_source,
        'num_documents': len(documents),
        'num_questions_tested': len(questions),
        'total_chunks': len(all_chunks),
        'baseline_accuracy': baseline_results,
        'method_a': {
            'accuracy':    smart_acc_a,
            'avg_latency_s': avg_lat_a,
            'token_stats': _token_stats_block(avg_out_a, 'A'),
            'timing_stats': timing_a,
            'chunk_distribution': chunk_distribution,
            'detailed_results': results_a,
        },
        'method_b': {
            'accuracy':    smart_acc_b,
            'avg_latency_s': avg_lat_b,
            'token_stats': _token_stats_block(avg_out_b, 'B'),
            'timing_stats': timing_b,
            'detailed_results': results_b,
        },
        'method_c': {
            'accuracy':    smart_acc_c,
            'avg_latency_s': avg_lat_c,
            'token_stats': _token_stats_block(avg_out_c, 'C'),
            'timing_stats': timing_c,
            'diversity_stats': diversity_stats,
            'detailed_results': results_c,
        },
    }

    out_file = f'results/experiment_results_{data_source.lower()}_three_methods.json'
    with open(out_file, 'w') as f:
        json.dump(combined_results, f, indent=2, default=str)
    print(f"[OK] Results saved to {out_file}\n")

    # --------------------------------------------------------------------------
    #  Visualisations
    # --------------------------------------------------------------------------
    if not args.no_graphs:
        print_box("[GRAPHS] CREATING VISUALISATIONS", ["Generating separate + combined charts..."], color='yellow')
        create_visualizations(combined_results)

    print("""
+======================================================================+
|                    DEMO COMPLETE!                                    |
|                                                                      |
|  Check results/ folder for:                                          |
|  - combined_accuracy_comparison.png  all 5 bars                     |
|  - token_flow_method_a/b/c.png       per-method token reduction      |
|  - timing_method_a/b/c.png           per-method step timing          |
|  - chunk_distribution.png            strategy distribution           |
|  - method_comparison_efficiency.png  latency & token comparison      |
|  - cluster_diversity_method_c.png    KMeans diversity (Method C)     |
|  - per_query_hits_heatmap.png        hit/miss heatmap all 3 methods  |
|  - method_b_per_query_details.png    B: tokens + running accuracy    |
|  - method_c_per_query_details.png    C: tokens + diversity + acc     |
|  - experiment_results_*.json         full numeric results             |
+======================================================================+
""")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:  # noqa
        print("\n\n[WARN] Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
