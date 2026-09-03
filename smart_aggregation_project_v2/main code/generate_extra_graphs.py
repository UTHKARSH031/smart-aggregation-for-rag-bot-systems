"""
Generate Extra Graphs for Method B and C
=========================================
Run this AFTER demo_enhanced.py has completed.
Reads the saved JSON results and produces 3 new charts:

  - per_query_hits_heatmap.png        (hit/miss heatmap all 3 methods)
  - method_b_per_query_details.png    (B: tokens + running accuracy)
  - method_c_per_query_details.png    (C: tokens + diversity + accuracy)

Usage:
    python generate_extra_graphs.py
    python generate_extra_graphs.py --results results/experiment_results_financebench_three_methods.json
"""

import argparse
import json
import os
import glob
import sys

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("ERROR: matplotlib / numpy not installed. Run: pip install matplotlib numpy")
    sys.exit(1)

COLORS = {
    'method_a': '#3B82F6',
    'method_b': '#10B981',
    'method_c': '#F59E0B',
}


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _plot_per_query_hits_heatmap(results, output_dir):
    """Heatmap: which method scored a hit per question (green=hit, red=miss)."""
    det_a = results.get('method_a', {}).get('detailed_results', [])
    det_b = results.get('method_b', {}).get('detailed_results', [])
    det_c = results.get('method_c', {}).get('detailed_results', [])

    n = max(len(det_a), len(det_b), len(det_c))
    if n == 0:
        print("  [SKIP] No detailed_results found for heatmap.")
        return

    hits_a = [1 if r.get('metrics', {}).get('recall@10', 0) > 0 else 0 for r in det_a]
    hits_b = [r.get('hit', 0) for r in det_b]
    hits_c = [r.get('hit', 0) for r in det_c]

    # Pad to same length
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

    totals = [sum(hits_a), sum(hits_b), sum(hits_c)]
    for row_idx, total in enumerate(totals):
        ax.text(n + 0.3, row_idx, f'{total}/{n}', va='center', fontsize=9,
                fontweight='bold', color='black')

    plt.tight_layout()
    out = f'{output_dir}/per_query_hits_heatmap.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {out}")


def _plot_method_a_per_query(method_a_results, output_dir):
    """Method A: per-query output tokens (hit/miss coloured) + running accuracy."""
    det = method_a_results.get('detailed_results', [])
    if not det:
        print("  [SKIP] No detailed_results for Method A.")
        return

    n         = len(det)
    queries   = list(range(1, n + 1))
    tokens    = [r.get('output_tokens', 0) for r in det]
    hits      = [r.get('hit', 1 if r.get('metrics', {}).get('recall@10', 0) > 0 else 0) for r in det]
    cum_acc   = [sum(hits[:i + 1]) / (i + 1) * 100 for i in range(n)]
    avg_tok   = sum(tokens) / max(n, 1)
    final_acc = method_a_results.get('accuracy', 0) * 100

    bar_colors = [COLORS['method_a'] if h else '#DC2626' for h in hits]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(14, n // 3), 8), sharex=True)
    fig.suptitle('Method A (Smart Aggregation MMR) — Per-Query Analysis',
                 fontsize=14, fontweight='bold', y=1.01)

    # Top: output tokens per query, coloured by hit/miss
    ax1.bar(queries, tokens, color=bar_colors, alpha=0.85, edgecolor='black', linewidth=0.8)
    ax1.axhline(y=avg_tok, color='black', linestyle='--', linewidth=1.5,
                label=f'Avg: {int(avg_tok):,} tokens')
    ax1.set_ylabel('Output Tokens', fontsize=11, fontweight='bold')
    ax1.set_title('Output Tokens per Query  (blue = hit, red = miss)',
                  fontsize=11, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.legend(fontsize=9)

    # Bottom: cumulative accuracy
    ax2.plot(queries, cum_acc, marker='o', linewidth=2, markersize=4,
             color=COLORS['method_a'], label='Running accuracy')
    ax2.fill_between(queries, cum_acc, alpha=0.2, color=COLORS['method_a'])
    ax2.axhline(y=final_acc, color='red', linestyle='--', linewidth=1.5,
                label=f'Final: {final_acc:.1f}%')
    ax2.set_xlabel('Question Index', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Cumulative Accuracy (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Running Accuracy', fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 110)
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.legend(fontsize=9)

    plt.tight_layout()
    out = f'{output_dir}/method_a_per_query_details.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {out}")


def _plot_method_b_per_query(method_b_results, output_dir):
    """Method B: per-query output tokens (hit/miss coloured) + running accuracy."""
    det = method_b_results.get('detailed_results', [])
    if not det:
        print("  [SKIP] No detailed_results for Method B.")
        return

    n         = len(det)
    queries   = list(range(1, n + 1))
    tokens    = [r.get('output_tokens', 0) for r in det]
    hits      = [r.get('hit', 0) for r in det]
    cum_acc   = [sum(hits[:i + 1]) / (i + 1) * 100 for i in range(n)]
    avg_tok   = sum(tokens) / max(n, 1)
    final_acc = method_b_results.get('accuracy', 0) * 100

    bar_colors = [COLORS['method_b'] if h else '#DC2626' for h in hits]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(14, n // 3), 8), sharex=True)
    fig.suptitle('Method B (Query-Aware Compression) — Per-Query Analysis',
                 fontsize=14, fontweight='bold', y=1.01)

    ax1.bar(queries, tokens, color=bar_colors, alpha=0.85, edgecolor='black', linewidth=0.8)
    ax1.axhline(y=avg_tok, color='black', linestyle='--', linewidth=1.5,
                label=f'Avg: {int(avg_tok):,} tokens')
    ax1.set_ylabel('Output Tokens', fontsize=11, fontweight='bold')
    ax1.set_title('Output Tokens per Query  (green = hit, red = miss)',
                  fontsize=11, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.legend(fontsize=9)

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
        print("  [SKIP] No detailed_results for Method C.")
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

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(max(14, n // 3), 11), sharex=True)
    fig.suptitle('Method C (Cluster-Based Aggregation) — Per-Query Analysis',
                 fontsize=14, fontweight='bold', y=1.01)

    ax1.bar(queries, tokens, color=bar_colors, alpha=0.85, edgecolor='black', linewidth=0.8)
    ax1.axhline(y=avg_tok, color='black', linestyle='--', linewidth=1.5,
                label=f'Avg: {int(avg_tok):,} tokens')
    ax1.set_ylabel('Output Tokens', fontsize=11, fontweight='bold')
    ax1.set_title('Output Tokens per Query  (amber = hit, red = miss)',
                  fontsize=11, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.legend(fontsize=9)

    ax2.bar(queries, diversity, color=COLORS['method_c'], alpha=0.75,
            edgecolor='black', linewidth=0.8)
    ax2.axhline(y=avg_div, color='red', linestyle='--', linewidth=1.5,
                label=f'Avg: {avg_div:.3f}')
    ax2.set_ylabel('Mean Inter-Chunk\nCosine Similarity', fontsize=10, fontweight='bold')
    ax2.set_title('Cluster Diversity per Query  (lower = more diverse)',
                  fontsize=11, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.legend(fontsize=9)

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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Generate extra graphs from saved JSON results')
    parser.add_argument('--results', type=str, default=None,
                        help='Path to experiment_results_*.json (auto-detected if omitted)')
    args = parser.parse_args()

    # Auto-detect JSON if not specified
    if args.results:
        json_path = args.results
    else:
        matches = glob.glob('results/experiment_results_*.json')
        if not matches:
            print("ERROR: No results JSON found in results/. Run demo_enhanced.py first.")
            sys.exit(1)
        json_path = sorted(matches)[-1]  # pick most recent

    print(f"\nLoading results from: {json_path}")
    with open(json_path, 'r') as f:
        results = json.load(f)

    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nDataset  : {results.get('dataset', '?')}")
    print(f"Docs     : {results.get('num_documents', '?')}")
    print(f"Questions: {results.get('num_questions_tested', '?')}")
    print(f"\nGenerating extra graphs...\n")

    _plot_per_query_hits_heatmap(results, output_dir)
    _plot_method_a_per_query(results.get('method_a', {}), output_dir)
    _plot_method_b_per_query(results.get('method_b', {}), output_dir)
    _plot_method_c_per_query(results.get('method_c', {}), output_dir)

    print("\nDone! New graphs saved to results/")


if __name__ == '__main__':
    main()
