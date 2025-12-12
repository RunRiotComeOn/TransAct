"""
Visualize vocabulary analysis results - comparing baseline vs intervention
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import sys

# Configuration
RESULTS_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./vocabulary_analysis_results")

def load_all_results(results_dir):
    """Load all vocabulary analysis results"""
    results = []

    for dir_path in results_dir.iterdir():
        if not dir_path.is_dir():
            continue

        summary_file = dir_path / "vocabulary_analysis_summary.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                data = json.load(f)
                results.append(data)
                print(f"✓ Loaded: {data['config']['steering_type']} × {data['config']['dataset']}")

    return results

def plot_frequency_comparison(results):
    """Plot frequency comparison between baseline and intervention"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = [
        ('aggregate_causal_freq_per_1k', 'Causal Words'),
        ('aggregate_logical_freq_per_1k', 'Logical Words'),
        ('aggregate_reasoning_freq_per_1k', 'Total Reasoning')
    ]

    for ax, (metric, title) in zip(axes, metrics):
        labels = []
        baseline_vals = []
        intervention_vals = []

        for result in results:
            config = result['config']
            label = f"{config['steering_type'].split('_')[0][:6]}\n{config['dataset'][:8]}"
            labels.append(label)
            baseline_vals.append(result['baseline'][metric])
            intervention_vals.append(result['intervention'][metric])

        x = np.arange(len(labels))
        width = 0.35

        bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline', color='#3498db', alpha=0.8)
        bars2 = ax.bar(x + width/2, intervention_vals, width, label='Intervention', color='#e74c3c', alpha=0.8)

        ax.set_ylabel('Frequency per 1000 words')
        ax.set_title(f'{title} Frequency', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=7)
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig('visualization/vocab_frequency_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: vocab_frequency_comparison.png")

def plot_improvement_summary(results):
    """Plot improvement percentages"""
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = []
    causal_impr = []
    logical_impr = []
    reasoning_impr = []

    for result in results:
        config = result['config']
        label = f"{config['steering_type'].replace('_alpha5.0_7b', '')[:20]}\n{config['dataset']}"
        labels.append(label)
        causal_impr.append(result['improvements']['causal_improvement_pct'])
        logical_impr.append(result['improvements']['logical_improvement_pct'])
        reasoning_impr.append(result['improvements']['reasoning_improvement_pct'])

    x = np.arange(len(labels))
    width = 0.25

    bars1 = ax.bar(x - width, causal_impr, width, label='Causal', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x, logical_impr, width, label='Logical', color='#9b59b6', alpha=0.8)
    bars3 = ax.bar(x + width, reasoning_impr, width, label='Reasoning', color='#f39c12', alpha=0.8)

    ax.set_ylabel('Improvement (%)')
    ax.set_title('Vocabulary Usage Improvement (Intervention vs Baseline)', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.savefig('visualization/vocab_improvement_summary.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: vocab_improvement_summary.png")

def plot_word_counts_comparison(results):
    """Plot raw counts comparison"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    count_metrics = [
        ('total_causal_count', 'Causal Word Count'),
        ('total_logical_count', 'Logical Word Count'),
        ('total_reasoning_count', 'Total Reasoning Count')
    ]

    for ax, (metric, title) in zip(axes, count_metrics):
        labels = []
        baseline_vals = []
        intervention_vals = []

        for result in results:
            config = result['config']
            label = f"{config['steering_type'].split('_')[0][:6]}\n{config['dataset'][:8]}"
            labels.append(label)
            baseline_vals.append(result['baseline'][metric])
            intervention_vals.append(result['intervention'][metric])

        x = np.arange(len(labels))
        width = 0.35

        bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline', color='#3498db', alpha=0.8)
        bars2 = ax.bar(x + width/2, intervention_vals, width, label='Intervention', color='#e74c3c', alpha=0.8)

        ax.set_ylabel('Count')
        ax.set_title(title, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig('visualization/vocab_counts_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: vocab_counts_comparison.png")

def plot_top_words_comparison(results):
    """Plot top words comparison for each result"""
    n_results = len(results)
    fig, axes = plt.subplots(n_results, 2, figsize=(14, 5 * n_results))

    if n_results == 1:
        axes = axes.reshape(1, 2)

    for idx, result in enumerate(results):
        config = result['config']
        title_prefix = f"{config['steering_type'].replace('_alpha5.0_7b', '')} × {config['dataset']}"

        # Causal words comparison
        ax_causal = axes[idx, 0]
        baseline_causal = result['baseline']['top_causal_words']
        intervention_causal = result['intervention']['top_causal_words']

        # Merge all words
        all_causal_words = set(baseline_causal.keys()) | set(intervention_causal.keys())
        words = sorted(all_causal_words, key=lambda w: baseline_causal.get(w, 0) + intervention_causal.get(w, 0), reverse=True)[:8]

        baseline_counts = [baseline_causal.get(w, 0) for w in words]
        intervention_counts = [intervention_causal.get(w, 0) for w in words]

        x = np.arange(len(words))
        width = 0.35

        ax_causal.barh(x - width/2, baseline_counts, width, label='Baseline', color='#3498db', alpha=0.8)
        ax_causal.barh(x + width/2, intervention_counts, width, label='Intervention', color='#e74c3c', alpha=0.8)
        ax_causal.set_yticks(x)
        ax_causal.set_yticklabels(words)
        ax_causal.set_xlabel('Count')
        ax_causal.set_title(f'{title_prefix}\nTop Causal Words', fontweight='bold')
        ax_causal.legend(loc='lower right')
        ax_causal.grid(axis='x', alpha=0.3)

        # Logical words comparison
        ax_logical = axes[idx, 1]
        baseline_logical = result['baseline']['top_logical_words']
        intervention_logical = result['intervention']['top_logical_words']

        all_logical_words = set(baseline_logical.keys()) | set(intervention_logical.keys())
        words = sorted(all_logical_words, key=lambda w: baseline_logical.get(w, 0) + intervention_logical.get(w, 0), reverse=True)[:8]

        baseline_counts = [baseline_logical.get(w, 0) for w in words]
        intervention_counts = [intervention_logical.get(w, 0) for w in words]

        x = np.arange(len(words))

        ax_logical.barh(x - width/2, baseline_counts, width, label='Baseline', color='#3498db', alpha=0.8)
        ax_logical.barh(x + width/2, intervention_counts, width, label='Intervention', color='#e74c3c', alpha=0.8)
        ax_logical.set_yticks(x)
        ax_logical.set_yticklabels(words)
        ax_logical.set_xlabel('Count')
        ax_logical.set_title(f'{title_prefix}\nTop Logical Words', fontweight='bold')
        ax_logical.legend(loc='lower right')
        ax_logical.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('visualization/vocab_top_words_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: vocab_top_words_comparison.png")

def plot_combined_metrics(results):
    """Create a combined dashboard view"""
    fig = plt.figure(figsize=(10, 6))

    # Use gridspec for complex layout
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    for result in results:
        baseline = result['baseline']
        intervention = result['intervention']
        config = result['config']

        # Top left: Frequency per 1k words comparison
        ax1 = fig.add_subplot(gs[0, 0])
        metrics = ['Causal', 'Logical', 'Reasoning']
        baseline_freqs = [
            baseline['aggregate_causal_freq_per_1k'],
            baseline['aggregate_logical_freq_per_1k'],
            baseline['aggregate_reasoning_freq_per_1k']
        ]
        intervention_freqs = [
            intervention['aggregate_causal_freq_per_1k'],
            intervention['aggregate_logical_freq_per_1k'],
            intervention['aggregate_reasoning_freq_per_1k']
        ]

        x = np.arange(len(metrics))
        width = 0.35
        bars1 = ax1.bar(x - width/2, baseline_freqs, width, label='Baseline', color='#3498db', alpha=0.8)
        bars2 = ax1.bar(x + width/2, intervention_freqs, width, label='Intervention', color='#e74c3c', alpha=0.8)

        ax1.set_ylabel('Frequency per 1000 words', fontsize=10)
        ax1.set_title('Word Frequency Comparison', fontweight='bold', fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)

        # Top right: Raw counts comparison
        ax2 = fig.add_subplot(gs[0, 1])
        counts_baseline = [
            baseline['total_causal_count'],
            baseline['total_logical_count'],
            baseline['total_reasoning_count']
        ]
        counts_intervention = [
            intervention['total_causal_count'],
            intervention['total_logical_count'],
            intervention['total_reasoning_count']
        ]

        bars1 = ax2.bar(x - width/2, counts_baseline, width, label='Baseline', color='#3498db', alpha=0.8)
        bars2 = ax2.bar(x + width/2, counts_intervention, width, label='Intervention', color='#e74c3c', alpha=0.8)

        ax2.set_ylabel('Total Count', fontsize=10)
        ax2.set_title('Word Count Comparison', fontweight='bold', fontsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        for bar in bars1:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9)

        # Bottom left: Top causal words
        ax3 = fig.add_subplot(gs[1, 0])
        baseline_causal = baseline['top_causal_words']
        intervention_causal = intervention['top_causal_words']
        all_words = set(baseline_causal.keys()) | set(intervention_causal.keys())
        words = sorted(all_words, key=lambda w: baseline_causal.get(w, 0) + intervention_causal.get(w, 0), reverse=True)[:6]

        y = np.arange(len(words))
        height = 0.35

        ax3.barh(y - height/2, [baseline_causal.get(w, 0) for w in words], height,
                label='Baseline', color='#3498db', alpha=0.8)
        ax3.barh(y + height/2, [intervention_causal.get(w, 0) for w in words], height,
                label='Intervention', color='#e74c3c', alpha=0.8)
        ax3.set_yticks(y)
        ax3.set_yticklabels(words)
        ax3.set_xlabel('Count')
        ax3.set_title('Top Causal Words', fontweight='bold', fontsize=12)
        ax3.legend(loc='lower right')
        ax3.grid(axis='x', alpha=0.3)

        # Bottom right: Top logical words
        ax4 = fig.add_subplot(gs[1, 1])
        baseline_logical = baseline['top_logical_words']
        intervention_logical = intervention['top_logical_words']
        all_words = set(baseline_logical.keys()) | set(intervention_logical.keys())
        words = sorted(all_words, key=lambda w: baseline_logical.get(w, 0) + intervention_logical.get(w, 0), reverse=True)[:6]

        y = np.arange(len(words))

        ax4.barh(y - height/2, [baseline_logical.get(w, 0) for w in words], height,
                label='Baseline', color='#3498db', alpha=0.8)
        ax4.barh(y + height/2, [intervention_logical.get(w, 0) for w in words], height,
                label='Intervention', color='#e74c3c', alpha=0.8)
        ax4.set_yticks(y)
        ax4.set_yticklabels(words)
        ax4.set_xlabel('Count')
        ax4.set_title('Top Logical Words', fontweight='bold', fontsize=12)
        ax4.legend(loc='lower right')
        ax4.grid(axis='x', alpha=0.3)

        # Only use the first result for combined view
        break

    # Add overall title
    config = results[0]['config']
    improvements = results[0]['improvements']
    fig.suptitle(f"Vocabulary Analysis\n"
                f"Improvements - Causal: {improvements['causal_improvement_pct']:.1f}%, "
                f"Logical: {improvements['logical_improvement_pct']:.1f}%, "
                f"Reasoning: {improvements['reasoning_improvement_pct']:.1f}%",
                fontsize=18, fontweight='bold', y=1.02)

    plt.savefig('visualization/vocab_combined_dashboard.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: vocab_combined_dashboard.png")

def print_summary(results):
    """Print summary statistics"""
    print("\n" + "="*70)
    print("VOCABULARY ANALYSIS SUMMARY")
    print("="*70)

    for result in results:
        config = result['config']
        baseline = result['baseline']
        intervention = result['intervention']
        improvements = result['improvements']

        print(f"\n{config['steering_type']} × {config['dataset']}")
        print("-"*50)
        print(f"{'Metric':<25} {'Baseline':>12} {'Intervention':>12} {'Δ%':>10}")
        print("-"*50)

        print(f"{'Total Words':<25} {baseline['total_words']:>12} {intervention['total_words']:>12}")
        print(f"{'Causal Count':<25} {baseline['total_causal_count']:>12} {intervention['total_causal_count']:>12} {improvements['causal_improvement_pct']:>+9.1f}%")
        print(f"{'Logical Count':<25} {baseline['total_logical_count']:>12} {intervention['total_logical_count']:>12} {improvements['logical_improvement_pct']:>+9.1f}%")
        print(f"{'Reasoning Count':<25} {baseline['total_reasoning_count']:>12} {intervention['total_reasoning_count']:>12} {improvements['reasoning_improvement_pct']:>+9.1f}%")
        print("-"*50)
        print(f"{'Causal Freq/1k':<25} {baseline['aggregate_causal_freq_per_1k']:>12.2f} {intervention['aggregate_causal_freq_per_1k']:>12.2f}")
        print(f"{'Logical Freq/1k':<25} {baseline['aggregate_logical_freq_per_1k']:>12.2f} {intervention['aggregate_logical_freq_per_1k']:>12.2f}")
        print(f"{'Reasoning Freq/1k':<25} {baseline['aggregate_reasoning_freq_per_1k']:>12.2f} {intervention['aggregate_reasoning_freq_per_1k']:>12.2f}")

def main():
    print("="*70)
    print("VOCABULARY ANALYSIS VISUALIZATION")
    print("="*70)
    print(f"\nResults directory: {RESULTS_DIR}")

    if not RESULTS_DIR.exists():
        print(f"\n✗ Results directory not found: {RESULTS_DIR}")
        return

    # Create output directory
    Path("visualization").mkdir(exist_ok=True)

    # Load results
    print("\n" + "="*70)
    print("LOADING RESULTS")
    print("="*70)
    results = load_all_results(RESULTS_DIR)

    if not results:
        print("\n✗ No results found!")
        return

    print(f"\nLoaded {len(results)} result(s)")

    # Print summary
    print_summary(results)

    # Create visualizations
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)

    plot_combined_metrics(results)
    plot_frequency_comparison(results)
    plot_word_counts_comparison(results)
    plot_top_words_comparison(results)
    plot_improvement_summary(results)

    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE!")
    print("="*70)
    print("\nGenerated files in visualization/:")
    print("  - vocab_combined_dashboard.png")
    print("  - vocab_frequency_comparison.png")
    print("  - vocab_counts_comparison.png")
    print("  - vocab_top_words_comparison.png")
    print("  - vocab_improvement_summary.png")

if __name__ == "__main__":
    main()
