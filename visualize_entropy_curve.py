#!/usr/bin/env python3
"""
Visualize entropy curve across VLM layers

This script visualizes how entropy changes across different layers of a VLM
when processing a multimodal input (image + text prompt).

Usage:
    # Visualize entropy for a single sample
    python visualize_entropy_curve.py --dataset scienceqa --sample_idx 0

    # Visualize average entropy across multiple samples
    python visualize_entropy_curve.py --dataset geometry3k --num_samples 10

    # Save to specific directory
    python visualize_entropy_curve.py --dataset mathvista --output_dir ./entropy_analysis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from dataset_loaders_vlm import load_vlm_dataset, format_vlm_prompt

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11


def compute_layer_entropies(model, processor, image, prompt: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute entropy for each layer's prediction distribution

    Args:
        model: VLM model
        processor: VLM processor
        image: Input image
        prompt: Text prompt

    Returns:
        entropies: Array of shape [num_layers] with entropy values
        entropy_deltas: Array of shape [num_layers-1] with entropy changes
    """
    # Prepare input
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ]
    }]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    # Get output embedding matrix for projecting hidden states to vocabulary
    W_unembed = model.get_output_embeddings().weight  # [vocab_size, hidden_size]

    # Forward pass with hidden states
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )
        hidden_states = outputs.hidden_states  # Tuple of [batch_size, seq_len, hidden_size]

    all_layer_entropies = []

    # Skip embedding layer (index 0), process decoder layers (index 1 onwards)
    for layer_idx, h_i in enumerate(hidden_states[1:]):
        # Get last token's hidden state
        last_token_hidden = h_i[:, -1, :]  # [batch_size, hidden_size]

        # Project to vocabulary space
        W_unembed_device = W_unembed.to(last_token_hidden.device)
        logits_i = last_token_hidden @ W_unembed_device.T  # [batch_size, vocab_size]

        # Compute probability distribution
        log_probs_i = torch.log_softmax(logits_i, dim=-1)
        probs_i = torch.softmax(logits_i, dim=-1)

        # Compute entropy: H(X) = -sum(P(x) * log(P(x)))
        entropy_i = -(probs_i * log_probs_i).sum(dim=-1)  # [batch_size]

        all_layer_entropies.append(entropy_i.cpu().item())

    entropies = np.array(all_layer_entropies)

    # Compute entropy deltas (entropy change between consecutive layers)
    entropy_deltas = entropies[:-1] - entropies[1:]  # Positive means entropy is dropping

    return entropies, entropy_deltas


def visualize_single_sample(entropies: np.ndarray, entropy_deltas: np.ndarray,
                           sample_info: Dict, output_path: Path):
    """
    Visualize entropy curve for a single sample

    Args:
        entropies: Array of entropy values
        entropy_deltas: Array of entropy changes
        sample_info: Dictionary with sample metadata
        output_path: Path to save figure
    """
    num_layers = len(entropies)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Entropy across layers
    ax1.plot(range(num_layers), entropies, marker='o', linewidth=2,
             markersize=6, color='#2E86AB', label='Entropy')
    ax1.axhline(y=entropies.mean(), color='gray', linestyle='--',
                alpha=0.5, label=f'Mean: {entropies.mean():.2f}')
    ax1.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Entropy', fontsize=12, fontweight='bold')
    ax1.set_title('Entropy Across VLM Layers', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # Highlight regions
    early_region = num_layers // 3
    late_region = 2 * num_layers // 3
    ax1.axvspan(0, early_region, alpha=0.1, color='green', label='Early Layers')
    ax1.axvspan(early_region, late_region, alpha=0.1, color='blue', label='Middle Layers')
    ax1.axvspan(late_region, num_layers, alpha=0.1, color='red', label='Late Layers')

    # Plot 2: Entropy change (delta) between consecutive layers
    ax2.bar(range(len(entropy_deltas)), entropy_deltas, color='#A23B72', alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Entropy Change (ΔH)', fontsize=12, fontweight='bold')
    ax2.set_title('Entropy Change Between Consecutive Layers (Positive = Entropy Decreasing)',
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Highlight top-2 entropy drops (turning points)
    top_k = min(2, len(entropy_deltas))
    top_k_indices = np.argsort(entropy_deltas)[-top_k:][::-1]
    for idx in top_k_indices:
        ax2.bar(idx, entropy_deltas[idx], color='#F18F01', alpha=0.9)
        ax2.text(idx, entropy_deltas[idx], f'Layer {idx}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)

    # Add sample info
    dataset_name = sample_info.get('dataset', 'Unknown')
    question_preview = sample_info.get('question', '')[:60] + '...'
    fig.suptitle(f'Dataset: {dataset_name} | Question: {question_preview}',
                 fontsize=13, fontweight='bold', y=0.995)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved single sample visualization to: {output_path}")
    plt.close()


def visualize_multiple_samples(all_entropies: List[np.ndarray],
                               all_entropy_deltas: List[np.ndarray],
                               dataset_name: str, output_path: Path):
    """
    Visualize average entropy curve across multiple samples

    Args:
        all_entropies: List of entropy arrays
        all_entropy_deltas: List of entropy delta arrays
        dataset_name: Name of the dataset
        output_path: Path to save figure
    """
    # Convert to arrays and compute statistics
    entropies_array = np.array(all_entropies)  # [num_samples, num_layers]
    deltas_array = np.array(all_entropy_deltas)  # [num_samples, num_layers-1]

    mean_entropies = entropies_array.mean(axis=0)
    std_entropies = entropies_array.std(axis=0)

    mean_deltas = deltas_array.mean(axis=0)
    std_deltas = deltas_array.std(axis=0)

    num_layers = len(mean_entropies)
    num_samples = len(all_entropies)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 14))

    # Plot 1: Average entropy with confidence band
    ax1.plot(range(num_layers), mean_entropies, marker='o', linewidth=2.5,
             markersize=7, color='#2E86AB', label='Mean Entropy')
    ax1.fill_between(range(num_layers),
                     mean_entropies - std_entropies,
                     mean_entropies + std_entropies,
                     alpha=0.3, color='#2E86AB', label='±1 Std Dev')
    ax1.axhline(y=mean_entropies.mean(), color='gray', linestyle='--',
                alpha=0.5, label=f'Overall Mean: {mean_entropies.mean():.2f}')
    ax1.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Entropy', fontsize=12, fontweight='bold')
    ax1.set_title(f'Average Entropy Across VLM Layers ({num_samples} samples)',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # Highlight regions
    early_region = num_layers // 3
    late_region = 2 * num_layers // 3
    ax1.axvspan(0, early_region, alpha=0.1, color='green')
    ax1.axvspan(early_region, late_region, alpha=0.1, color='blue')
    ax1.axvspan(late_region, num_layers, alpha=0.1, color='red')

    # Plot 2: Average entropy change
    ax2.bar(range(len(mean_deltas)), mean_deltas, color='#A23B72', alpha=0.7,
            yerr=std_deltas, capsize=3, label='Mean ΔH ± Std')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Entropy Change (ΔH)', fontsize=12, fontweight='bold')
    ax2.set_title('Average Entropy Change Between Consecutive Layers',
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(fontsize=10)

    # Highlight top-2 entropy drops
    top_k = min(2, len(mean_deltas))
    top_k_indices = np.argsort(mean_deltas)[-top_k:][::-1]
    for idx in top_k_indices:
        ax2.bar(idx, mean_deltas[idx], color='#F18F01', alpha=0.9)
        ax2.text(idx, mean_deltas[idx], f'Layer {idx}\n({mean_deltas[idx]:.3f})',
                ha='center', va='bottom', fontweight='bold', fontsize=9)

    # Plot 3: Heatmap of all samples
    im = ax3.imshow(entropies_array, aspect='auto', cmap='viridis',
                    interpolation='nearest')
    ax3.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Sample Index', fontsize=12, fontweight='bold')
    ax3.set_title('Entropy Heatmap Across All Samples', fontsize=14, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('Entropy', fontsize=11, fontweight='bold')

    fig.suptitle(f'Dataset: {dataset_name} | Samples: {num_samples}',
                 fontsize=15, fontweight='bold', y=0.997)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved multi-sample visualization to: {output_path}")
    plt.close()


def visualize_layer_selection_distribution(all_selected_layers: List[int],
                                           num_layers: int,
                                           dataset_name: str,
                                           output_path: Path):
    """
    Visualize distribution of selected layers (based on entropy drops)

    Args:
        all_selected_layers: List of selected layer indices
        num_layers: Total number of layers
        dataset_name: Name of dataset
        output_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

    # Plot 1: Histogram
    counts = np.bincount(all_selected_layers, minlength=num_layers)
    ax1.bar(range(num_layers), counts, color='#2E86AB', alpha=0.7)
    ax1.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Selection Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution of Dynamically Selected Layers',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Highlight most common layer
    most_common_layer = np.argmax(counts)
    ax1.bar(most_common_layer, counts[most_common_layer], color='#F18F01', alpha=0.9)
    ax1.text(most_common_layer, counts[most_common_layer],
             f'Most Common\nLayer {most_common_layer}',
             ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Plot 2: Cumulative distribution
    percentages = counts / len(all_selected_layers) * 100
    ax2.plot(range(num_layers), np.cumsum(percentages),
             marker='o', linewidth=2.5, markersize=7, color='#A23B72')
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50%')
    ax2.axhline(y=90, color='gray', linestyle='--', alpha=0.5, label='90%')
    ax2.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Percentage (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Cumulative Selection Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_ylim([0, 105])

    fig.suptitle(f'Layer Selection Analysis | Dataset: {dataset_name} | Samples: {len(all_selected_layers)}',
                 fontsize=15, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved layer selection distribution to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize entropy curves across VLM layers"
    )
    parser.add_argument("--dataset", type=str, default="scienceqa",
                       choices=["scienceqa", "mathvista", "geometry3k", "okvqa",
                               "chartqa", "gqa", "vqav2"],
                       help="Dataset to use")
    parser.add_argument("--sample_idx", type=int, default=None,
                       help="Specific sample index to visualize (default: visualize multiple)")
    parser.add_argument("--num_samples", type=int, default=20,
                       help="Number of samples to average over (default: 20)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                       help="VLM model to use")
    parser.add_argument("--output_dir", type=str, default="./entropy_analysis",
                       help="Output directory for visualizations")
    parser.add_argument("--split", type=str, default="test",
                       help="Dataset split (test/testmini)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("ENTROPY CURVE VISUALIZATION")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Split: {args.split}")

    # Load VLM model
    print(f"\nLoading VLM model: {args.model}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model)
    model.eval()
    print(f"✓ Loaded VLM model")

    # Get model info
    num_layers = len(model.model.language_model.layers)
    print(f"Number of layers: {num_layers}")

    # Load dataset
    print(f"\nLoading {args.dataset} dataset...")
    # Override split for mathvista
    split = "testmini" if args.dataset == "mathvista" else args.split

    if args.sample_idx is not None:
        # Load single sample
        dataset = load_vlm_dataset(args.dataset, split=split, num_samples=args.sample_idx + 1)
        sample = dataset[args.sample_idx]
        dataset = [sample]
        num_samples = 1
    else:
        # Load multiple samples
        dataset = load_vlm_dataset(args.dataset, split=split,
                                   num_samples=args.num_samples, random_seed=args.seed)
        num_samples = len(dataset)

    print(f"✓ Loaded {num_samples} sample(s)")

    # Compute entropies
    print("\nComputing layer entropies...")
    all_entropies = []
    all_entropy_deltas = []
    all_selected_layers = []

    for i, sample in enumerate(tqdm(dataset, desc="Processing samples")):
        try:
            # Format prompt
            prompt = format_vlm_prompt(sample, include_cot=True)

            # Compute entropies
            entropies, entropy_deltas = compute_layer_entropies(
                model, processor, sample['image'], prompt
            )

            all_entropies.append(entropies)
            all_entropy_deltas.append(entropy_deltas)

            # Determine selected layer (top-1 for visualization)
            selected_layer = np.argmax(entropy_deltas)
            all_selected_layers.append(selected_layer)

            # For single sample, also create individual visualization
            if num_samples == 1:
                sample_info = {
                    'dataset': args.dataset,
                    'question': sample['metadata'].get('question', prompt[:100])
                }
                output_path = output_dir / f"{args.dataset}_sample_{args.sample_idx}_entropy_curve.png"
                visualize_single_sample(entropies, entropy_deltas, sample_info, output_path)

                # Print statistics
                print(f"\n{'='*70}")
                print("ENTROPY STATISTICS")
                print(f"{'='*70}")
                print(f"Min entropy: {entropies.min():.4f} (Layer {entropies.argmin()})")
                print(f"Max entropy: {entropies.max():.4f} (Layer {entropies.argmax()})")
                print(f"Mean entropy: {entropies.mean():.4f}")
                print(f"Std entropy: {entropies.std():.4f}")
                print(f"\nMax entropy drop: {entropy_deltas.max():.4f} (Layer {entropy_deltas.argmax()})")
                print(f"Selected layer (turning point): {selected_layer}")
                print(f"{'='*70}")

        except Exception as e:
            print(f"\n⚠ Error processing sample {i}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # For multiple samples, create aggregate visualizations
    if num_samples > 1:
        print("\nCreating aggregate visualizations...")

        # Average entropy curve
        output_path = output_dir / f"{args.dataset}_average_entropy_curve.png"
        visualize_multiple_samples(all_entropies, all_entropy_deltas,
                                   args.dataset, output_path)

        # Layer selection distribution
        output_path = output_dir / f"{args.dataset}_layer_selection_distribution.png"
        visualize_layer_selection_distribution(all_selected_layers, num_layers,
                                               args.dataset, output_path)

        # Print statistics
        mean_entropies = np.array(all_entropies).mean(axis=0)
        mean_deltas = np.array(all_entropy_deltas).mean(axis=0)

        print(f"\n{'='*70}")
        print("AGGREGATE STATISTICS")
        print(f"{'='*70}")
        print(f"Samples processed: {len(all_entropies)}")
        print(f"\nAverage min entropy: {mean_entropies.min():.4f} (Layer {mean_entropies.argmin()})")
        print(f"Average max entropy: {mean_entropies.max():.4f} (Layer {mean_entropies.argmax()})")
        print(f"Average mean entropy: {mean_entropies.mean():.4f}")
        print(f"\nAverage max entropy drop: {mean_deltas.max():.4f} (Layer {mean_deltas.argmax()})")
        print(f"Most frequently selected layer: {np.bincount(all_selected_layers).argmax()}")
        print(f"Layer selection distribution:")
        layer_counts = np.bincount(all_selected_layers, minlength=num_layers)
        for layer_idx, count in enumerate(layer_counts):
            if count > 0:
                percentage = count / len(all_selected_layers) * 100
                print(f"  Layer {layer_idx:2d}: {count:3d} samples ({percentage:5.1f}%)")
        print(f"{'='*70}")

    print(f"\n✓ All visualizations saved to: {output_dir}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
