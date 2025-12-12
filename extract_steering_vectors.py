#!/usr/bin/env python3
"""
Standalone script to extract steering vectors from model activation differences

This script:
1. Loads configuration from config.py
2. Loads base and reasoning models
3. Loads training data from specified dataset
4. Extracts steering vectors from activation differences
5. Saves to steering_outputs/{dataset_name}_{num_samples}_{model_name}/

Usage:
    python extract_steering_vectors.py [--ability ABILITY] [--num_samples NUM]

Examples:
    # Extract steering vectors for mathematical reasoning (default)
    python extract_steering_vectors.py

    # Extract for science reasoning with 100 samples
    python extract_steering_vectors.py --ability science_reasoning --num_samples 100

    # Extract for logical reasoning
    python extract_steering_vectors.py --ability logical_reasoning
"""

import os
import json
import argparse
from datetime import datetime
import torch
import numpy as np
from tqdm import tqdm

from config import Config
from model_utils import load_model, get_activations, generate_response, get_model_info
from dataset_loaders import load_dataset_by_name, extract_answer, evaluate_answer
from steering_vectors import (
    extract_steering_vectors,
    analyze_layer_groups,
    save_steering_vectors
)


def filter_correct_samples(base_model, base_tokenizer, reasoning_model, reasoning_tokenizer,
                           dataset, device, max_new_tokens=512):
    """
    Filter samples where BOTH models answer correctly

    Args:
        base_model: Base model
        base_tokenizer: Base tokenizer
        reasoning_model: Reasoning model
        reasoning_tokenizer: Reasoning tokenizer
        dataset: List of samples with 'question' and 'answer' keys
        device: Device to use
        max_new_tokens: Max tokens to generate

    Returns:
        List of filtered samples where both models are correct
    """
    filtered_samples = []

    print(f"\nFiltering samples where both models are correct...")
    print(f"Total samples to check: {len(dataset)}")

    base_correct = 0
    reasoning_correct = 0
    both_correct = 0

    for i, sample in enumerate(tqdm(dataset, desc="Filtering samples")):
        question = sample['question']
        expected_answer = sample['answer']

        # Generate response from base model
        base_response = generate_response(
            base_model, base_tokenizer, question,
            max_new_tokens=max_new_tokens, device=device
        )
        base_extracted = extract_answer(base_response)
        base_is_correct = evaluate_answer(base_extracted, expected_answer)

        # Generate response from reasoning model
        reasoning_response = generate_response(
            reasoning_model, reasoning_tokenizer, question,
            max_new_tokens=max_new_tokens, device=device
        )
        reasoning_extracted = extract_answer(reasoning_response)
        reasoning_is_correct = evaluate_answer(reasoning_extracted, expected_answer)

        if base_is_correct:
            base_correct += 1
        if reasoning_is_correct:
            reasoning_correct += 1

        # Only keep if BOTH models are correct
        if base_is_correct and reasoning_is_correct:
            filtered_samples.append(sample)
            both_correct += 1

    print(f"\nFiltering Results:")
    print(f"  Base model correct: {base_correct}/{len(dataset)} ({100*base_correct/len(dataset):.1f}%)")
    print(f"  Reasoning model correct: {reasoning_correct}/{len(dataset)} ({100*reasoning_correct/len(dataset):.1f}%)")
    print(f"  Both correct: {both_correct}/{len(dataset)} ({100*both_correct/len(dataset):.1f}%)")
    print(f"  Filtered samples: {len(filtered_samples)}")

    return filtered_samples


def get_model_short_name(model_name):
    """
    Extract short model name for output directory

    Examples:
        "Qwen/Qwen2.5-7B-Instruct" -> "qwen2_5_7b"
        "Qwen/Qwen2.5-Math-7B-Instruct" -> "qwen2_5_math_7b"
        "Qwen/Qwen2.5-3B-Instruct" -> "qwen2_5_3b"
    """
    # Extract the last part after '/'
    name = model_name.split('/')[-1]

    # Remove common suffixes
    name = name.replace('-Instruct', '').replace('-instruct', '')

    # Convert to lowercase and replace dots/dashes with underscores
    name = name.lower().replace('.', '_').replace('-', '_')

    return name


def get_output_dir_name(dataset_name, num_samples, model_name):
    """
    Generate output directory name

    Format: {dataset_name}_{num_samples}_{model_short_name}

    Examples:
        gsm8k_50_qwen2_5_7b
        arc_challenge_100_qwen2_5_7b
        winogrande_50_qwen2_5_3b
    """
    model_short = get_model_short_name(model_name)
    return f"{dataset_name}_{num_samples}_{model_short}"


def main():
    parser = argparse.ArgumentParser(
        description="Extract steering vectors from model activation differences"
    )
    parser.add_argument(
        "--ability",
        type=str,
        default=None,
        choices=[
            "mathematical_reasoning",
            "advanced_mathematical_reasoning",
            "logical_reasoning",
            "commonsense_reasoning",
            "science_reasoning",
            "reading_comprehension",
            "multihop_reasoning"
        ],
        help="Reasoning ability to target (overrides config.py)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of extraction samples (overrides config.py)"
    )
    parser.add_argument(
        "--num_candidates",
        type=int,
        default=None,
        help="Number of candidate samples to filter (default: 3x num_samples)"
    )
    parser.add_argument(
        "--skip_filter",
        action="store_true",
        help="Skip filtering and use all samples (not recommended)"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Base model name (overrides config.py)"
    )
    parser.add_argument(
        "--reasoning_model",
        type=str,
        default=None,
        help="Reasoning model name (overrides config.py)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (overrides config.py)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, overrides config.py)"
    )

    args = parser.parse_args()

    # Override config with command-line arguments if provided
    if args.ability is not None:
        Config.ability = args.ability
    if args.num_samples is not None:
        Config.num_extraction_samples = args.num_samples
    if args.base_model is not None:
        Config.base_model = args.base_model
    if args.reasoning_model is not None:
        Config.reasoning_model = args.reasoning_model
    if args.output_dir is not None:
        Config.output_dir = args.output_dir
    if args.device is not None:
        Config.device = args.device

    # Print configuration
    print("\n" + "="*80)
    print("STEERING VECTOR EXTRACTION")
    print("="*80)
    print(f"Ability: {Config.ability}")
    print(f"Dataset: {Config.get_dataset()}")
    print(f"Extraction Samples: {Config.num_extraction_samples}")
    print(f"Base Model: {Config.base_model}")
    print(f"Reasoning Model: {Config.reasoning_model}")
    print(f"Device: {Config.device}")
    print(f"Torch Dtype: {Config.torch_dtype}")
    print("="*80 + "\n")

    # Get dataset name
    dataset_name = Config.get_dataset()

    # Generate output directory name
    output_dir_name = get_output_dir_name(
        dataset_name,
        Config.num_extraction_samples,
        Config.base_model
    )
    output_dir = os.path.join(Config.output_dir, output_dir_name)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # ============================================
    # 1. Load Dataset
    # ============================================
    print("="*80)
    print("LOADING DATASET")
    print("="*80)

    # Load more samples for filtering (default: 3x target)
    num_candidates = args.num_candidates if args.num_candidates else Config.num_extraction_samples * 3

    extraction_data = load_dataset_by_name(
        dataset_name,
        split='train',
        num_samples=num_candidates,
        random_seed=Config.random_seed
    )

    print(f"✓ Loaded {len(extraction_data)} candidate samples for filtering")
    print(f"Example prompt: {extraction_data[0]['question'][:100]}...\n")

    # ============================================
    # 2. Load Models
    # ============================================
    print("="*80)
    print("LOADING MODELS")
    print("="*80)

    # Load base model
    base_model, base_tokenizer = load_model(
        Config.base_model,
        device=Config.device,
        torch_dtype=Config.torch_dtype
    )

    # Load reasoning model
    reasoning_model, reasoning_tokenizer = load_model(
        Config.reasoning_model,
        device=Config.device,
        torch_dtype=Config.torch_dtype
    )

    # Get model info
    model_info = get_model_info(base_model)
    num_layers = model_info['num_layers']
    hidden_size = model_info['hidden_size']

    print(f"\nModel Info:")
    print(f"  Layers: {num_layers}")
    print(f"  Hidden Size: {hidden_size}")
    print()

    # ============================================
    # 2.5. Filter Samples (Both Models Correct)
    # ============================================
    if not args.skip_filter:
        print("="*80)
        print("FILTERING SAMPLES (BOTH MODELS CORRECT)")
        print("="*80)

        filtered_data = filter_correct_samples(
            base_model=base_model,
            base_tokenizer=base_tokenizer,
            reasoning_model=reasoning_model,
            reasoning_tokenizer=reasoning_tokenizer,
            dataset=extraction_data,
            device=Config.device,
            max_new_tokens=Config.max_new_tokens
        )

        if len(filtered_data) < Config.num_extraction_samples:
            print(f"\n⚠ Warning: Only {len(filtered_data)} samples passed filtering, "
                  f"but {Config.num_extraction_samples} were requested.")
            print(f"  Using all {len(filtered_data)} filtered samples.")
            extraction_prompts = [item['question'] for item in filtered_data]
        else:
            # Use only the requested number of samples
            filtered_data = filtered_data[:Config.num_extraction_samples]
            extraction_prompts = [item['question'] for item in filtered_data]
            print(f"\n✓ Using {len(extraction_prompts)} filtered samples for extraction")
    else:
        print("\n⚠ Skipping filter (using all samples)")
        extraction_prompts = [item['question'] for item in extraction_data[:Config.num_extraction_samples]]

    print()

    # ============================================
    # 3. Extract Steering Vectors
    # ============================================
    print("="*80)
    print("EXTRACTING STEERING VECTORS")
    print("="*80)

    steering_vectors, steering_stats = extract_steering_vectors(
        base_model=base_model,
        base_tokenizer=base_tokenizer,
        reasoning_model=reasoning_model,
        reasoning_tokenizer=reasoning_tokenizer,
        training_prompts=extraction_prompts,
        num_layers=num_layers,
        hidden_size=hidden_size,
        device=Config.device,
        max_length=Config.max_length
    )

    # ============================================
    # 4. Analyze Layer Groups
    # ============================================
    # Define layer groups (early, middle, late)
    layer_groups = {
        'early': list(range(0, num_layers // 3)),
        'middle': list(range(num_layers // 3, 2 * num_layers // 3)),
        'late': list(range(2 * num_layers // 3, num_layers))
    }

    best_layers_per_group = analyze_layer_groups(steering_stats, layer_groups)

    # ============================================
    # 5. Save Results
    # ============================================
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    # Save steering vectors
    sv_path = os.path.join(output_dir, "steering_vectors.pt")
    save_steering_vectors(
        steering_vectors=steering_vectors,
        steering_stats=steering_stats,
        save_path=sv_path,
        layer_groups=layer_groups,
        best_layers_per_group=best_layers_per_group
    )

    # Save metadata
    metadata = {
        'ability': Config.ability,
        'dataset': dataset_name,
        'num_extraction_samples': len(extraction_prompts),
        'num_candidate_samples': len(extraction_data),
        'filtered': not args.skip_filter,
        'base_model': Config.base_model,
        'reasoning_model': Config.reasoning_model,
        'num_layers': num_layers,
        'hidden_size': hidden_size,
        'layer_groups': layer_groups,
        'best_layers_per_group': best_layers_per_group,
        'extraction_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'config': {
            'max_length': Config.max_length,
            'max_new_tokens': Config.max_new_tokens,
            'device': Config.device,
            'torch_dtype': Config.torch_dtype,
            'random_seed': Config.random_seed
        }
    }

    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata to: {metadata_path}")

    # Save steering statistics as JSON for easy viewing
    stats_path = os.path.join(output_dir, "layer_statistics.json")
    with open(stats_path, 'w') as f:
        json.dump(steering_stats, f, indent=2)
    print(f"✓ Saved layer statistics to: {stats_path}")

    # Save summary statistics
    summary = {
        'output_directory': output_dir,
        'dataset': dataset_name,
        'num_extraction_samples': len(extraction_prompts),
        'num_candidate_samples': len(extraction_data),
        'filtered': not args.skip_filter,
        'model': get_model_short_name(Config.base_model),
        'num_layers': num_layers,
        'hidden_size': hidden_size,
        'best_layers': {
            group: layers
            for group, layers in best_layers_per_group.items()
        },
        'layer_stats_summary': {
            'avg_consistency': np.mean([s['consistency'] for s in steering_stats]),
            'avg_original_norm': np.mean([s['original_norm'] for s in steering_stats]),
            'max_consistency': max([s['consistency'] for s in steering_stats]),
            'max_original_norm': max([s['original_norm'] for s in steering_stats])
        }
    }

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary to: {summary_path}")

    # ============================================
    # 6. Print Summary
    # ============================================
    print("\n" + "="*80)
    print("EXTRACTION COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print(f"\nFiles created:")
    print(f"  1. steering_vectors.pt    - Steering vectors for all layers")
    print(f"  2. metadata.json          - Extraction metadata")
    print(f"  3. layer_statistics.json  - Detailed statistics for each layer")
    print(f"  4. summary.json           - Summary statistics")

    print(f"\nBest layers (middle group):")
    for layer in best_layers_per_group.get('middle', []):
        stat = steering_stats[layer]
        print(f"  Layer {layer}: consistency={stat['consistency']:.4f}, "
              f"norm={stat['original_norm']:.2f}")

    print("\n" + "="*80)
    print(f"To use these steering vectors in experiments:")
    print(f"  --steering_vectors {output_dir}/steering_vectors.pt")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
