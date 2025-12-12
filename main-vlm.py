"""
Main script for VLM steering with entropy-based dynamic layer selection
Alpha is fixed at 5.0, layers are selected dynamically based on entropy
"""

import torch
import numpy as np
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from pathlib import Path
import json
from tqdm import tqdm
import argparse
from typing import List, Dict, Tuple, Optional
import random
from datetime import datetime

from dataset_loaders_vlm import (
    load_vlm_dataset,
    format_vlm_prompt,
    extract_answer_from_response,
    evaluate_vlm_answer
)

from optimize_entropy_layer import (
    EntropyLayerSelector,
    evaluate_with_entropy_steering
)


class VLMSteeringConfig:
    """Configuration for VLM steering experiments"""

    # Model configuration
    vlm_model = "Qwen/Qwen2.5-VL-7B-Instruct"

    # Steering vector source
    steering_output_dir = "./steering_outputs"

    # Target VLM dataset
    target_dataset = "scienceqa"  # Options: scienceqa, mathvista, geometry3k, okvqa
    dataset_split = "test"  # Will be overridden to testmini for mathvista
    num_validation_samples = 100  # For validation
    num_test_samples = 200  # For final testing

    # Steering configuration
    fixed_alpha = 5.0  # Fixed steering strength
    entropy_mode = 'per_prompt'  # Entropy: dynamic layer selection per prompt

    # Generation parameters
    max_new_tokens = 2048
    temperature = 0.1
    do_sample = False

    # Output
    output_dir = "./vlm_steering_results"
    random_seed = 42

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_steering_vectors(steering_path: Path) -> Tuple[List[torch.Tensor], Dict]:
    """Load steering vectors from pre-computed outputs"""
    print(f"\nLoading steering vectors from: {steering_path}")

    if not steering_path.exists():
        raise FileNotFoundError(f"Steering vectors not found at: {steering_path}")

    data = torch.load(steering_path, map_location='cpu')

    steering_vectors = data['steering_vectors']
    metadata = {
        'version': data.get('version', 'unknown'),
        'stats': data.get('steering_stats', []),
    }

    print(f"✓ Loaded {len(steering_vectors)} steering vectors")
    print(f"  Version: {metadata['version']}")

    return steering_vectors, metadata


def load_vlm_model(model_name: str, device: str):
    """Load VLM model and processor"""
    print(f"\nLoading VLM model: {model_name}...")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_name)

    model.eval()
    print(f"✓ Loaded VLM model")

    return model, processor


def run_vlm_steering_experiment(steering_type: str, config: VLMSteeringConfig):
    """
    Run VLM steering experiment with entropy-based dynamic layer selection

    Args:
        steering_type: Type of steering vectors to use
        config: Configuration object
    """
    print("\n" + "="*70)
    print("VLM STEERING EXPERIMENT")
    print("="*70)
    print(f"Steering Type: {steering_type}")
    print(f"Target Dataset: {config.target_dataset}")
    print(f"VLM Model: {config.vlm_model}")
    print(f"Fixed Alpha: {config.fixed_alpha}")

    print(f"  Entropy: {config.entropy_mode} (layers selected dynamically per sample)")
    print("="*70)

    # Set random seed
    set_seed(config.random_seed)

    # Load steering vectors
    steering_path = Path(config.steering_output_dir) / steering_type / "steering_vectors.pt"
    steering_vectors, metadata = load_steering_vectors(steering_path)

    # Load VLM model
    model, processor = load_vlm_model(config.vlm_model, config.device)

    # Load dataset
    print(f"\nLoading {config.target_dataset} dataset...")
    full_dataset = load_vlm_dataset(
        config.target_dataset,
        split=config.dataset_split,
        num_samples=config.num_validation_samples + config.num_test_samples,
        random_seed=config.random_seed
    )

    # Split dataset
    validation_data = full_dataset[:config.num_validation_samples]
    test_data = full_dataset[config.num_validation_samples:
                            config.num_validation_samples + config.num_test_samples]

    print(f"Dataset split: {len(validation_data)} val / {len(test_data)} test")

    # Create selectors for entropy-based steering
    print("\n" + "="*70)
    print("VALIDATION")
    print("="*70)

    # Baseline selector (alpha=0, no steering)
    baseline_selector = EntropyLayerSelector(
        model=model,
        steering_vectors=steering_vectors,
        alpha=0.0,
        mode=config.entropy_mode
    )

    # Steering selector (alpha=5.0, dynamic layers)
    steering_selector = EntropyLayerSelector(
        model=model,
        steering_vectors=steering_vectors,
        alpha=config.fixed_alpha,
        mode=config.entropy_mode
    )

    # Evaluate on validation set
    print("\nEvaluating baseline (no steering)...")
    baseline_val_results = evaluate_with_entropy_steering(
        model, processor, validation_data,
        selector=baseline_selector,
        extract_answer_fn=extract_answer_from_response,
        evaluate_answer_fn=evaluate_vlm_answer
    )
    print(f"✓ Baseline Validation Accuracy: {baseline_val_results['accuracy']:.4f}")

    print("\nEvaluating with entropy-based steering...")
    steered_val_results = evaluate_with_entropy_steering(
        model, processor, validation_data,
        selector=steering_selector,
        extract_answer_fn=extract_answer_from_response,
        evaluate_answer_fn=evaluate_vlm_answer
    )
    print(f"✓ Steered Validation Accuracy: {steered_val_results['accuracy']:.4f}")
    print(f"✓ Mean Selected Layer: {steered_val_results['mean_selected_layer']:.1f}")
    print(f"✓ Most Common Layer: {steered_val_results['selector_statistics']['most_common_layer']}")
    print(f"✓ Improvement: {steered_val_results['accuracy'] - baseline_val_results['accuracy']:+.4f}")

    # Final test
    print("\n" + "="*70)
    print("FINAL TEST")
    print("="*70)

    # Reset selectors for test
    baseline_test_selector = EntropyLayerSelector(
        model=model,
        steering_vectors=steering_vectors,
        alpha=0.0,
        mode=config.entropy_mode
    )

    steering_test_selector = EntropyLayerSelector(
        model=model,
        steering_vectors=steering_vectors,
        alpha=config.fixed_alpha,
        mode=config.entropy_mode
    )

    print("\nEvaluating baseline on test set...")
    baseline_test_results = evaluate_with_entropy_steering(
        model, processor, test_data,
        selector=baseline_test_selector,
        extract_answer_fn=extract_answer_from_response,
        evaluate_answer_fn=evaluate_vlm_answer
    )
    print(f"✓ Baseline Test Accuracy: {baseline_test_results['accuracy']:.4f}")

    print("\nEvaluating with steering on test set...")
    steered_test_results = evaluate_with_entropy_steering(
        model, processor, test_data,
        selector=steering_test_selector,
        extract_answer_fn=extract_answer_from_response,
        evaluate_answer_fn=evaluate_vlm_answer
    )
    print(f"✓ Steered Test Accuracy: {steered_test_results['accuracy']:.4f}")
    print(f"✓ Mean Selected Layer: {steered_test_results['mean_selected_layer']:.1f}")
    print(f"✓ Most Common Layer: {steered_test_results['selector_statistics']['most_common_layer']}")
    print(f"✓ Improvement: {steered_test_results['accuracy'] - baseline_test_results['accuracy']:+.4f}")

    # Print layer distribution analysis
    print("\n" + "="*70)
    print("LAYER SELECTION ANALYSIS")
    print("="*70)
    print(f"Layer distribution (test set):")
    layer_dist = steered_test_results['layer_distribution']
    for layer_idx, count in enumerate(layer_dist):
        if count > 0:
            percentage = count / steered_test_results['total'] * 100
            bar = "█" * int(percentage / 2)
            print(f"  Layer {layer_idx:2d}: {count:3d} ({percentage:5.1f}%) {bar}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    method_name = 'entropy'
    output_dir = Path(config.output_dir) / f"{steering_type}_{config.target_dataset}_{method_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'method': 'entropy_dynamic_layers',
        'steering_type': steering_type,
        'target_dataset': config.target_dataset,
        'vlm_model': config.vlm_model,
        'alpha': config.fixed_alpha,
        'entropy_mode': config.entropy_mode,
        'validation': {
            'baseline': {
                'accuracy': baseline_val_results['accuracy'],
                'correct': baseline_val_results['correct'],
                'total': baseline_val_results['total']
            },
            'steered': {
                'accuracy': steered_val_results['accuracy'],
                'correct': steered_val_results['correct'],
                'total': steered_val_results['total'],
                'mean_selected_layer': steered_val_results['mean_selected_layer'],
                'layer_distribution': steered_val_results['layer_distribution'],
                'selector_statistics': steered_val_results['selector_statistics']
            }
        },
        'test': {
            'baseline': {
                'accuracy': baseline_test_results['accuracy'],
                'correct': baseline_test_results['correct'],
                'total': baseline_test_results['total']
            },
            'steered': {
                'accuracy': steered_test_results['accuracy'],
                'correct': steered_test_results['correct'],
                'total': steered_test_results['total'],
                'mean_selected_layer': steered_test_results['mean_selected_layer'],
                'layer_distribution': steered_test_results['layer_distribution'],
                'selector_statistics': steered_test_results['selector_statistics']
            }
        },
        'timestamp': timestamp
    }

    # Save results JSON
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n✓ Results saved to: {output_dir}")

    # Print final summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print(f"Method: Entropy-based dynamic layer selection")
    print(f"Alpha: {config.fixed_alpha} (fixed)")
    print(f"Layers: Dynamic (mean={steered_test_results['mean_selected_layer']:.1f}, "
            f"most common={steered_test_results['selector_statistics']['most_common_layer']})")

    print(f"\nValidation: {baseline_val_results['accuracy']:.4f} → {steered_val_results['accuracy']:.4f} "
          f"({steered_val_results['accuracy'] - baseline_val_results['accuracy']:+.4f})")
    print(f"Test:       {baseline_test_results['accuracy']:.4f} → {steered_test_results['accuracy']:.4f} "
          f"({steered_test_results['accuracy'] - baseline_test_results['accuracy']:+.4f})")
    print("="*70)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="VLM Steering with Entropy-based Dynamic Layer Selection"
    )
    parser.add_argument("--steering_type", type=str, required=True,
                       help="Type of steering vectors (e.g., mathematical_reasoning_gsm8k_alpha5.0_7b)")
    parser.add_argument("--target_dataset", type=str, default="scienceqa",
                       choices=["scienceqa", "mathvista", "geometry3k", "okvqa", "chartqa", "gqa", "vqav2"],
                       help="Target VLM dataset (scienceqa/mathvista/geometry3k/okvqa/chartqa/gqa/vqav2)")
    parser.add_argument("--alpha", type=float, default=5.0,
                       help="Fixed steering strength (default: 5.0)")
    parser.add_argument("--num_val_samples", type=int, default=100,
                       help="Number of samples for validation")
    parser.add_argument("--num_test_samples", type=int, default=200,
                       help="Number of samples for testing")
    parser.add_argument("--output_dir", type=str, default="./vlm_steering_results",
                       help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    args = parser.parse_args()

    # Update config
    config = VLMSteeringConfig()
    config.target_dataset = args.target_dataset
    # MathVista test split has no labels, use testmini instead
    config.dataset_split = "testmini" if args.target_dataset == "mathvista" else "test"
    config.fixed_alpha = args.alpha
    config.num_validation_samples = args.num_val_samples
    config.num_test_samples = args.num_test_samples
    config.output_dir = args.output_dir
    config.random_seed = args.seed

    # Run experiment
    run_vlm_steering_experiment(args.steering_type, config)


if __name__ == "__main__":
    main()
