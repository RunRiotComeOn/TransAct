"""
Steering vector extraction and analysis
"""

import torch
import numpy as np
from tqdm import tqdm
from model_utils import get_activations, generate_response


def extract_steering_vectors(base_model, base_tokenizer, reasoning_model, reasoning_tokenizer,
                            training_prompts, num_layers, hidden_size, device, max_length=2048):
    """
    Extract steering vectors from model activation differences

    Args:
        base_model: Base model
        base_tokenizer: Base tokenizer
        reasoning_model: Reasoning model
        reasoning_tokenizer: Reasoning tokenizer
        training_prompts: List of prompts for extraction
        num_layers: Number of layers in the model
        hidden_size: Hidden size of the model
        device: Device to use
        max_length: Maximum sequence length

    Returns:
        steering_vectors: List of normalized steering vectors (one per layer)
        steering_stats: Statistics for each layer
    """
    print("\n" + "="*60)
    print("Extracting Steering Vectors")
    print("="*60)

    layer_diffs = [[] for _ in range(num_layers)]

    for prompt in tqdm(training_prompts, desc="Processing extraction samples"):
        # Generate responses from both models
        base_response = generate_response(base_model, base_tokenizer, prompt, device)
        reasoning_response = generate_response(reasoning_model, reasoning_tokenizer, prompt, device)

        # Get activations for full sequences with chat template
        base_acts, base_len = get_activations(base_model, base_tokenizer, prompt, device, max_length,
                                              apply_chat_template=True, response=base_response)
        reasoning_acts, reasoning_len = get_activations(reasoning_model, reasoning_tokenizer, prompt, device, max_length,
                                                       apply_chat_template=True, response=reasoning_response)

        min_len = min(base_len, reasoning_len)

        # Calculate prompt length (with chat template)
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = base_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_len = base_tokenizer(formatted_prompt, return_tensors="pt").input_ids.shape[1]

        # Focus on reasoning part (generated tokens, especially the latter half)
        thinking_start = max(prompt_len, int(min_len * 0.6))
        thinking_end = min_len

        if thinking_end <= thinking_start:
            continue

        # Extract differences for each layer
        for layer_idx in range(num_layers):
            base_h = base_acts[layer_idx][:min_len]
            reasoning_h = reasoning_acts[layer_idx][:min_len]

            diff = reasoning_h[thinking_start:thinking_end] - base_h[thinking_start:thinking_end]
            avg_diff = diff.mean(dim=0)
            layer_diffs[layer_idx].append(avg_diff)

    # Aggregate and normalize
    print("\nAggregating and normalizing steering vectors...")

    steering_vectors = []
    steering_stats = []

    for layer_idx in range(num_layers):
        if len(layer_diffs[layer_idx]) == 0:
            steering_vectors.append(torch.zeros(hidden_size))
            steering_stats.append({
                'layer': layer_idx,
                'original_norm': 0.0,
                'norm': 0.0,
                'std': 0.0,
                'consistency': 0.0
            })
            continue

        diffs = torch.stack(layer_diffs[layer_idx])
        steering_vector = diffs.mean(dim=0)

        # Calculate statistics
        norm = steering_vector.norm().item()
        std = diffs.std(dim=0).mean().item()

        # Calculate consistency (average cosine similarity between normalized diffs)
        normalized_diffs = diffs / (diffs.norm(dim=1, keepdim=True) + 1e-8)
        similarity_matrix = torch.mm(normalized_diffs, normalized_diffs.t())
        mask = torch.triu(torch.ones_like(similarity_matrix), diagonal=1).bool()
        consistency = similarity_matrix[mask].mean().item()

        # Normalize to unit vector
        normalized_sv = steering_vector / (norm + 1e-8)

        steering_vectors.append(normalized_sv)
        steering_stats.append({
            'layer': layer_idx,
            'original_norm': norm,
            'norm': normalized_sv.norm().item(),  # Should be close to 1.0
            'std': std,
            'consistency': consistency
        })

    print("✓ Steering vectors extracted and normalized")

    return steering_vectors, steering_stats


def analyze_layer_groups(steering_stats, layer_groups):
    """
    Analyze steering vector statistics by layer groups

    Args:
        steering_stats: List of statistics for each layer
        layer_groups: Dictionary of layer groups

    Returns:
        best_layers_per_group: Best layers for each group
    """
    print("\n" + "="*60)
    print("Layer Group Analysis")
    print("="*60)

    print("\nLayer Statistics by Group:")
    print(f"{'Group':<12} {'Layers':<15} {'Avg Orig Norm':<15} {'Avg Consistency':<15}")
    print("-" * 60)

    for group_name, layers in layer_groups.items():
        group_stats = [s for s in steering_stats if s['layer'] in layers]
        if len(group_stats) > 0:
            avg_norm = np.mean([s['original_norm'] for s in group_stats])
            avg_consistency = np.mean([s['consistency'] for s in group_stats])
            print(f"{group_name:<12} {str(layers[0]) + '-' + str(layers[-1]):<15} "
                  f"{avg_norm:<15.2f} {avg_consistency:<15.4f}")

    # Select best layers from middle group
    best_layers_per_group = {}
    group_name = 'middle'
    layers = layer_groups[group_name]
    group_stats = [s for s in steering_stats if s['layer'] in layers]

    # Combined score: consistency * log(original_norm + 1)
    for s in group_stats:
        s['score'] = s['consistency'] * np.log(s['original_norm'] + 1)

    sorted_group = sorted(group_stats, key=lambda x: x['score'], reverse=True)
    best_layers_per_group[group_name] = [s['layer'] for s in sorted_group[:5]]

    print(f"\n{group_name.capitalize()} Group - Top 5 Layers:")
    for stat in sorted_group[:5]:
        print(f"  Layer {stat['layer']}: orig_norm={stat['original_norm']:.2f}, "
              f"consistency={stat['consistency']:.4f}, score={stat['score']:.4f}")

    return best_layers_per_group


def save_steering_vectors(steering_vectors, steering_stats, save_path,
                         layer_groups=None, best_layers_per_group=None):
    """
    Save steering vectors to file

    Args:
        steering_vectors: List of steering vectors
        steering_stats: List of statistics
        save_path: Path to save to
        layer_groups: (Optional) Dictionary of layer groups
        best_layers_per_group: (Optional) Best layers for each group
    """
    save_dict = {
        'steering_vectors': [v.cpu() for v in steering_vectors],
        'steering_stats': steering_stats,
        'version': '1.0_modular_simplified'
    }

    # Only add layer group info if provided
    if layer_groups is not None:
        save_dict['layer_groups'] = layer_groups
    if best_layers_per_group is not None:
        save_dict['best_layers_per_group'] = best_layers_per_group

    torch.save(save_dict, save_path)

    print(f"✓ Saved steering vectors to: {save_path}")


def load_steering_vectors(load_path):
    """
    Load steering vectors from file

    Args:
        load_path: Path to load from

    Returns:
        Dictionary containing steering vectors and metadata
    """
    data = torch.load(load_path)
    print(f"✓ Loaded steering vectors from: {load_path}")
    return data
