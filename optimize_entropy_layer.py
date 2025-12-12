"""
Entropy-based dynamic layer selection for VLM steering
Selects the layer with maximum entropy CHANGE (steepest entropy drop) to apply steering

NEW STRATEGY: Find the "turning point" where model starts to converge
- In shallow layers: high entropy (uncertainty)
- At turning point: entropy drops sharply (model making decision)
- Apply steering at L_i where H(L_i) - H(L_{i+1}) is maximal
- This is the critical "crossroads" before the model commits to its answer

Inspired by contrastive decoding - steering at the "critical thinking" point
"""

import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class EntropyLayerSelector:
    """
    Dynamically selects which layer to apply steering based on entropy change

    Uses the "turning point" strategy: finds layer L_i where entropy drop
    H(L_i) - H(L_{i+1}) is maximal, indicating the critical decision point.
    """

    def __init__(self, model, steering_vectors: List[torch.Tensor],
                 alpha: float = 5.0, mode: str = 'per_prompt'):
        """
        Initialize entropy-based layer selector

        Args:
            model: VLM model
            steering_vectors: List of steering vectors (one per layer)
            alpha: Fixed steering strength
            mode: 'per_prompt' (compute once per prompt) or 'per_token' (compute per token)
        """
        self.model = model
        self.steering_vectors = steering_vectors
        self.alpha = alpha
        self.mode = mode

        # Get output embedding matrix for projecting hidden states to vocabulary
        self.W_unembed = model.get_output_embeddings().weight  # [vocab_size, hidden_size]

        # Track selected layers for analysis
        self.selected_layers_history = []
        self.entropy_history = []

        # Hook handles
        self.hooks = []
        self.target_layer = None  # Will be set dynamically
        self.current_step = 0

    def compute_layer_entropies(self, hidden_states: Tuple[torch.Tensor]) -> torch.Tensor:
        """
        Compute entropy for each layer's prediction distribution

        Args:
            hidden_states: Tuple of hidden states from all layers
                          Each tensor shape: [batch_size, seq_len, hidden_size]

        Returns:
            entropies: Tensor of shape [num_layers, batch_size]
        """
        all_layer_entropies = []

        # Skip embedding layer (index 0), process decoder layers (index 1 onwards)
        for layer_idx, h_i in enumerate(hidden_states[1:]):
            # Get last token's hidden state
            last_token_hidden = h_i[:, -1, :]  # [batch_size, hidden_size]

            # Project to vocabulary space
            # IMPORTANT: Ensure W_unembed is on same device as hidden states
            W_unembed = self.W_unembed.to(last_token_hidden.device)
            logits_i = last_token_hidden @ W_unembed.T  # [batch_size, vocab_size]

            # Compute probability distribution
            log_probs_i = torch.log_softmax(logits_i, dim=-1)
            probs_i = torch.softmax(logits_i, dim=-1)

            # Compute entropy: H(X) = -sum(P(x) * log(P(x)))
            # Use stable computation to avoid numerical issues
            entropy_i = -(probs_i * log_probs_i).sum(dim=-1)  # [batch_size]

            all_layer_entropies.append(entropy_i)

        # Stack to [num_layers, batch_size]
        entropies_tensor = torch.stack(all_layer_entropies)

        return entropies_tensor

    def select_max_entropy_layer(self, hidden_states: Tuple[torch.Tensor]) -> int:
        """
        Select from top-2 layers with maximum entropy change (steepest entropy drop)

        **TOP-2 STRATEGY**: Find the two "turning points" with largest entropy drops,
        then randomly select one to increase diversity.

        Core logic:
        - In shallow layers: high entropy (uncertainty)
        - At the "turning point": model starts to converge, entropy drops sharply
        - This sharp drop moment is the critical "crossroads" for reasoning
        - We find top-2 candidates and randomly pick one for diversity

        Args:
            hidden_states: Tuple of hidden states from all layers

        Returns:
            layer_idx: Index of layer selected from top-2 entropy drops
        """
        entropies = self.compute_layer_entropies(hidden_states)  # [num_layers, batch_size]

        # For simplicity, average across batch (or take first element if batch_size=1)
        avg_entropies = entropies.mean(dim=1)  # [num_layers]

        # **NEW**: Compute entropy change (entropy drop) between consecutive layers
        # H(L_i) - H(L_{i+1})
        entropies_current_layer = avg_entropies[:-1]  # H(L_i) for i=0 to num_layers-2
        entropies_next_layer = avg_entropies[1:]      # H(L_{i+1}) for i=0 to num_layers-2

        # Entropy deltas: positive value means entropy is dropping
        # Shape: [num_layers-1]
        entropy_deltas = entropies_current_layer - entropies_next_layer

        # **TOP-2 SELECTION**: Find top-2 layers with largest entropy drops
        top_k = min(2, len(entropy_deltas))  # Handle edge case if < 2 layers
        top_k_values, top_k_indices = torch.topk(entropy_deltas, k=top_k)

        # Randomly select one from top-2 for diversity
        selected_idx = np.random.randint(0, top_k)
        max_delta_layer = top_k_indices[selected_idx].item()

        # Store for analysis
        self.entropy_history.append(avg_entropies.detach().cpu().float().numpy())
        self.selected_layers_history.append(max_delta_layer)

        # Note: max_delta_layer is 0-indexed relative to decoder layers
        # This represents L_i where entropy is high and about to drop sharply
        return max_delta_layer

    def create_dynamic_hook(self, layer_idx: int):
        """
        Create a forward hook that applies steering only when this is the target layer

        Args:
            layer_idx: Current layer index

        Returns:
            hook_fn: Hook function
        """
        def hook_fn(module, input, output):
            # Check if this is the target layer for steering
            if self.target_layer is not None and layer_idx == self.target_layer:
                if isinstance(output, tuple):
                    hidden_states = output[0]
                    rest = output[1:]
                else:
                    hidden_states = output
                    rest = None

                # Apply steering to last token
                if hidden_states.shape[1] > 0 and layer_idx < len(self.steering_vectors):
                    # Ensure steering vector is on same device and dtype as hidden states
                    sv = self.steering_vectors[layer_idx].to(
                        device=hidden_states.device,
                        dtype=hidden_states.dtype
                    )
                    # Create a copy to avoid in-place modification issues
                    hidden_states = hidden_states.clone()
                    hidden_states[:, -1, :] = hidden_states[:, -1, :] + self.alpha * sv

                return (hidden_states,) + rest if rest is not None else hidden_states

            # No modification if not target layer
            return output

        return hook_fn

    def register_hooks(self):
        """Register forward hooks on all decoder layers"""
        self.hooks = []

        try:
            num_layers = len(self.model.model.language_model.layers)
            for layer_idx in range(num_layers):
                hook = self.model.model.language_model.layers[layer_idx].register_forward_hook(
                    self.create_dynamic_hook(layer_idx)
                )
                self.hooks.append(hook)
        except AttributeError as e:
            print(f"Error registering hooks: {e}")
            raise

    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_statistics(self) -> Dict:
        """
        Get statistics about selected layers and entropies

        Returns:
            Dictionary with statistics
        """
        if not self.selected_layers_history:
            return {}

        selected_layers = np.array(self.selected_layers_history)

        return {
            'mean_selected_layer': np.mean(selected_layers),
            'std_selected_layer': np.std(selected_layers),
            'most_common_layer': np.bincount(selected_layers).argmax(),
            'layer_distribution': np.bincount(selected_layers).tolist(),
            'num_samples': len(selected_layers)
        }


def generate_with_entropy_steering(model, processor, image, prompt: str,
                                   selector: EntropyLayerSelector,
                                   max_new_tokens: int = 512,
                                   temperature: float = 0.1) -> Dict:
    """
    Generate response with entropy-based dynamic layer selection

    Args:
        model: VLM model
        processor: VLM processor
        image: Input image
        prompt: Text prompt
        selector: EntropyLayerSelector instance
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature

    Returns:
        Dictionary with response and selected layer info
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

    # Register hooks
    selector.register_hooks()

    # Per-prompt mode: Compute target layer once before generation
    if selector.mode == 'per_prompt':
        # First, do a forward pass to get hidden states
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
            hidden_states = outputs.hidden_states

        # Select target layer based on entropy
        target_layer = selector.select_max_entropy_layer(hidden_states)
        selector.target_layer = target_layer

        # Now generate with the selected layer
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0
            )

    # Per-token mode: Would require custom generation loop (much more complex)
    # For now, we only implement per-prompt mode
    else:
        raise NotImplementedError("Per-token mode not yet implemented. Use 'per_prompt' mode.")

    # Remove hooks
    selector.remove_hooks()

    # Decode output
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return {
        'response': response_text,
        'selected_layer': selector.target_layer,
        'alpha': selector.alpha
    }


def evaluate_with_entropy_steering(model, processor, dataset: List[Dict],
                                   selector: EntropyLayerSelector,
                                   extract_answer_fn,
                                   evaluate_answer_fn,
                                   max_new_tokens: int = 512) -> Dict:
    """
    Evaluate model on dataset using entropy-based steering

    Args:
        model: VLM model
        processor: VLM processor
        dataset: List of samples
        selector: EntropyLayerSelector instance
        extract_answer_fn: Function to extract answers
        evaluate_answer_fn: Function to evaluate answers
        max_new_tokens: Max tokens to generate

    Returns:
        Dictionary with evaluation results
    """
    correct = 0
    total = 0
    selected_layers = []

    dataset_name = dataset[0]['metadata']['dataset'] if dataset else 'unknown'

    for sample in tqdm(dataset, desc="Evaluating with entropy steering"):
        try:
            from dataset_loaders_vlm import format_vlm_prompt
            prompt = format_vlm_prompt(sample, include_cot=True)

            result = generate_with_entropy_steering(
                model, processor,
                sample['image'],
                prompt,
                selector=selector,
                max_new_tokens=max_new_tokens
            )

            response = result['response']
            selected_layers.append(result['selected_layer'])

            # Extract and evaluate answer
            predicted = extract_answer_fn(
                response,
                dataset_name,
                sample['metadata'].get('question_type'),
                sample['metadata'].get('answer_type'),
                sample['metadata'].get('choices')
            )
            expected = sample['answer']

            # Evaluate with all metadata
            is_correct = evaluate_answer_fn(
                predicted, expected, dataset_name,
                all_answers=sample['metadata'].get('all_answers'),
                question_type=sample['metadata'].get('question_type'),
                answer_type=sample['metadata'].get('answer_type')
            )

            if is_correct:
                correct += 1
            total += 1

        except Exception as e:
            print(f"Error evaluating sample: {e}")
            import traceback
            traceback.print_exc()
            total += 1
            continue

    accuracy = correct / total if total > 0 else 0

    # Get selector statistics
    selector_stats = selector.get_statistics()

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'selected_layers': selected_layers,
        'selector_statistics': selector_stats,
        'mean_selected_layer': np.mean(selected_layers) if selected_layers else None,
        'layer_distribution': np.bincount(selected_layers).tolist() if selected_layers else []
    }


def optimize_alpha_with_entropy_layers(model, processor, steering_vectors: List[torch.Tensor],
                                       optimization_data: List[Dict],
                                       alpha_candidates: List[float],
                                       extract_answer_fn,
                                       evaluate_answer_fn,
                                       mode: str = 'per_prompt') -> Dict:
    """
    Optimize only alpha parameter while using entropy-based layer selection

    Args:
        model: VLM model
        processor: VLM processor
        steering_vectors: Steering vectors
        optimization_data: Dataset for optimization
        alpha_candidates: List of alpha values to try
        extract_answer_fn: Function to extract answers
        evaluate_answer_fn: Function to evaluate answers
        mode: 'per_prompt' or 'per_token'

    Returns:
        Dictionary with best alpha and results
    """
    print("\n" + "="*60)
    print("ENTROPY-BASED DYNAMIC LAYER SELECTION")
    print("="*60)
    print(f"Mode: {mode}")
    print(f"Alpha candidates: {alpha_candidates}")
    print("="*60)

    best_score = -float('inf')
    best_alpha = None
    best_results = None
    all_results = []

    for alpha in tqdm(alpha_candidates, desc="Testing alpha values"):
        # Create selector with current alpha
        selector = EntropyLayerSelector(
            model=model,
            steering_vectors=steering_vectors,
            alpha=alpha,
            mode=mode
        )

        # Evaluate
        results = evaluate_with_entropy_steering(
            model, processor,
            optimization_data,
            selector=selector,
            extract_answer_fn=extract_answer_fn,
            evaluate_answer_fn=evaluate_answer_fn
        )

        score = results['accuracy']
        results['alpha'] = alpha
        all_results.append(results)

        # Update best
        if score > best_score:
            best_score = score
            best_alpha = alpha
            best_results = results

            print(f"\n✓ New best alpha: {alpha:.1f}, accuracy: {score:.4f}")
            print(f"  Mean selected layer: {results['mean_selected_layer']:.1f}")
            print(f"  Selector stats: {results['selector_statistics']}")

    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"Best Alpha: {best_alpha}")
    print(f"Best Accuracy: {best_score:.4f}")
    print(f"Mean Selected Layer: {best_results['mean_selected_layer']:.1f}")
    print("="*60)

    return {
        'best_alpha': best_alpha,
        'best_accuracy': best_score,
        'best_results': best_results,
        'all_results': all_results,
        'mode': mode
    }

