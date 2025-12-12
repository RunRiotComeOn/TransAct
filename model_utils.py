"""
Model loading and generation utilities
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name, device="cuda", torch_dtype="bfloat16"):
    """
    Load model and tokenizer

    Args:
        model_name: HuggingFace model name
        device: Device to load model on
        torch_dtype: Data type for model weights

    Returns:
        model, tokenizer
    """
    print(f"\nLoading {model_name}...")

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32
    }
    dtype = dtype_map.get(torch_dtype, torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    print(f"✓ Loaded {model_name}")
    return model, tokenizer


def get_activations(model, tokenizer, text, device, max_length=2048, apply_chat_template=False, response=None):
    """
    Get model activations for a text

    Args:
        model: The model
        tokenizer: The tokenizer
        text: Input text (prompt if apply_chat_template=True, otherwise full text)
        device: Device
        max_length: Maximum sequence length
        apply_chat_template: Whether to apply chat template
        response: Response text (required if apply_chat_template=True)

    Returns:
        activations: List of hidden states for each layer
        seq_length: Sequence length
    """
    if apply_chat_template:
        messages = [
            {"role": "user", "content": text},
            {"role": "assistant", "content": response}
        ]
        formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = tokenizer(formatted_text, return_tensors="pt", truncation=True,
                          max_length=max_length).to(device)
    else:
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                          max_length=max_length).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)

    hidden_states = outputs.hidden_states[1:]  # Skip embedding layer
    activations = [h.squeeze(0).cpu() for h in hidden_states]

    return activations, inputs.input_ids.shape[1]


def generate_response(model, tokenizer, prompt, device, max_new_tokens=1000):
    """
    Generate response from model

    Args:
        model: The model
        tokenizer: The tokenizer
        prompt: Input prompt
        device: Device
        max_new_tokens: Maximum tokens to generate

    Returns:
        generated_text: Generated response
    """
    # Apply chat template for instruction-tuned models
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )

    return generated_text


def generate_with_steering(model, tokenizer, prompt, steering_vectors,
                           layer_indices, alpha=1.0, max_new_tokens=1000):
    """
    Generate response with steering vectors applied

    Args:
        model: The model
        tokenizer: The tokenizer
        prompt: Input prompt
        steering_vectors: List of steering vectors (one per layer)
        layer_indices: Layers to apply steering to
        alpha: Steering strength
        max_new_tokens: Maximum tokens to generate

    Returns:
        generated_text: Generated response with steering applied
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]

    hooks = []

    def create_steering_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
                rest = output[1:]
            else:
                hidden_states = output
                rest = None

            if not hidden_states.requires_grad:
                hidden_states = hidden_states.clone()

            current_len = hidden_states.shape[1]

            # Only apply steering to generated tokens (not input tokens)
            if current_len > input_len:
                sv = steering_vectors[layer_idx].to(hidden_states.device).to(hidden_states.dtype)
                hidden_states[:, input_len:, :] = hidden_states[:, input_len:, :] + alpha * sv

            if rest is not None:
                return (hidden_states,) + rest
            else:
                return hidden_states

        return hook_fn

    # Register hooks
    for layer_idx in layer_indices:
        hook = model.model.layers[layer_idx].register_forward_hook(
            create_steering_hook(layer_idx)
        )
        hooks.append(hook)

    # Generate with steering
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=False,  # Important: disable cache for steering
            pad_token_id=tokenizer.eos_token_id
        )

    # Remove hooks
    for hook in hooks:
        hook.remove()

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


def get_model_info(model):
    """
    Get model information

    Args:
        model: The model

    Returns:
        dict: Model information (num_layers, hidden_size)
    """
    return {
        'num_layers': model.config.num_hidden_layers,
        'hidden_size': model.config.hidden_size
    }
