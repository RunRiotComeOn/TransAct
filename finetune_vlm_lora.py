#!/usr/bin/env python3
"""
Fine-tune VLM with LoRA on the same data used for steering vector extraction

This script:
1. Loads the same training data used for steering vector extraction
2. Fine-tunes the VLM using LoRA
3. Evaluates on the same test sets used for TransAct evaluation

Usage:
    python finetune_vlm_lora.py --steering_type mathematical_reasoning_gsm8k_alpha5.0_7b \
                                --target_dataset geometry3k \
                                --num_epochs 3 \
                                --rank 8
"""

import os
import json
import argparse
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset, DataLoader
from qwen_vl_utils import process_vision_info

from dataset_loaders import load_dataset_by_name
from dataset_loaders_vlm import (
    load_vlm_dataset,
    format_vlm_prompt,
    extract_answer_from_response,
    evaluate_vlm_answer
)


class VLMFineTuningDataset(Dataset):
    """Dataset for VLM fine-tuning"""

    def __init__(self, samples, processor, include_cot=True):
        self.samples = samples
        self.processor = processor
        self.include_cot = include_cot

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Format prompt
        prompt = format_vlm_prompt(sample, include_cot=self.include_cot)

        # Get answer
        answer = str(sample['answer'])

        # Create messages with both prompt and answer
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": sample['image']},
                {"type": "text", "text": prompt}
            ]
        }, {
            "role": "assistant",
            "content": answer
        }]

        # Process with full conversation
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        image_inputs, video_inputs = process_vision_info(messages)

        # Tokenize full sequence
        full_inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            return_tensors="pt"
        )

        # Create labels (same as input_ids, but mask prompt part)
        # First, get prompt-only length
        prompt_messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": sample['image']},
                {"type": "text", "text": prompt}
            ]
        }]
        prompt_text = self.processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        prompt_inputs = self.processor(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            return_tensors="pt"
        )
        prompt_length = prompt_inputs.input_ids.shape[1]

        # Create labels: mask prompt part with -100, keep answer part
        labels = full_inputs.input_ids.clone()
        labels[:, :prompt_length] = -100

        return {
            'input_ids': full_inputs.input_ids.squeeze(0),
            'attention_mask': full_inputs.attention_mask.squeeze(0),
            'pixel_values': full_inputs.pixel_values.squeeze(0) if 'pixel_values' in full_inputs else None,
            'image_grid_thw': full_inputs.image_grid_thw.squeeze(0) if 'image_grid_thw' in full_inputs else None,
            'labels': labels.squeeze(0),
            'prompt_length': prompt_length
        }


def load_training_data_from_steering_source(steering_type, num_samples=50):
    """
    Load the same training data used for steering vector extraction

    Args:
        steering_type: e.g., "mathematical_reasoning_gsm8k_alpha5.0_7b"
        num_samples: Number of samples (default: 50, same as steering extraction)

    Returns:
        List of text-based training samples (question, answer pairs)
    """
    # Parse steering type to get dataset name
    # Format: {ability}_{dataset}_{alpha}_{model}
    parts = steering_type.split('_')

    # Map ability to dataset
    ability_to_dataset = {
        'mathematical': 'gsm8k',
        'science': 'arc_challenge',
        'logical': 'winogrande',
        'commonsense': 'commonsense_qa',
        'multihop': 'hotpotqa',
        'reading': 'boolq'
    }

    # Extract ability
    ability = parts[0]  # e.g., "mathematical"

    # Try to find dataset name
    if 'gsm8k' in steering_type.lower():
        dataset_name = 'gsm8k'
    elif 'math' in steering_type.lower() and '50' in steering_type:
        dataset_name = 'hendrycks_math'
        num_samples = 50  # MATH dataset uses 50 samples
    elif 'arc' in steering_type.lower():
        dataset_name = 'arc_challenge'
    elif 'winogrande' in steering_type.lower():
        dataset_name = 'winogrande'
    elif 'commonsense' in steering_type.lower():
        dataset_name = 'commonsense_qa'
    elif 'hotpot' in steering_type.lower():
        dataset_name = 'hotpotqa'
    elif 'boolq' in steering_type.lower():
        dataset_name = 'boolq'
    else:
        raise ValueError(f"Cannot parse dataset from steering_type: {steering_type}")

    print(f"\nLoading training data from: {dataset_name}")
    print(f"Number of samples: {num_samples}")

    # Load dataset
    data = load_dataset_by_name(
        dataset_name,
        split='train',
        num_samples=num_samples * 3,  # Load more for filtering
        random_seed=42
    )

    # Return first num_samples (same as steering extraction)
    return data[:num_samples]


def create_vlm_training_data(text_questions, vlm_dataset):
    """
    Pair text questions with images from VLM dataset

    We need to convert text-based training questions into multimodal format
    by pairing them with images from the target VLM dataset.

    Args:
        text_questions: List of {'question': str, 'answer': str}
        vlm_dataset: VLM dataset samples with images

    Returns:
        List of multimodal training samples
    """
    training_samples = []

    # Use images from VLM dataset (cycle through if needed)
    num_images = len(vlm_dataset)

    for idx, text_sample in enumerate(text_questions):
        # Get corresponding image (cycle if needed)
        vlm_sample = vlm_dataset[idx % num_images]

        # Create multimodal training sample
        training_samples.append({
            'image': vlm_sample['image'],
            'question': text_sample['question'],
            'answer': text_sample['answer'],
            'metadata': {
                'dataset': vlm_sample['metadata']['dataset'],
                'source': 'lora_training',
                'original_text_question': text_sample['question']
            }
        })

    return training_samples


def collate_fn(batch):
    """Custom collate function for DataLoader"""
    # Get max length
    max_len = max(item['input_ids'].shape[0] for item in batch)

    # Pad sequences
    input_ids = []
    attention_mask = []
    labels = []
    pixel_values = []
    image_grid_thw = []

    for item in batch:
        # Pad input_ids and attention_mask
        pad_len = max_len - item['input_ids'].shape[0]
        input_ids.append(torch.cat([
            item['input_ids'],
            torch.zeros(pad_len, dtype=torch.long)
        ]))
        attention_mask.append(torch.cat([
            item['attention_mask'],
            torch.zeros(pad_len, dtype=torch.long)
        ]))

        # Pad labels
        labels.append(torch.cat([
            item['labels'],
            torch.full((pad_len,), -100, dtype=torch.long)
        ]))

        # Collect vision inputs
        if item['pixel_values'] is not None:
            pixel_values.append(item['pixel_values'])
        if item['image_grid_thw'] is not None:
            image_grid_thw.append(item['image_grid_thw'])

    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_mask),
        'labels': torch.stack(labels),
        'pixel_values': torch.stack(pixel_values) if pixel_values else None,
        'image_grid_thw': torch.stack(image_grid_thw) if image_grid_thw else None
    }


def evaluate_model(model, processor, test_dataset):
    """
    Evaluate model on test dataset

    Args:
        model: Fine-tuned VLM model
        processor: VLM processor
        test_dataset: List of test samples

    Returns:
        Dictionary with accuracy and other metrics
    """
    model.eval()
    correct = 0
    total = 0

    dataset_name = test_dataset[0]['metadata']['dataset']

    for sample in tqdm(test_dataset, desc="Evaluating"):
        try:
            # Format prompt
            prompt = format_vlm_prompt(sample, include_cot=True)

            # Prepare input
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": sample['image']},
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

            # Generate
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    temperature=0.1,
                    do_sample=False
                )

            # Decode
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            # Extract and evaluate answer
            predicted = extract_answer_from_response(
                response,
                dataset_name,
                sample['metadata'].get('question_type'),
                sample['metadata'].get('answer_type'),
                sample['metadata'].get('choices')
            )
            expected = sample['answer']

            is_correct = evaluate_vlm_answer(
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
            total += 1
            continue

    accuracy = correct / total if total > 0 else 0

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune VLM with LoRA on steering vector training data"
    )
    parser.add_argument("--steering_type", type=str, required=True,
                       help="Steering type (e.g., mathematical_reasoning_gsm8k_alpha5.0_7b)")
    parser.add_argument("--target_dataset", type=str, required=True,
                       help="Target VLM dataset (e.g., geometry3k)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                       help="VLM model to fine-tune")
    parser.add_argument("--num_train_samples", type=int, default=50,
                       help="Number of training samples (same as steering extraction)")
    parser.add_argument("--num_test_samples", type=int, default=100,
                       help="Number of test samples (same as TransAct evaluation)")
    parser.add_argument("--rank", type=int, default=8,
                       help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=16,
                       help="LoRA alpha")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="./lora_results",
                       help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    print("\n" + "="*80)
    print("VLM LoRA FINE-TUNING")
    print("="*80)
    print(f"Steering Type: {args.steering_type}")
    print(f"Target Dataset: {args.target_dataset}")
    print(f"Training Samples: {args.num_train_samples}")
    print(f"Test Samples: {args.num_test_samples}")
    print(f"LoRA Rank: {args.rank}")
    print(f"LoRA Alpha: {args.alpha}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Learning Rate: {args.learning_rate}")
    print("="*80)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"{args.steering_type}_{args.target_dataset}_lora_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load text-based training data (same as steering extraction)
    print("\n" + "="*80)
    print("LOADING TRAINING DATA")
    print("="*80)
    text_training_data = load_training_data_from_steering_source(
        args.steering_type,
        args.num_train_samples
    )
    print(f"✓ Loaded {len(text_training_data)} text training samples")

    # 2. Load VLM target dataset
    print("\n" + "="*80)
    print("LOADING VLM DATASET")
    print("="*80)
    split = "testmini" if args.target_dataset == "mathvista" else "test"
    vlm_full_dataset = load_vlm_dataset(
        args.target_dataset,
        split=split,
        num_samples=args.num_train_samples + args.num_test_samples,
        random_seed=args.seed
    )

    # Split into train and test (use first samples for training, rest for testing)
    vlm_train_images = vlm_full_dataset[:args.num_train_samples]
    vlm_test_dataset = vlm_full_dataset[args.num_train_samples:args.num_train_samples + args.num_test_samples]

    print(f"✓ Loaded {len(vlm_train_images)} VLM training images")
    print(f"✓ Loaded {len(vlm_test_dataset)} VLM test samples")

    # 3. Create multimodal training data
    print("\n" + "="*80)
    print("CREATING MULTIMODAL TRAINING DATA")
    print("="*80)
    training_samples = create_vlm_training_data(text_training_data, vlm_train_images)
    print(f"✓ Created {len(training_samples)} multimodal training samples")

    # 4. Load model and processor
    print("\n" + "="*80)
    print("LOADING MODEL")
    print("="*80)

    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    print(f"✓ Loaded model: {args.model}")

    # 5. Configure LoRA
    print("\n" + "="*80)
    print("CONFIGURING LORA")
    print("="*80)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 6. Prepare dataset and dataloader
    print("\n" + "="*80)
    print("PREPARING TRAINING")
    print("="*80)

    train_dataset = VLMFineTuningDataset(training_samples, processor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    # 7. Training
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    model.train()

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        total_loss = 0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            # Move to device
            batch = {k: v.to(model.device) if torch.is_tensor(v) else v
                    for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"  Batch {batch_idx + 1}: Loss = {avg_loss:.4f}")

        avg_epoch_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} Average Loss: {avg_epoch_loss:.4f}")

    # 8. Evaluation
    print("\n" + "="*80)
    print("EVALUATION")
    print("="*80)

    results = evaluate_model(model, processor, vlm_test_dataset)

    print(f"\nTest Accuracy: {results['accuracy']:.4f}")
    print(f"Correct: {results['correct']}/{results['total']}")

    # 9. Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    # Save model
    model.save_pretrained(output_dir / "lora_model")
    processor.save_pretrained(output_dir / "processor")

    # Save results
    results_dict = {
        'method': 'lora_finetuning',
        'steering_type': args.steering_type,
        'target_dataset': args.target_dataset,
        'num_train_samples': args.num_train_samples,
        'num_test_samples': args.num_test_samples,
        'lora_config': {
            'rank': args.rank,
            'alpha': args.alpha,
            'target_modules': list(lora_config.target_modules) if isinstance(lora_config.target_modules, set) else lora_config.target_modules
        },
        'training_config': {
            'num_epochs': args.num_epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate
        },
        'results': results,
        'timestamp': timestamp
    }

    with open(output_dir / "results.json", 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"✓ Saved model to: {output_dir / 'lora_model'}")
    print(f"✓ Saved results to: {output_dir / 'results.json'}")

    print("\n" + "="*80)
    print("FINE-TUNING COMPLETE")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
