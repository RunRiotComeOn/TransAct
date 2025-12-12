"""
Dataset loading and formatting functions for different reasoning abilities
"""

from datasets import load_dataset
from tqdm import tqdm
import re


def extract_answer(response):
    """
    Extract answer from model response

    Handles various formats:
    - GSM8K: #### followed by number
    - Multiple choice: (A), A), letter patterns
    - General: "answer is X", "= X"
    """
    response = response.strip()

    if not response:
        return ""

    # GSM8K format: #### number
    match = re.search(r'####\s*([-+]?\d*\.?\d+)', response)
    if match:
        return match.group(1)

    # Multiple choice: look for letter answers
    # Pattern: "answer is (A)" or "answer is A" or "(A)" at end
    mc_patterns = [
        r'(?:answer|choice)(?:\s+is)?(?:\s*:)?\s*\(?([A-E])\)?',
        r'\(([A-E])\)\s*$',
        r'\b([A-E])\)\s*$',
    ]
    for pattern in mc_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # General patterns: "answer is X", "= X"
    patterns = [
        r'(?:final\s+)?answer(?:\s+is)?(?:\s*:)?\s*["\']?([^"\'\n,]+)["\']?',
        r'=\s*([^\n=]+?)\s*$',
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            extraction = match.group(1).strip()
            # Try to extract number if present
            num_match = re.search(r'[-+]?\d*\.?\d+', extraction)
            if num_match:
                return num_match.group(0)
            return extraction

    # Fallback: last number in response
    numbers = re.findall(r'[-+]?\d*\.?\d+', response)
    if numbers:
        return numbers[-1]

    # Last resort: last line
    return response.split('\n')[-1].strip()


def evaluate_answer(predicted, expected):
    """
    Evaluate if predicted answer matches expected answer

    Args:
        predicted: Extracted predicted answer
        expected: Ground truth answer

    Returns:
        True if correct, False otherwise
    """
    if not predicted or not expected:
        return False

    predicted = str(predicted).strip()
    expected = str(expected).strip()

    # Direct match (case-insensitive)
    if predicted.upper() == expected.upper():
        return True

    # Multiple choice: compare letters
    pred_letter = re.search(r'^([A-E])$', predicted.upper())
    exp_letter = re.search(r'^([A-E])$', expected.upper())
    if pred_letter and exp_letter:
        return pred_letter.group(1) == exp_letter.group(1)

    # Numerical comparison
    try:
        # Extract numbers
        pred_nums = re.findall(r'[-+]?\d*\.?\d+', predicted)
        exp_nums = re.findall(r'[-+]?\d*\.?\d+', expected)

        if pred_nums and exp_nums:
            pred_num = float(pred_nums[-1])
            exp_num = float(exp_nums[-1])

            # Exact match for integers
            if pred_num == exp_num:
                return True

            # Relative tolerance for floats
            if exp_num != 0:
                return abs(pred_num - exp_num) / abs(exp_num) < 0.01
            else:
                return abs(pred_num) < 1e-6
    except (ValueError, IndexError):
        pass

    # Boolean answers
    pred_bool = predicted.lower() in ['true', 'yes', '1']
    exp_bool = expected.lower() in ['true', 'yes', '1']
    pred_neg = predicted.lower() in ['false', 'no', '0']
    exp_neg = expected.lower() in ['false', 'no', '0']

    if (pred_bool and exp_bool) or (pred_neg and exp_neg):
        return True

    return False


def load_gsm8k(split, num_samples, random_seed=42):
    """Load GSM8K dataset for mathematical reasoning"""
    print(f"\nLoading {num_samples} samples from GSM8K ({split} split)...")

    dataset = load_dataset("gsm8k", "main", split=split)
    shuffled_dataset = dataset.shuffle(seed=random_seed)
    samples = shuffled_dataset.select(range(min(num_samples, len(shuffled_dataset))))

    formatted_prompts = []

    for item in tqdm(samples, desc=f"Formatting GSM8K {split} samples"):
        if split == 'train':
            prompt = f"""Question: {item['question']}

Please solve this problem step by step, and put your final numerical answer after "####" at the end.

Solution:"""
            # Extract answer from training data
            full_answer = item['answer']
            answer_str = full_answer.split('####')[-1].strip().replace(',', '')
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", answer_str)
            answer = numbers[-1] if numbers else ""

            formatted_prompts.append({
                'question': prompt,
                'answer': answer
            })

        elif split == 'test':
            try:
                full_answer_text = item['answer']
                answer_str = full_answer_text.split('####')[-1].strip().replace(',', '')
                numbers = re.findall(r"[-+]?\d*\.\d+|\d+", answer_str)
                if not numbers:
                    continue

                prompt = f"""Question: {item['question']}

Please solve this problem step by step. After your solution, write your final answer in the format:
"The final answer is: [your answer]"

Solution:"""

                formatted_prompts.append({
                    'prompt': prompt,
                    'expected_answer': numbers[-1],
                    'reference_answer': full_answer_text,
                    'reasoning_keywords': ['calculate', 'solve', 'math'],
                    'difficulty': 'math'
                })
            except Exception:
                continue

    print(f"✓ Prepared {len(formatted_prompts)} prompts from GSM8K")
    return formatted_prompts


def load_winogrande(split, num_samples, random_seed=42):
    """Load Winogrande dataset for logical reasoning (pronoun disambiguation)"""
    print(f"\nLoading {num_samples} samples from Winogrande ({split} split)...")

    # Map to actual split names
    actual_split = 'train' if split == 'train' else 'validation'
    dataset = load_dataset("winogrande", "winogrande_xl", split=actual_split)
    shuffled_dataset = dataset.shuffle(seed=random_seed)
    samples = shuffled_dataset.select(range(min(num_samples, len(shuffled_dataset))))

    formatted_prompts = []

    for item in tqdm(samples, desc=f"Formatting Winogrande {split} samples"):
        sentence = item['sentence']
        option1 = item['option1']
        option2 = item['option2']
        answer = item.get('answer', '')  # '1' or '2'

        # Replace underscore with blank
        question = sentence.replace('_', '______')

        prompt = f"""Fill in the blank with the correct option:

Sentence: {question}

Options:
A) {option1}
B) {option2}

Please reason which option fits better and provide the correct option letter (A or B):"""

        # Convert answer to letter
        if answer == '1':
            correct_option = 'A'
        elif answer == '2':
            correct_option = 'B'
        else:
            continue

        if split == 'train':
            formatted_prompts.append({
                'question': prompt,
                'answer': correct_option
            })

        else:  # validation
            formatted_prompts.append({
                'prompt': prompt,
                'expected_answer': option1 if answer == '1' else option2,
                'expected_option': correct_option,
                'reasoning_keywords': ['reason', 'context', 'logic', 'because'],
                'difficulty': 'logic'
            })

    print(f"✓ Prepared {len(formatted_prompts)} prompts from Winogrande")
    return formatted_prompts


def load_commonsense_qa(split, num_samples, random_seed=42):
    """Load CommonsenseQA dataset for commonsense reasoning"""
    print(f"\nLoading {num_samples} samples from CommonsenseQA ({split} split)...")

    dataset = load_dataset("commonsense_qa", split=split)
    shuffled_dataset = dataset.shuffle(seed=random_seed)
    samples = shuffled_dataset.select(range(min(num_samples, len(shuffled_dataset))))

    formatted_prompts = []

    for item in tqdm(samples, desc=f"Formatting CommonsenseQA {split} samples"):
        question = item['question']
        choices = item['choices']
        options_text = "\n".join([f"{label}) {text}" for label, text in zip(choices['label'], choices['text'])])
        correct_option_char = item['answerKey']
        correct_answer_text = choices['text'][choices['label'].index(correct_option_char)]

        prompt = f"""Question: {question}

Options:
{options_text}

Please analyze the options and provide the correct option letter:"""

        if split == 'train':
            formatted_prompts.append({
                'question': prompt,
                'answer': correct_option_char
            })
        elif split == 'validation':
            formatted_prompts.append({
                'prompt': prompt,
                'expected_answer': correct_answer_text,
                'expected_option': correct_option_char,
                'reasoning_keywords': ['common sense', 'because', 'therefore', 'since'],
                'difficulty': 'commonsense'
            })

    print(f"✓ Prepared {len(formatted_prompts)} prompts from CommonsenseQA")
    return formatted_prompts


def load_arc_challenge(split, num_samples, random_seed=42):
    """Load ARC-Challenge dataset for science reasoning"""
    print(f"\nLoading {num_samples} samples from ARC-Challenge ({split} split)...")

    # Map to actual split names
    actual_split = 'train' if split == 'train' else 'test'
    dataset = load_dataset("ai2_arc", "ARC-Challenge", split=actual_split)
    shuffled_dataset = dataset.shuffle(seed=random_seed)
    samples = shuffled_dataset.select(range(min(num_samples, len(shuffled_dataset))))

    formatted_prompts = []

    for item in tqdm(samples, desc=f"Formatting ARC-Challenge {split} samples"):
        question = item['question']
        choices = item['choices']
        options_text = "\n".join([f"{label}) {text}" for label, text in zip(choices['label'], choices['text'])])
        correct_option_char = item['answerKey']

        # Find correct answer text
        try:
            correct_idx = choices['label'].index(correct_option_char)
            correct_answer_text = choices['text'][correct_idx]
        except (ValueError, IndexError):
            continue

        prompt = f"""Question: {question}

Options:
{options_text}

Please analyze the options and provide the correct option letter:"""

        if split == 'train':
            formatted_prompts.append({
                'question': prompt,
                'answer': correct_option_char
            })
        else:
            formatted_prompts.append({
                'prompt': prompt,
                'expected_answer': correct_answer_text,
                'expected_option': correct_option_char,
                'reasoning_keywords': ['science', 'because', 'therefore', 'reason'],
                'difficulty': 'science'
            })

    print(f"✓ Prepared {len(formatted_prompts)} prompts from ARC-Challenge")
    return formatted_prompts


def load_boolq(split, num_samples, random_seed=42):
    """Load BoolQ dataset for reading comprehension"""
    print(f"\nLoading {num_samples} samples from BoolQ ({split} split)...")

    dataset = load_dataset("boolq", split=split)
    shuffled_dataset = dataset.shuffle(seed=random_seed)
    samples = shuffled_dataset.select(range(min(num_samples, len(shuffled_dataset))))

    formatted_prompts = []

    for item in tqdm(samples, desc=f"Formatting BoolQ {split} samples"):
        passage = item['passage']
        question = item['question']
        answer = item['answer']

        prompt = f"""Passage: {passage}

Question: {question}

Please reason step by step and answer Yes or No:"""

        expected_str = "Yes" if answer else "No"

        if split == 'train':
            formatted_prompts.append({
                'question': prompt,
                'answer': expected_str
            })
        elif split == 'validation':
            formatted_prompts.append({
                'prompt': prompt,
                'expected_answer': expected_str,
                'reference_answer': str(answer),
                'reasoning_keywords': ['reason', 'because', 'therefore', 'since', 'thus'],
                'difficulty': 'reading'
            })

    print(f"✓ Prepared {len(formatted_prompts)} prompts from BoolQ")
    return formatted_prompts


def load_hotpotqa(split, num_samples, random_seed=42):
    """Load HotpotQA dataset for multi-hop reasoning"""
    print(f"\nLoading {num_samples} samples from HotpotQA ({split} split)...")

    # Map to actual split names
    actual_split = 'train' if split == 'train' else 'validation'
    dataset = load_dataset("hotpot_qa", "distractor", split=actual_split)
    shuffled_dataset = dataset.shuffle(seed=random_seed)
    samples = shuffled_dataset.select(range(min(num_samples, len(shuffled_dataset))))

    formatted_prompts = []

    for item in tqdm(samples, desc=f"Formatting HotpotQA {split} samples"):
        question = item['question']
        context_paragraphs = item['context']
        answer = item.get('answer', '')

        # Combine context (take first 2 paragraphs to keep it manageable)
        context_parts = []
        titles = context_paragraphs['title']
        sentences_list = context_paragraphs['sentences']

        for i in range(min(2, len(titles))):  # Only use first 2 context paragraphs
            title = titles[i]
            sentences = sentences_list[i]
            context_text = ' '.join(sentences)
            context_parts.append(f"[{title}] {context_text}")

        context = "\n\n".join(context_parts)

        prompt = f"""Context:
{context}

Question: {question}

Please use multi-hop reasoning to answer the question:"""

        if not answer:
            continue

        if split == 'train':
            formatted_prompts.append({
                'question': prompt,
                'answer': answer
            })
        else:  # For evaluation
            formatted_prompts.append({
                'prompt': prompt,
                'expected_answer': answer,
                'reference_answer': answer,
                'reasoning_keywords': ['reason', 'because', 'therefore', 'multi-step', 'context'],
                'difficulty': 'multihop'
            })

    print(f"✓ Prepared {len(formatted_prompts)} prompts from HotpotQA")
    return formatted_prompts


def load_hendrycks_math(split, num_samples, random_seed=42):
    """
    Load MATH dataset (Hendrycks et al.) for mathematical reasoning

    This is a challenging dataset of competition-level math problems
    covering algebra, geometry, number theory, etc.

    Args:
        split: 'train' or 'test'
        num_samples: Number of samples to load
        random_seed: Random seed for reproducibility

    Returns:
        List of formatted question strings
    """
    print(f"\nLoading MATH dataset ({split} split)...")

    # MATH dataset has separate configs for each subject area
    # Load all subjects and combine them
    subjects = [
        'algebra',
        'counting_and_probability',
        'geometry',
        'intermediate_algebra',
        'number_theory',
        'prealgebra',
        'precalculus'
    ]

    all_items = []
    for subject in subjects:
        try:
            subject_dataset = load_dataset('EleutherAI/hendrycks_math', subject, split=split)
            print(f"  ✓ Loaded {len(subject_dataset)} samples from {subject}")
            all_items.extend(list(subject_dataset))
        except Exception as e:
            print(f"  Warning: Could not load {subject}: {e}")
            continue

    print(f"✓ Loaded {len(all_items)} total samples from MATH across {len(subjects)} subjects")

    # Shuffle and sample if requested
    import random
    random.seed(random_seed)
    random.shuffle(all_items)

    if num_samples and num_samples < len(all_items):
        all_items = all_items[:num_samples]

    # Format questions
    formatted_prompts = []
    for item in tqdm(all_items, desc="Formatting MATH"):
        # MATH dataset format: problem, solution, level, type
        problem = item['problem']

        # Add subject/type information if available
        subject = item.get('type', 'Math')
        level = item.get('level', '')

        # Format the prompt
        if level:
            formatted_question = f"[{subject} - Level {level}]\n{problem}"
        else:
            formatted_question = f"[{subject}]\n{problem}"

        # Extract final answer from solution (usually in \boxed{})
        solution = item.get('solution', '')
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', solution)
        if boxed_match:
            final_answer = boxed_match.group(1)
        else:
            # Fallback: try to get last number
            numbers = re.findall(r'[-+]?\d*\.?\d+', solution)
            final_answer = numbers[-1] if numbers else solution[-50:] if solution else ''

        formatted_prompts.append({
            'question': formatted_question,
            'answer': final_answer,
            'full_solution': solution,  # Keep full solution for reference
            'metadata': {
                'dataset': 'math',
                'subject': subject,
                'level': level
            }
        })

    print(f"✓ Prepared {len(formatted_prompts)} prompts from MATH")
    return formatted_prompts


# Dataset loader registry
DATASET_LOADERS = {
    'gsm8k': load_gsm8k,
    'winogrande': load_winogrande,
    'commonsense_qa': load_commonsense_qa,
    'arc_challenge': load_arc_challenge,
    'boolq': load_boolq,
    'hotpotqa': load_hotpotqa,
    'hendrycks_math': load_hendrycks_math
}


def load_dataset_by_name(dataset_name, split, num_samples, random_seed=42):
    """
    Load dataset by name

    Args:
        dataset_name: Name of the dataset
        split: 'train' or 'test'/'validation'
        num_samples: Number of samples to load
        random_seed: Random seed for reproducibility

    Returns:
        List of formatted prompts
    """
    if dataset_name not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_LOADERS.keys())}")

    loader = DATASET_LOADERS[dataset_name]
    return loader(split, num_samples, random_seed)
