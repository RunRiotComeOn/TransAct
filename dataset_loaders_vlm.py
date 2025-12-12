"""
VLM Dataset loading and formatting functions for vision-language tasks
Supports: ScienceQA, MathVista, Geometry3K, OK-VQA
"""

from datasets import load_dataset
from tqdm import tqdm
from typing import List, Dict
import re
from PIL import Image
import requests
from io import BytesIO
import json
import urllib.request
import zipfile
from pathlib import Path


def load_geometry3k(split: str = "test", num_samples: int = None, random_seed: int = 42) -> List[Dict]:
    """
    Load Geometry3K dataset

    Args:
        split: Dataset split (only 'test' available)
        num_samples: Number of samples to load (None for all)
        random_seed: Random seed for sampling

    Returns:
        List of samples with image, question, answer, and metadata
    """
    print(f"\nLoading Geometry3K ({split} split)...")
    dataset = load_dataset("hiyouga/geometry3k", split="test")

    # Shuffle if random sampling
    if num_samples is not None:
        dataset = dataset.shuffle(seed=random_seed)
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    data = []
    for idx, item in enumerate(tqdm(dataset, desc="Processing Geometry3K")):
        # Skip samples without images
        if not item.get('images') or len(item['images']) == 0:
            print(f"Warning: sample {idx} has no image, skipping.")
            continue

        # Extract answer from the text (usually a number or simple expression)
        answer_text = str(item['answer']).strip()

        data.append({
            'image': item['images'][0],           # PIL Image
            'question': item['problem'],          # Already includes <image> placeholder
            'answer': answer_text,
            'metadata': {
                'problem_id': f'geo3k_{idx}',
                'dataset': 'geometry3k'
            }
        })

    print(f"✓ Loaded {len(data)} valid Geometry3K samples")
    return data


def load_scienceqa(split: str = "train", num_samples: int = None, random_seed: int = 42) -> List[Dict]:
    """
    Load ScienceQA dataset

    Args:
        split: Dataset split ('train', 'validation', or 'test')
        num_samples: Number of samples to load (None for all)
        random_seed: Random seed for sampling

    Returns:
        List of samples with image, question, answer, and metadata
    """
    print(f"\nLoading ScienceQA ({split} split)...")
    dataset = load_dataset("derek-thomas/ScienceQA", split=split)

    # Shuffle and sample if needed
    if num_samples is not None:
        dataset = dataset.shuffle(seed=random_seed)
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    data = []
    for idx, item in enumerate(tqdm(dataset, desc="Processing ScienceQA")):
        # Filter only items with images
        if item.get('image') is None:
            continue

        question = item['question']
        choices = item.get('choices', [])
        answer_idx = item.get('answer', 0)

        # Format question with choices (A, B, C, ...)
        if choices:
            question_with_choices = f"{question}\n"
            for i, choice in enumerate(choices):
                question_with_choices += f"({chr(ord('A') + i)}) {choice}\n"
            question_with_choices += "Answer:"

            # Get answer as letter and text
            answer_letter = chr(ord('A') + answer_idx) if answer_idx < len(choices) else str(answer_idx)
            answer_text = choices[answer_idx] if answer_idx < len(choices) else str(answer_idx)
            answer = f"{answer_letter}"  # Use letter for easier evaluation
        else:
            question_with_choices = question
            answer = str(answer_idx)
            answer_text = str(answer_idx)

        data.append({
            'image': item['image'],
            'question': question_with_choices,
            'answer': answer,
            'answer_text': answer_text,
            'metadata': {
                'subject': item.get('subject', ''),
                'topic': item.get('topic', ''),
                'grade': item.get('grade', ''),
                'dataset': 'scienceqa'
            }
        })

    print(f"✓ Loaded {len(data)} ScienceQA samples with images")
    return data


def load_mathvista(split: str = "testmini", num_samples: int = None, random_seed: int = 42) -> List[Dict]:
    """
    Load MathVista dataset

    Args:
        split: Dataset split ('testmini' or 'test')
        num_samples: Number of samples to load (None for all)
        random_seed: Random seed for sampling

    Returns:
        List of samples with image, question, answer, and metadata
    """
    print(f"\nLoading MathVista ({split} split)...")
    dataset = load_dataset("AI4Math/MathVista", split=split)

    # Shuffle and sample if needed
    if num_samples is not None:
        dataset = dataset.shuffle(seed=random_seed)
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    data = []
    for idx, item in enumerate(tqdm(dataset, desc="Processing MathVista")):
        # Get image - handle different possible formats
        # IMPORTANT: Check decoded_image FIRST (PIL Image), not image (string path)
        image = None
        if item.get('decoded_image') is not None:
            image = item['decoded_image']
        elif item.get('image') is not None and not isinstance(item['image'], str):
            # Only use 'image' if it's not a string path
            image = item['image']
        elif item.get('image_url') is not None:
            # Download image from URL if needed
            try:
                import requests
                from io import BytesIO
                from PIL import Image as PILImage
                response = requests.get(item['image_url'], timeout=10)
                image = PILImage.open(BytesIO(response.content)).convert('RGB')
            except Exception as e:
                print(f"Warning: Failed to download image for sample {idx}: {e}")
                continue

        if image is None:
            continue

        question = item.get('question', '')
        answer = item.get('answer', '')

        # Get question type and metadata
        question_type = item.get('question_type', 'unknown')
        metadata = item.get('metadata', {})

        # Handle different answer formats
        if isinstance(answer, (int, float)):
            answer = str(answer)

        # Format multiple choice questions
        if question_type == 'multi_choice' and item.get('choices'):
            choices = item['choices']
            formatted_question = f"{question}\n"
            for i, choice in enumerate(choices):
                formatted_question += f"({chr(ord('A') + i)}) {choice}\n"
            formatted_question += "Answer:"
            question = formatted_question

        data.append({
            'image': image,
            'question': question,
            'answer': str(answer),
            'metadata': {
                'question_type': question_type,
                'answer_type': item.get('answer_type', 'text'),
                'choices': item.get('choices', []),
                'task': item.get('task', ''),
                'subject': item.get('subject', ''),
                'pid': item.get('pid', f'mathvista_{idx}'),
                'dataset': 'mathvista',
                **metadata
            }
        })

    print(f"✓ Loaded {len(data)} MathVista samples with images")
    return data


def download_and_extract_okvqa(url: str, cache_dir: Path) -> Path:
    """Download and extract a zip file, return the extracted directory path."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Get filename from URL
    filename = url.split('/')[-1]
    filepath = cache_dir / filename
    extracted_name = filename.replace('.zip', '')
    extracted_path = cache_dir / extracted_name

    # Check if already extracted
    if extracted_path.exists():
        print(f"  Using cached: {extracted_name}")
        return extracted_path

    # Download if not exists
    if not filepath.exists():
        print(f"  Downloading: {filename}")
        urllib.request.urlretrieve(url, filepath)

    # Extract
    print(f"  Extracting: {filename}")
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall(cache_dir)

    # Handle nested directory structure
    if extracted_path.is_file():
        return extracted_path

    # For train2014/val2014 images, the directory structure is different
    if 'train2014' in filename or 'val2014' in filename:
        # Images are extracted to a directory like train2014/
        img_dir = cache_dir / extracted_name
        if img_dir.exists():
            return img_dir

    return extracted_path


def load_chartqa(split: str = "test", num_samples: int = None, random_seed: int = 42) -> List[Dict]:
    """
    Load ChartQA dataset for chart understanding and reasoning

    ChartQA requires understanding charts (bar, line, pie charts) and
    performing reasoning/calculations based on visual data.

    Args:
        split: Dataset split ('train' or 'test', default: 'test')
        num_samples: Number of samples to load (None for all)
        random_seed: Random seed for sampling

    Returns:
        List of samples with image, question, answer, and metadata
    """
    print(f"\nLoading ChartQA ({split} split)...")

    try:
        # Load from HuggingFace
        # ChartQA has 'train' and 'test' splits
        dataset = load_dataset('ahmed-masry/ChartQA', split=split)
        print(f"✓ Loaded ChartQA {split} split: {len(dataset)} samples")

    except Exception as e:
        print(f"Error loading ChartQA: {e}")
        print("Trying alternative loading method...")
        # Fallback
        dataset = load_dataset('ahmed-masry/ChartQA', split=split, cache_dir=None)

    # Convert to our format
    samples = []

    for idx, item in enumerate(tqdm(dataset, desc="Processing ChartQA")):
        # ChartQA format:
        # - image: bytes (need to convert to PIL Image)
        # - query: str (question)
        # - label: str (answer, can be number or text)
        # - type: str ('human' or 'augmented')

        # Convert image bytes to PIL Image
        import io
        from PIL import Image as PILImage
        image_bytes = item['image']
        image = PILImage.open(io.BytesIO(image_bytes)).convert('RGB')

        question = item['query']  # Question field
        answer = str(item['label']).strip()  # Answer is in 'label' field

        sample = {
            'image': image,
            'question': question,
            'answer': answer,
            'metadata': {
                'dataset': 'chartqa',
                'task': 'chart_understanding',
                'index': idx,
                'source': item.get('type', 'unknown'),  # human or augmented
            }
        }

        samples.append(sample)

    # Sample if requested
    if num_samples is not None and num_samples < len(samples):
        import random
        random.seed(random_seed)
        samples = random.sample(samples, num_samples)

    print(f"✓ Prepared {len(samples)} ChartQA samples")

    return samples


def load_okvqa(split: str = "train", num_samples: int = None, random_seed: int = 42) -> List[Dict]:
    """
    Load OK-VQA dataset from official sources.

    OK-VQA (Outside Knowledge VQA) requires commonsense and world knowledge
    to answer questions about images.

    Args:
        split: Dataset split ('train' or 'val')
        num_samples: Number of samples to load (None for all)
        random_seed: Random seed for sampling

    Returns:
        List of samples with image, question, answer, and metadata
    """
    print(f"\nLoading OK-VQA ({split} split)...")

    # Setup cache directory
    cache_dir = Path.home() / ".cache" / "okvqa"

    # URLs for train split
    if split == "train":
        urls = {
            "annotations": "https://okvqa.allenai.org/static/data/mscoco_train2014_annotations.json.zip",
            "questions": "https://okvqa.allenai.org/static/data/OpenEnded_mscoco_train2014_questions.json.zip",
            "images": "http://images.cocodataset.org/zips/train2014.zip",
        }
        img_prefix = "train2014"
    else:  # val split
        urls = {
            "annotations": "https://okvqa.allenai.org/static/data/mscoco_val2014_annotations.json.zip",
            "questions": "https://okvqa.allenai.org/static/data/OpenEnded_mscoco_val2014_questions.json.zip",
            "images": "http://images.cocodataset.org/zips/val2014.zip",
        }
        img_prefix = "val2014"

    # Download and extract files
    print("Downloading OK-VQA data (this may take a while on first run)...")
    annotations_path = download_and_extract_okvqa(urls["annotations"], cache_dir)
    questions_path = download_and_extract_okvqa(urls["questions"], cache_dir)
    images_dir = download_and_extract_okvqa(urls["images"], cache_dir)

    # Load JSON files
    print("Loading annotations and questions...")
    with open(annotations_path, 'r') as f:
        annotations_data = json.load(f)

    with open(questions_path, 'r') as f:
        questions_data = json.load(f)

    # Create lookup for annotations by question_id
    annotations_dict = {}
    for ann in annotations_data['annotations']:
        annotations_dict[ann['question_id']] = ann

    # Combine questions with annotations and load images
    data = []
    print("Processing OK-VQA samples...")

    for question_item in tqdm(questions_data['questions'], desc="Processing OK-VQA"):
        question_id = question_item['question_id']

        if question_id not in annotations_dict:
            continue

        annotation = annotations_dict[question_id]
        image_id = question_item['image_id']

        # Construct image path
        # Format: COCO_train2014_000000xxxxxx.jpg or COCO_val2014_000000xxxxxx.jpg
        image_filename = f"COCO_{img_prefix}_{image_id:012d}.jpg"
        image_path = images_dir / image_filename

        # Skip if image doesn't exist
        if not image_path.exists():
            continue

        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue

        # Extract most common answer (OK-VQA has 10 answers per question)
        answers_list = [ans['answer'] for ans in annotation['answers']]
        # Count answer frequencies
        from collections import Counter
        answer_counts = Counter(answers_list)
        most_common_answer = answer_counts.most_common(1)[0][0]

        data.append({
            'image': image,
            'question': question_item['question'],
            'answer': most_common_answer,  # Use most common answer
            'metadata': {
                'question_id': question_id,
                'image_id': image_id,
                'question_type': annotation.get('question_type', ''),
                'answer_type': annotation.get('answer_type', ''),
                'all_answers': answers_list,  # Keep all 10 answers for evaluation
                'dataset': 'okvqa',
                'task': 'commonsense_vqa'  # OK-VQA is a commonsense reasoning task
            }
        })

    # Shuffle and sample if needed
    if num_samples is not None and num_samples < len(data):
        import random
        random.seed(random_seed)
        data = random.sample(data, num_samples)

    print(f"✓ Loaded {len(data)} OK-VQA samples with images")
    return data


def load_gqa(split: str = "test", num_samples: int = None, random_seed: int = 42) -> List[Dict]:
    """
    Load GQA dataset for visual question answering

    GQA is a large-scale visual reasoning dataset with compositional questions
    requiring multi-step reasoning over real images.

    Args:
        split: Dataset split ('train', 'val', 'test', 'testdev', default: 'test')
        num_samples: Number of samples to load (None for all)
        random_seed: Random seed for sampling

    Returns:
        List of samples with image, question, answer, and metadata
    """
    print(f"\nLoading GQA ({split} split)...")

    # GQA dataset requires a config name. Use 'balanced_images' configs which have cleaner data
    # Map split names to (config_name, hf_split) tuples
    split_config_map = {
        'train': ('train_balanced_images', 'train'),
        'val': ('val_balanced_images', 'val'),
        'validation': ('val_balanced_images', 'val'),
        'test': ('testdev_balanced_images', 'testdev'),
        'testdev': ('testdev_balanced_images', 'testdev')
    }

    config_name, hf_split = split_config_map.get(split, ('testdev_balanced_images', 'testdev'))

    try:
        # Load from HuggingFace with config
        dataset = load_dataset('lmms-lab/GQA', config_name, split=hf_split)
        print(f"✓ Loaded GQA {config_name} ({hf_split}): {len(dataset)} samples")
    except Exception as e:
        print(f"Error loading GQA with config {config_name}: {e}")
        # Fallback to testdev_balanced_images
        dataset = load_dataset('lmms-lab/GQA', 'testdev_balanced_images', split='testdev')
        print(f"✓ Loaded GQA testdev_balanced_images (fallback): {len(dataset)} samples")

    # Convert to our format
    samples = []

    for idx, item in enumerate(tqdm(dataset, desc="Processing GQA")):
        # GQA format varies, adapt based on what's available
        image = item['image']
        if isinstance(image, bytes):
            import io
            from PIL import Image as PILImage
            image = PILImage.open(io.BytesIO(image)).convert('RGB')

        question = item.get('question', item.get('sent', ''))
        answer = str(item.get('answer', item.get('label', ''))).strip()

        sample = {
            'image': image,
            'question': question,
            'answer': answer,
            'metadata': {
                'dataset': 'gqa',
                'task': 'visual_reasoning',
                'index': idx,
                'question_id': item.get('question_id', item.get('questionId', idx))
            }
        }

        samples.append(sample)

    # Sample if requested
    if num_samples is not None and num_samples < len(samples):
        import random
        random.seed(random_seed)
        samples = random.sample(samples, num_samples)

    print(f"✓ Prepared {len(samples)} GQA samples")
    return samples


def load_vqav2(split: str = "validation", num_samples: int = None, random_seed: int = 42) -> List[Dict]:
    """
    Load VQA v2 dataset for visual question answering

    VQA v2 contains open-ended questions about images requiring understanding
    of vision, language and commonsense knowledge.

    Args:
        split: Dataset split ('train', 'validation', 'testdev', 'test', default: 'validation')
        num_samples: Number of samples to load (None for all)
        random_seed: Random seed for sampling

    Returns:
        List of samples with image, question, answer, and metadata
    """
    print(f"\nLoading VQA v2 ({split} split)...")

    # Map split names
    split_map = {
        'train': 'train',
        'val': 'validation',
        'validation': 'validation',
        'test': 'test'
    }
    hf_split = split_map.get(split, 'validation')

    # Load from lmms-lab mirror
    dataset = load_dataset('lmms-lab/VQAv2', split=hf_split)
    print(f"✓ Loaded VQA v2 {hf_split} split: {len(dataset)} samples")

    # Convert to our format
    samples = []

    for idx, item in enumerate(tqdm(dataset, desc="Processing VQA v2")):
        image = item['image']
        if isinstance(image, bytes):
            import io
            from PIL import Image as PILImage
            image = PILImage.open(io.BytesIO(image)).convert('RGB')

        question = item['question']

        # VQA v2 has multiple answers - use the most common one
        if 'answers' in item and item['answers'] is not None:
            answers_list = item['answers']
            if isinstance(answers_list, list) and len(answers_list) > 0:
                # Get the most common answer
                answer = answers_list[0].get('answer', '')
            else:
                answer = str(item.get('multiple_choice_answer', '')).strip()
        else:
            answer = str(item.get('multiple_choice_answer', '')).strip()

        # Collect all answers for flexible evaluation (like OK-VQA)
        all_answers = []
        if 'answers' in item and item['answers'] is not None:
            all_answers = [ans.get('answer', '') for ans in item['answers'] if 'answer' in ans]

        sample = {
            'image': image,
            'question': question,
            'answer': answer,
            'metadata': {
                'dataset': 'vqav2',
                'task': 'visual_qa',
                'index': idx,
                'question_id': item.get('question_id', idx),
                'question_type': item.get('question_type', 'unknown'),
                'answer_type': item.get('answer_type', 'unknown'),
                'all_answers': all_answers  # For flexible matching
            }
        }

        samples.append(sample)

    # Sample if requested
    if num_samples is not None and num_samples < len(samples):
        import random
        random.seed(random_seed)
        samples = random.sample(samples, num_samples)

    print(f"✓ Prepared {len(samples)} VQA v2 samples")
    return samples


# VLM Dataset loader registry
VLM_DATASET_LOADERS = {
    'geometry3k': load_geometry3k,
    'scienceqa': load_scienceqa,
    'mathvista': load_mathvista,
    'okvqa': load_okvqa,
    'chartqa': load_chartqa,
    'gqa': load_gqa,
    'vqav2': load_vqav2,
}


def load_vlm_dataset(dataset_name: str, split: str = "test", num_samples: int = None,
                     random_seed: int = 42) -> List[Dict]:
    """
    Load VLM dataset by name

    Args:
        dataset_name: Name of the dataset ('geometry3k', 'scienceqa', 'mathvista', 'okvqa', 'chartqa', 'gqa', 'vqav2')
        split: Dataset split
        num_samples: Number of samples to load (None for all)
        random_seed: Random seed for reproducibility

    Returns:
        List of formatted samples
    """
    if dataset_name not in VLM_DATASET_LOADERS:
        raise ValueError(
            f"Unknown VLM dataset: {dataset_name}. "
            f"Available: {list(VLM_DATASET_LOADERS.keys())}"
        )

    loader = VLM_DATASET_LOADERS[dataset_name]
    return loader(split=split, num_samples=num_samples, random_seed=random_seed)


def format_vlm_prompt(sample: Dict, include_cot: bool = True) -> str:
    """
    Format a VLM sample into a prompt

    Args:
        sample: Sample dictionary with 'question' field
        include_cot: Whether to include chain-of-thought prompting

    Returns:
        Formatted prompt string
    """
    question = sample['question']

    if include_cot:
        # Add CoT prompting for better reasoning
        if sample['metadata'].get('dataset') == 'mathvista':
            prompt = f"{question}\nPlease solve this step by step and provide your final answer."
        elif sample['metadata'].get('dataset') == 'geometry3k':
            prompt = f"{question}\nPlease solve this geometry problem step by step."
        elif sample['metadata'].get('dataset') == 'scienceqa':
            prompt = question  # Already formatted with choices
        elif sample['metadata'].get('dataset') == 'okvqa':
            prompt = f"{question}\nPlease think about what common knowledge is needed and provide your answer."
        elif sample['metadata'].get('dataset') == 'chartqa':
            prompt = f"{question}\nPlease analyze the chart carefully and provide your answer based on the data shown."
        elif sample['metadata'].get('dataset') == 'gqa':
            prompt = f"{question}\nPlease reason about the image and provide your answer."
        elif sample['metadata'].get('dataset') == 'vqav2':
            prompt = f"{question}\nPlease answer based on what you see in the image."
        else:
            prompt = f"{question}\nPlease answer step by step."
    else:
        prompt = question

    return prompt


def extract_answer_from_response(response: str, dataset: str, question_type: str = None,
                                  answer_type: str = None, choices: list = None) -> str:
    """
    Extract answer from model response

    Args:
        response: Model generated response
        dataset: Dataset name for dataset-specific extraction
        question_type: Type of question (for MathVista: 'multi_choice' or 'free_form')
        answer_type: Type of answer (for MathVista: 'integer', 'float', 'text')
        choices: List of choices for multi-choice questions

    Returns:
        Extracted answer string
    """
    response = response.strip()

    if not response:
        return ""

    # === MathVista-specific extraction (following official implementation) ===
    if dataset == 'mathvista':
        # For multi-choice: if response is directly one of the choices, return it
        if question_type == 'multi_choice' and choices:
            if response in choices:
                return response
            # Also check if response matches choice letter (A, B, C, D, E)
            for i, choice in enumerate(choices):
                letter = chr(ord('A') + i)
                if response.upper() == letter:
                    return letter

        # For integer answers: try direct parsing
        if answer_type == 'integer':
            try:
                extraction = int(float(response))  # handle "3.0" -> 3
                return str(extraction)
            except:
                pass

        # For float answers: try direct parsing
        if answer_type == 'float':
            try:
                extraction = float(response)
                return str(extraction)
            except:
                pass

        # Quick extraction patterns (from official MathVista code)
        # Pattern: The answer is "text".
        try:
            result = re.search(r'[Tt]he answer is ["\']?([^"\'\.]+)["\']?\.?', response)
            if result:
                extraction = result.group(1).strip()
                # Try to convert to number if answer_type expects it
                if answer_type == 'integer':
                    try:
                        return str(int(float(extraction)))
                    except:
                        pass
                elif answer_type == 'float':
                    try:
                        return str(float(extraction))
                    except:
                        pass
                return extraction
        except:
            pass

        # Pattern: answer is X or final answer is X
        patterns = [
            r'(?:final\s+)?answer\s*(?:is|:)\s*["\']?([^"\'\n,]+)["\']?',
            r'####\s*(.+?)(?:\n|$)',
            r'=\s*([^\n=]+?)\s*$',
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                extraction = match.group(1).strip()
                # Clean up and convert if needed
                if answer_type == 'integer':
                    try:
                        nums = re.findall(r'[-+]?\d+', extraction)
                        if nums:
                            return nums[0]
                    except:
                        pass
                elif answer_type == 'float':
                    try:
                        nums = re.findall(r'[-+]?\d*\.?\d+', extraction)
                        if nums:
                            return nums[0]
                    except:
                        pass
                # For multi-choice, extract letter
                if question_type == 'multi_choice':
                    letter_match = re.search(r'\b([A-E])\b', extraction.upper())
                    if letter_match:
                        return letter_match.group(1)
                return extraction

        # Fallback: extract last number for numerical answers
        if answer_type in ['integer', 'float']:
            numbers = re.findall(r'[-+]?\d*\.?\d+', response)
            if numbers:
                if answer_type == 'integer':
                    try:
                        return str(int(float(numbers[-1])))
                    except:
                        return numbers[-1]
                return numbers[-1]

        # For multi-choice fallback: find any letter A-E
        if question_type == 'multi_choice':
            # Look for pattern like "A)", "(A)", "Answer: A", etc.
            mc_patterns = [
                r'(?:answer|choice)(?:\s+is)?(?:\s*:)?\s*\(?([A-E])\)?',
                r'\(([A-E])\)',
                r'\b([A-E])\)',
            ]
            for pattern in mc_patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    return match.group(1).upper()

        # Last resort: return cleaned response
        return response.split('\n')[-1].strip()

    # === ScienceQA extraction ===
    if dataset == 'scienceqa':
        # Look for pattern like "A)", "(A)", "Answer: A", etc.
        patterns = [
            r'(?:answer|Answer)(?:\s+is)?(?:\s*:)?\s*\(?([A-E])\)?',
            r'\(([A-E])\)',
            r'\b([A-E])\)',
            r'\b([A-E])\b',
        ]
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1)

    # === Numerical datasets (Geometry3K, ChartQA) ===
    if dataset in ['geometry3k', 'chartqa']:
        # Look for final answer patterns
        patterns = [
            r'(?:final answer|answer)(?:\s+is)?(?:\s*:)?\s*([-+]?\d*\.?\d+)',
            r'####\s*([-+]?\d*\.?\d+)',
            r'=\s*([-+]?\d*\.?\d+)\s*$',
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1)

        # Fallback: get last number in response
        numbers = re.findall(r'[-+]?\d*\.?\d+', response)
        if numbers:
            return numbers[-1]

    # Return full response if no pattern matched
    return response


def evaluate_vlm_answer(predicted: str, expected: str, dataset: str,
                        all_answers: list = None, question_type: str = None,
                        answer_type: str = None) -> bool:
    """
    Evaluate if predicted answer matches expected answer

    Args:
        predicted: Predicted answer
        expected: Expected/ground truth answer
        dataset: Dataset name for dataset-specific evaluation
        all_answers: List of all valid answers (for OK-VQA which has multiple answers)
        question_type: Type of question (for MathVista)
        answer_type: Type of answer (for MathVista: 'integer', 'float', 'text')

    Returns:
        True if correct, False otherwise
    """
    predicted = predicted.strip()
    expected = expected.strip()

    # Handle empty expected (invalid ground truth)
    if not expected:
        return False

    # === MathVista-specific evaluation (following official implementation) ===
    if dataset == 'mathvista':
        # Normalize for comparison
        pred_upper = predicted.upper()
        exp_upper = expected.upper()

        # For multi-choice questions
        if question_type == 'multi_choice':
            # Extract letters and compare
            pred_letter = re.search(r'([A-E])', pred_upper)
            exp_letter = re.search(r'([A-E])', exp_upper)
            if pred_letter and exp_letter:
                return pred_letter.group(1) == exp_letter.group(1)
            # Also check if predicted matches expected text directly
            return pred_upper == exp_upper

        # For integer answers
        if answer_type == 'integer':
            try:
                pred_int = int(float(predicted))
                exp_int = int(float(expected))
                return pred_int == exp_int
            except (ValueError, TypeError):
                # Fallback to string comparison
                return pred_upper == exp_upper

        # For float answers
        if answer_type == 'float':
            try:
                pred_float = float(predicted)
                exp_float = float(expected)
                # Use relative tolerance for floats
                if exp_float == 0:
                    return abs(pred_float) < 1e-6
                return abs(pred_float - exp_float) / abs(exp_float) < 0.01
            except (ValueError, TypeError):
                # Fallback to string comparison
                return pred_upper == exp_upper

        # For text answers (default)
        # Try numerical comparison first
        try:
            pred_nums = re.findall(r'[-+]?\d*\.?\d+', predicted)
            exp_nums = re.findall(r'[-+]?\d*\.?\d+', expected)
            if pred_nums and exp_nums:
                pred_num = float(pred_nums[0])
                exp_num = float(exp_nums[0])
                if exp_num == 0:
                    return abs(pred_num) < 1e-6
                return abs(pred_num - exp_num) / abs(exp_num) < 0.01
        except:
            pass

        # String comparison (case-insensitive)
        return pred_upper == exp_upper

    # For OK-VQA and VQA v2: check if predicted matches any of the multiple answers
    if dataset in ['okvqa', 'vqav2'] and all_answers:
        predicted_upper = predicted.upper()
        # Normalize all answers
        normalized_answers = [str(ans).strip().upper() for ans in all_answers]
        # Check exact match with any answer
        if predicted_upper in normalized_answers:
            return True
        # Check if predicted is contained in any answer or vice versa
        for ans in normalized_answers:
            if predicted_upper in ans or ans in predicted_upper:
                return True
        # Check word-level overlap (flexible matching for OK-VQA)
        pred_words = set(predicted_upper.split())
        for ans in normalized_answers:
            ans_words = set(ans.split())
            # If there's significant overlap, consider it correct
            if len(pred_words & ans_words) > 0 and (
                len(pred_words & ans_words) / len(pred_words) > 0.5 or
                len(pred_words & ans_words) / len(ans_words) > 0.5
            ):
                return True
        return False

    # Direct string match
    predicted_upper = predicted.upper()
    expected_upper = expected.upper()
    if predicted_upper == expected_upper:
        return True

    # For multiple choice (ScienceQA), check letter only
    if dataset == 'scienceqa':
        pred_letter = re.search(r'([A-E])', predicted_upper)
        exp_letter = re.search(r'([A-E])', expected_upper)
        if pred_letter and exp_letter:
            return pred_letter.group(1) == exp_letter.group(1)

    # For numerical answers (Geometry3K, ChartQA), check numerical equivalence
    if dataset in ['geometry3k', 'chartqa']:
        try:
            pred_num = float(re.findall(r'[-+]?\d*\.?\d+', predicted)[0])
            exp_num = float(re.findall(r'[-+]?\d*\.?\d+', expected)[0])
            # Check if within 1% tolerance
            if exp_num == 0:
                return abs(pred_num) < 1e-6
            return abs(pred_num - exp_num) / abs(exp_num) < 0.01
        except (ValueError, IndexError):
            pass

    # Substring match as fallback
    return expected_upper in predicted_upper or predicted_upper in expected_upper
