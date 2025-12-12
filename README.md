# TransAct: Training-Free Transfer of Reasoning Abilities from LLMs to VLMs via Activation Steering

**TransAct** (Training-Free Transfer via Activation-Steered Reasoning) is a novel framework for transferring reasoning capabilities from text-based Large Language Models (LLMs) to Vision-Language Models (VLMs) through activation steering, without any training or fine-tuning.

## 🚀 Key Features

- **Training-Free**: No gradient computation or parameter updates required
- **Efficient**: Only one additional forward pass per input for entropy computation
- **Effective**: Achieves +8% on Geometry3K and +5% on ScienceQA
- **Better Generalization**: Outperforms LoRA fine-tuning on out-of-distribution samples
- **Dynamic Layer Selection**: Entropy-based mechanism to select optimal intervention layers per sample

## 📖 Overview

TransAct operates in two phases:

1. **Steering Vector Extraction** (Phase 1): Extract reasoning vectors from activation differences between a base LLM and a reasoning-enhanced LLM
2. **Entropy-Based Dynamic Injection** (Phase 2): Inject these vectors into VLM hidden states at inference time using entropy-guided layer selection

## 📦 Installation

###Dependencies

```bash
pip install torch transformers datasets pillow matplotlib seaborn numpy tqdm
pip install peft  # For LoRA fine-tuning comparison
pip install qwen-vl-utils  # For Qwen2.5-VL support
```

### Recommended Environment

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU acceleration)
- At least 24GB VRAM for 7B models

## 🗂️ Project Structure

```
TransAct/
├── extract_steering_vectors.py       # Phase 1: Extract steering vectors from LLMs
├── main-vlm.py                        # Phase 2: Apply steering to VLMs
├── finetune_vlm_lora.py              # Baseline: LoRA fine-tuning for comparison
├── visualize_entropy_curve.py        # Visualization: Entropy curves across layers
├── visualize_vocabulary_analysis.py  # Visualization: Reasoning vocabulary analysis
├── config.py                          # Configuration file
├── model_utils.py                     # Model loading and generation utilities
├── dataset_loaders.py                 # Text-based dataset loaders
├── dataset_loaders_vlm.py            # VLM dataset loaders
├── steering_vectors.py                # Steering vector extraction and analysis
├── optimize_entropy_layer.py         # Entropy-based dynamic layer selection
├── README.md                          # This file
└── demo.sh                            # Quick start demo script
```

## 🎯 Quick Start

### Demo: Complete Pipeline

Run the complete TransAct pipeline with a single command:

```bash
bash demo.sh
```

This will:
1. Extract steering vectors from GSM8K (mathematical reasoning)
2. Apply them to Geometry3K using entropy-based dynamic layer selection
3. Generate visualizations and results

### Step-by-Step Usage

#### Phase 1: Extract Steering Vectors

Extract reasoning steering vectors from LLM pairs:

```bash
# Extract mathematical reasoning vectors from GSM8K
python extract_steering_vectors.py \
    --ability mathematical_reasoning \
    --num_samples 50

# Extract science reasoning vectors from ARC-Challenge
python extract_steering_vectors.py \
    --ability science_reasoning \
    --num_samples 50

# Extract logical reasoning vectors from Winogrande
python extract_steering_vectors.py \
    --ability logical_reasoning \
    --num_samples 50
```

**Supported reasoning abilities:**
- `mathematical_reasoning` (GSM8K)
- `advanced_mathematical_reasoning` (MATH)
- `logical_reasoning` (Winogrande)
- `commonsense_reasoning` (CommonsenseQA)
- `science_reasoning` (ARC-Challenge)
- `reading_comprehension` (BoolQ)
- `multihop_reasoning` (HotpotQA)

The extracted steering vectors will be saved to `./steering_outputs/{dataset}_{num_samples}_{model}/`

#### Phase 2: Apply Steering to VLMs

Apply the extracted steering vectors to VLMs with entropy-based dynamic layer selection:

```bash
# Apply GSM8K steering vectors to Geometry3K
python main-vlm.py \
    --steering_type gsm8k_50_qwen2_5_7b \
    --target_dataset geometry3k \
    --alpha 5.0 \
    --num_val_samples 100 \
    --num_test_samples 200

# Apply ARC-Challenge steering vectors to ScienceQA
python main-vlm.py \
    --steering_type arc_challenge_50_qwen2_5_7b \
    --target_dataset scienceqa \
    --alpha 5.0
```

**Supported target VLM datasets:**
- `geometry3k`: Geometry problem solving with diagrams
- `scienceqa`: Multi-modal science questions
- `mathvista`: Mathematical reasoning with visual contexts
- `chartqa`: Chart understanding and reasoning
- `okvqa`: Open-ended VQA requiring external knowledge
- `gqa`: Visual reasoning
- `vqav2`: Visual question answering

## 📊 Visualization Tools

### Entropy Curve Visualization

Visualize how entropy changes across VLM layers to understand the model's decision-making process:

```bash
# Visualize entropy for a single sample
python visualize_entropy_curve.py \
    --dataset scienceqa \
    --sample_idx 0

# Visualize average entropy across multiple samples
python visualize_entropy_curve.py \
    --dataset geometry3k \
    --num_samples 20
```

### Vocabulary Analysis

Analyze reasoning vocabulary usage (causal words, logical words) before and after steering:

```bash
python visualize_vocabulary_analysis.py vocabulary_analysis_results/
```

## 🔬 Baseline Comparison: LoRA Fine-Tuning

Compare TransAct with LoRA fine-tuning:

```bash
python finetune_vlm_lora.py \
    --steering_type gsm8k_50_qwen2_5_7b \
    --target_dataset geometry3k \
    --num_train_samples 50 \
    --num_test_samples 100 \
    --rank 8 \
    --num_epochs 3
```

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Model Configuration
base_model = "Qwen/Qwen2.5-7B-Instruct"
reasoning_model = "Qwen/Qwen2.5-Math-7B-Instruct"

# Dataset Configuration
ability = "mathematical_reasoning"
num_extraction_samples = 50

# Generation Configuration
max_length = 2048
max_new_tokens = 2048

# Device Configuration
device = "cuda"
torch_dtype = "bfloat16"
```

## 📈 Expected Results

### Main Results (Test Accuracy)

| Steering Source | ChartQA | Geometry3K | MathVista | OK-VQA | ScienceQA |
|----------------|---------|------------|-----------|---------|-----------|
| **Baseline**   | 49.5%   | 24.0%      | 100.0%    | 76.0%   | 86.0%     |
| **ARC-Challenge** | 51.0% (+1.5) | **32.0% (+8.0)** | 100.0 (+0.0) | 81.0 (+4.0) | **95.3 (+4.7)** |
| **GSM8K**      | 47.0% (-4.0) | 26.0% (+2.0) | 100.0 (+0.0) | 80.0 (+4.0) | 88.4 (+2.3) |
| **Winogrande** | 49.0% (+1.5) | 25.0% (+1.0) | 100.0 (+0.0) | 81.0 (+1.0) | 93.0 (+2.3) |

### TransAct vs LoRA Fine-Tuning

| Method | Geometry3K (ID) | Geometry3K (OOD) | ScienceQA (ID) | ScienceQA (OOD) |
|--------|----------------|------------------|----------------|-----------------|
| **Baseline** | 24.0% | 22.5% | 86.0% | 78.3% |
| **LoRA Fine-tuning** | 32.5% | 24.0% | 91.2% | 80.1% |
| **TransAct (Ours)** | 31.0% | **28.5% (+4.5)** | 88.4% | **85.7% (+5.6)** |

TransAct achieves comparable in-distribution performance while **significantly outperforming** on out-of-distribution samples!

## 🔍 How It Works

### Phase 1: Steering Vector Extraction

1. Load base LLM (e.g., Qwen2.5-7B-Instruct) and reasoning LLM (e.g., Qwen2.5-Math-7B-Instruct)
2. For each training sample:
   - Generate responses from both models
   - Extract hidden states from all layers
   - Compute activation differences in the reasoning region
3. Aggregate differences across samples and normalize to unit vectors

### Phase 2: Entropy-Based Dynamic Injection

1. For each VLM input (image + text):
   - Perform forward pass to compute entropy at each layer
   - Calculate entropy changes between consecutive layers
   - Select top-2 layers with maximum entropy drops (turning points)
   - Randomly choose one from top-2 for diversity
2. Register forward hook on selected layer
3. Generate response with steering vector applied:
   ```
   h_steered = h_original + α * steering_vector
   ```
   where α = 5.0 is the fixed steering strength

### Why Entropy-Based Layer Selection?

The optimal intervention layer varies across samples:
- **Early layers** (0-9): High entropy, model is exploring
- **Turning point** (middle): Entropy drops sharply, model is converging
- **Late layers** (18-28): Low entropy, model has committed

By steering at the turning point, we influence the model's reasoning **before** it commits to a suboptimal path.

## 🔗 Related Work

- **Activation Addition** (Turner et al., 2024): Steering LLMs without optimization
- **Entropy Guided Decoding** (Das et al., 2024): Improving factuality via entropy
- **VISOR++** (Balakrishnan et al., 2024): Visual input-based steering for VLMs
- **Textual Steering Vectors** (2024): Improving visual understanding in MLLMs

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Acknowledgments

- Qwen Team for the Qwen2.5 model series
- All dataset creators: GSM8K, MATH, ARC, Winogrande, CommonsenseQA, HotpotQA, BoolQ, Geometry3K, ScienceQA, MathVista, ChartQA, OK-VQA, GQA, VQA v2

## 📧 Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact: yixhuang@ucdavis.edu

---

**Happy Steering! 🎯**
