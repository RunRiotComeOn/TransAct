# TransAct Quick Start Guide

## 🚀 Quick Demo (5 minutes)

Run the complete pipeline with default settings:

```bash
cd /nas03/yixuh/TransAct
bash demo.sh
```

This will extract GSM8K steering vectors and apply them to Geometry3K.

## 📦 File Overview

| File | Purpose | Key Function |
|------|---------|--------------|
| **extract_steering_vectors.py** | Phase 1: Extract steering vectors from LLMs | Main extraction pipeline |
| **main-vlm.py** | Phase 2: Apply steering to VLMs | Entropy-based dynamic layer selection |
| **finetune_vlm_lora.py** | Baseline comparison | LoRA fine-tuning |
| **visualize_entropy_curve.py** | Analysis tool | Visualize entropy across layers |
| **visualize_vocabulary_analysis.py** | Analysis tool | Analyze reasoning vocabulary |
| **config.py** | Configuration | Model and dataset settings |
| **model_utils.py** | Utility | Model loading and generation |
| **dataset_loaders.py** | Utility | Text dataset loaders |
| **dataset_loaders_vlm.py** | Utility | VLM dataset loaders |
| **steering_vectors.py** | Utility | Steering vector extraction |
| **optimize_entropy_layer.py** | Utility | Entropy-based layer selection |
| **demo.sh** | Demo script | Complete pipeline |

## 🎯 Common Use Cases

### 1. Extract Different Reasoning Types

```bash
# Mathematical reasoning (GSM8K)
python extract_steering_vectors.py --ability mathematical_reasoning --num_samples 50

# Science reasoning (ARC-Challenge)
python extract_steering_vectors.py --ability science_reasoning --num_samples 50

# Logical reasoning (Winogrande)
python extract_steering_vectors.py --ability logical_reasoning --num_samples 50
```

### 2. Apply to Different VLM Datasets

```bash
# Geometry3K
python main-vlm.py --steering_type gsm8k_50_qwen2_5_7b --target_dataset geometry3k

# ScienceQA
python main-vlm.py --steering_type arc_challenge_50_qwen2_5_7b --target_dataset scienceqa

# MathVista
python main-vlm.py --steering_type gsm8k_50_qwen2_5_7b --target_dataset mathvista

# ChartQA
python main-vlm.py --steering_type gsm8k_50_qwen2_5_7b --target_dataset chartqa

# OK-VQA
python main-vlm.py --steering_type commonsense_qa_50_qwen2_5_7b --target_dataset okvqa
```

### 3. Tune Steering Strength

```bash
# Try different alpha values
python main-vlm.py --steering_type gsm8k_50_qwen2_5_7b --target_dataset geometry3k --alpha 3.0
python main-vlm.py --steering_type gsm8k_50_qwen2_5_7b --target_dataset geometry3k --alpha 5.0
python main-vlm.py --steering_type gsm8k_50_qwen2_5_7b --target_dataset geometry3k --alpha 7.0
```

### 4. Compare with LoRA Fine-Tuning

```bash
# Train LoRA baseline
python finetune_vlm_lora.py \
    --steering_type gsm8k_50_qwen2_5_7b \
    --target_dataset geometry3k \
    --num_train_samples 50 \
    --num_test_samples 100 \
    --rank 8 \
    --num_epochs 3

# Compare results in ./lora_results/ vs ./vlm_steering_results/
```

### 5. Visualize Results

```bash
# Entropy analysis
python visualize_entropy_curve.py --dataset geometry3k --num_samples 20

# Vocabulary analysis (after running main-vlm.py)
python visualize_vocabulary_analysis.py ./vocabulary_analysis_results/
```

## 📊 Expected Runtime

| Task | Samples | Time (GPU) | Time (CPU) |
|------|---------|-----------|-----------|
| Extract steering vectors | 50 | ~10 min | ~2 hours |
| Apply to VLM (validation) | 100 | ~15 min | ~3 hours |
| Apply to VLM (test) | 200 | ~30 min | ~6 hours |
| LoRA fine-tuning | 50 | ~1 hour | Not recommended |
| Visualize entropy | 20 | ~5 min | ~30 min |

**Note:** Times are approximate and depend on your hardware. Using GPU is highly recommended.

## 🔧 Troubleshooting

### Out of Memory

If you encounter OOM errors:

```bash
# Reduce batch size (in code, set batch_size=1)
# Or use smaller model
# Or reduce max_new_tokens
python main-vlm.py --steering_type gsm8k_50_qwen2_5_7b --target_dataset geometry3k --num_val_samples 50 --num_test_samples 100
```

### Slow Generation

Enable mixed precision:

```python
# In config.py, set:
torch_dtype = "bfloat16"  # or "float16"
```

### Dataset Download Issues

Some datasets (e.g., OK-VQA) require manual download:

```bash
# The script will download automatically on first run
# But if it fails, check your internet connection
# Or manually download from the official sources
```

## 📈 Performance Tips

1. **Use GPU**: CPU inference is 10-20x slower
2. **Mixed Precision**: Use `bfloat16` for better speed/accuracy tradeoff
3. **Batch Processing**: For large-scale evaluation, increase batch size
4. **Caching**: Steering vectors are reusable across different target datasets

## 🎓 Learning Path

1. **Start with demo.sh** - Understand the complete pipeline
2. **Try different steering types** - See how reasoning transfer varies
3. **Compare with LoRA** - Understand training-free advantages
4. **Visualize results** - Gain insights into the mechanism
5. **Customize for your use case** - Adapt to your specific needs

## 📝 Citation

```bibtex
@article{transact2024,
  title={TransAct: Training-Free Transfer of Reasoning Abilities from LLMs to VLMs via Activation Steering},
  author={Huang, Yixu},
  journal={arXiv preprint},
  year={2024}
}
```

## 🤝 Support

- Issues: Open a GitHub issue
- Email: yixhuang@ucdavis.edu
- Full documentation: See README.md

---

**Happy experimenting! 🎯**
