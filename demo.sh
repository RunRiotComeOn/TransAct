#!/bin/bash

##############################################################################
# TransAct Demo Script
#
# This script demonstrates the complete TransAct pipeline:
# 1. Extract steering vectors from LLMs (Phase 1)
# 2. Apply steering to VLMs with entropy-based layer selection (Phase 2)
##############################################################################

set -e  # Exit on error

echo "=========================================================================="
echo "                       TransAct Demo Pipeline                            "
echo "=========================================================================="

# Configuration
STEERING_ABILITY="mathematical_reasoning"
STEERING_DATASET="gsm8k"
NUM_EXTRACTION_SAMPLES=50
TARGET_DATASET="geometry3k"
NUM_VAL_SAMPLES=100
NUM_TEST_SAMPLES=200
ALPHA=5.0

# Directory names
MODEL_NAME="qwen2_5_7b"
STEERING_OUTPUT_DIR="steering_outputs/${STEERING_DATASET}_${NUM_EXTRACTION_SAMPLES}_${MODEL_NAME}"

##############################################################################
# Phase 1: Extract Steering Vectors
##############################################################################

echo "=========================================================================="
echo "PHASE 1: Extracting Steering Vectors from LLMs"
echo "=========================================================================="
echo ""
echo "Steering Ability: ${STEERING_ABILITY}"
echo "Dataset: ${STEERING_DATASET}"
echo "Number of Samples: ${NUM_EXTRACTION_SAMPLES}"
echo ""

# Check if steering vectors already exist
if [ -d "${STEERING_OUTPUT_DIR}" ]; then
    echo "✓ Steering vectors already exist at: ${STEERING_OUTPUT_DIR}"
    echo "  Skipping extraction..."
    echo ""
else
    echo "Extracting steering vectors..."
    echo ""

    python extract_steering_vectors.py \
        --ability ${STEERING_ABILITY} \
        --num_samples ${NUM_EXTRACTION_SAMPLES} \
        --num_candidates $((NUM_EXTRACTION_SAMPLES * 3))

    echo ""
    echo "✓ Phase 1 complete! Steering vectors saved to: ${STEERING_OUTPUT_DIR}"
    echo ""
fi

##############################################################################
# Phase 2: Apply Steering to VLMs
##############################################################################

echo "=========================================================================="
echo "PHASE 2: Applying Steering to VLMs"
echo "=========================================================================="
echo ""
echo "Steering Type: ${STEERING_DATASET}_${NUM_EXTRACTION_SAMPLES}_${MODEL_NAME}"
echo "Target Dataset: ${TARGET_DATASET}"
echo "Alpha (Steering Strength): ${ALPHA}"
echo "Validation Samples: ${NUM_VAL_SAMPLES}"
echo "Test Samples: ${NUM_TEST_SAMPLES}"
echo ""

python main-vlm.py \
    --steering_type "${STEERING_DATASET}_${NUM_EXTRACTION_SAMPLES}_${MODEL_NAME}" \
    --target_dataset ${TARGET_DATASET} \
    --alpha ${ALPHA} \
    --num_val_samples ${NUM_VAL_SAMPLES} \
    --num_test_samples ${NUM_TEST_SAMPLES}

echo ""
echo "✓ Phase 2 complete!"
echo ""

##############################################################################
# Optional: Visualizations
##############################################################################

echo "=========================================================================="
echo "OPTIONAL: Generate Visualizations"
echo "=========================================================================="
echo ""
echo "Would you like to generate entropy curve visualizations? (requires more time)"
echo "Press Ctrl+C to skip, or press Enter to continue..."
read -t 10 || echo "Skipping visualizations..."

if [ $? -eq 0 ]; then
    echo ""
    echo "Generating entropy curve visualization..."
    echo ""

    python visualize_entropy_curve.py \
        --dataset ${TARGET_DATASET} \
        --num_samples 20 \
        --output_dir "./entropy_analysis"

    echo ""
    echo "✓ Visualizations saved to: ./entropy_analysis"
    echo ""
fi

