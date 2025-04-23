#!/bin/bash

set -e  # Exit on error

# ---------------------
# FoundationStereo Setup
# ---------------------

TARGET_DIR="spatialfusion-lm/FoundationStereo/pretrained_models"
mkdir -p "$TARGET_DIR"

# Public S3 URLs
FILES=(
  "https://spatialfusionlm.s3.amazonaws.com/models/FoundationStereo/cfg.yaml cfg.yaml"
  "https://spatialfusionlm.s3.amazonaws.com/models/FoundationStereo/model_best_bp2.pth model_best_bp2.pth"
)

# Download each file
for entry in "${FILES[@]}"; do
    read -r url filename <<< "$entry"
    echo "⬇️ Downloading ${filename}..."
    wget -q --show-progress -O "${TARGET_DIR}/${filename}" "$url"
done

echo "✅ All weights downloaded to '${TARGET_DIR}'!"

# ---------------------
# SpatialLM Setup
# ---------------------

SPATIALLM_DIR="spatialfusion-lm/SpatialLM/manycore-research/SpatialLM-Llama-1B"
SPATIALLM_PARENT=$(dirname "$SPATIALLM_DIR")

# Check for git-lfs
if ! command -v git-lfs &> /dev/null; then
    echo "🛠 git-lfs not found. Installing..."
    sudo apt update && sudo apt install -y git-lfs
    git lfs install
else
    echo "✅ git-lfs is already installed."
fi

# Ensure parent directory exists
mkdir -p "$SPATIALLM_PARENT"


if [ ! -d "$SPATIALLM_DIR" ]; then
    echo "⬇️ Cloning SpatialLM repo (LFS smudge skipped)..."
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/manycore-research/SpatialLM-Llama-1B "$SPATIALLM_DIR"

    echo "📦 Pulling LFS model files..."
    cd "$SPATIALLM_DIR"

    # Fetch all LFS objects and check them out
    git lfs pull
    git lfs checkout  # Ensures working tree gets the files

    cd - > /dev/null
else
    echo "✅ SpatialLM model already exists at ${SPATIALLM_DIR}"
fi

echo "✅ All model weights downloaded!"
