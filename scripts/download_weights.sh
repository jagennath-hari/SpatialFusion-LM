#!/bin/bash

set -e  # Exit on error

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
