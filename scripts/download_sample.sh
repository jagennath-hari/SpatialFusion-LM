#!/bin/bash
set -e

BASE_URL="https://spatialfusionlm.s3.amazonaws.com/datasets"
TARGET_DIR="datasets"
DATASET="indoor_0"

LOCAL_PATH="${TARGET_DIR}/${DATASET}"
MANIFEST_URL="${BASE_URL}/${DATASET}/manifest.txt"

echo "⬇️  Downloading manifest for ${DATASET}..."
mkdir -p "$LOCAL_PATH"
wget -c --tries=100 --timeout=30 -q -O "$LOCAL_PATH/manifest.txt" "$MANIFEST_URL"

echo "📄 Found files:"
cat "$LOCAL_PATH/manifest.txt"

echo "⬇️  Downloading files..."
while read -r filename; do
    wget -c --tries=100 --timeout=30 --show-progress \
      -O "${LOCAL_PATH}/${filename}" \
      "${BASE_URL}/${DATASET}/${filename}"
done < "$LOCAL_PATH/manifest.txt"

echo "✅ ${DATASET} downloaded to ${LOCAL_PATH}"

rm -f "$LOCAL_PATH/manifest.txt"
