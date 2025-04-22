#!/bin/bash
set -e

BASE_URL="https://spatialfusionlm.s3.amazonaws.com/datasets"
TARGET_DIR="datasets"
DATASET_LIST=("indoor_0")

echo "📦 Available datasets:"
for i in "${!DATASET_LIST[@]}"; do
    echo "  [$i] ${DATASET_LIST[$i]}"
done

read -p "🔢 Enter the number(s) of datasets to download (comma-separated, e.g., 0,2): " SELECTION

IFS=',' read -ra INDICES <<< "$SELECTION"
mkdir -p "${TARGET_DIR}"

for i in "${INDICES[@]}"; do
    DATASET="${DATASET_LIST[$i]}"
    if [ -z "$DATASET" ]; then
        echo "❌ Invalid index: $i"
        continue
    fi

    LOCAL_PATH="${TARGET_DIR}/${DATASET}"
    MANIFEST_URL="${BASE_URL}/${DATASET}/manifest.txt"

    echo "⬇️  Downloading manifest for ${DATASET}..."
    mkdir -p "$LOCAL_PATH"
    wget -q -O "$LOCAL_PATH/manifest.txt" "$MANIFEST_URL"

    echo "📄 Found files:"
    cat "$LOCAL_PATH/manifest.txt"

    echo "⬇️  Downloading files..."
    while read -r filename; do
        wget -q --show-progress \
          -O "${LOCAL_PATH}/${filename}" \
          "${BASE_URL}/${DATASET}/${filename}"
    done < "$LOCAL_PATH/manifest.txt"

    echo "✅ $DATASET downloaded to ${LOCAL_PATH}"

    rm -f "$LOCAL_PATH/manifest.txt"
done
