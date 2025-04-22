#!/bin/bash
set -e

BUCKET="spatialfusionlm"
S3_PREFIX="datasets"
TARGET_DIR="datasets"
DATASET_LIST=("indoor_0")

AWS_PROFILE="fusion"

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
    S3_PATH="s3://${BUCKET}/${S3_PREFIX}/${DATASET}"

    echo "⬇️  Syncing $DATASET from ${S3_PATH} → ${LOCAL_PATH} ..."
    aws s3 sync "$S3_PATH" "$LOCAL_PATH" --profile "$AWS_PROFILE"

    echo "✅ $DATASET downloaded to ${LOCAL_PATH}"
done
