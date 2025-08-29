#!/usr/bin/env bash
set -euo pipefail

# Default output directory (can be overridden by first arg)
OUTDIR="${1:-/workspace/fnn/data/train_digital_twin/}"

echo "Downloading training data to: $OUTDIR"

# Check for aws CLI
if ! command -v aws &> /dev/null; then
  echo "Error: AWS CLI not found. Install with 'sudo apt install awscli' or 'pip install awscli'."
  exit 1
fi

# Create directory if needed
mkdir -p "$OUTDIR"

# Source path
S3_PATH="s3://bossdb-open-data/iarpa_microns/minnie/functional_data/foundation_model/train_digital_twin"

# Perform the download (public bucket, no creds needed)
aws s3 sync "$S3_PATH" "$OUTDIR" --no-sign-request

echo "Download complete."
