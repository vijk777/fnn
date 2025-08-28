#!/usr/bin/env bash
set -euo pipefail

# Defaults
OUTDIR="/workspace/fnn/data/microns_digital_twin/properties/"
INCLUDE_RESPONSES=false

# Parse arguments
for arg in "$@"; do
  if [[ "$arg" == "--include-responses" ]]; then
    INCLUDE_RESPONSES=true
  elif [[ "$arg" != "" && "$arg" != --* ]]; then
    OUTDIR="$arg"
  fi
done

echo "Downloading MICrONS model properties to: $OUTDIR"
echo "Include responses: $INCLUDE_RESPONSES"

# Check for aws CLI
if ! command -v aws &> /dev/null; then
  echo "Error: AWS CLI not found. Please install it with 'sudo apt install awscli' or 'pip install awscli'."
  exit 1
fi

# Create directory if needed
mkdir -p "$OUTDIR"

# Base path
S3_PATH="s3://bossdb-open-data/iarpa_microns/minnie/functional_data/digital_twin_properties/v2"

# Perform the download
if [ "$INCLUDE_RESPONSES" = true ]; then
  aws s3 sync "$S3_PATH" "$OUTDIR" --no-sign-request
else
  echo "Skipping 'responses' directory..."
  aws s3 sync "$S3_PATH" "$OUTDIR" --exclude "responses/*" --no-sign-request
fi

echo "Download complete."
