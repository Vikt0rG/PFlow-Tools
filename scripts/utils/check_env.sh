#!/bin/bash

# Utility script to check if the environment is set up correctly.
# Run this once before executing any other scripts.

# Usage: `source scripts/utils/check_env.sh [conda_env_name]`

# Potential issues:
# - Conda environment not activated
#   -> Make sure to run with source and not as a regular script with `bash`

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed. Please install Miniforge or Anaconda and try again."
    exit 1
fi

# Initialize conda (find conda.sh dynamically)
CONDA_INIT_SCRIPT="${HOME}/miniforge3/etc/profile.d/conda.sh"
if [ ! -f "$CONDA_INIT_SCRIPT" ]; then
    CONDA_INIT_SCRIPT="${HOME}/anaconda3/etc/profile.d/conda.sh"
fi

if [ -f "$CONDA_INIT_SCRIPT" ]; then
    source "$CONDA_INIT_SCRIPT"
else
    echo "Warning: conda initialization script not found. Conda may not work properly."
fi

# Use provided environment name or default to "salt"
CONDA_ENV_NAME="${1:-salt}"

# Check if the conda environment exists
if ! conda env list | grep -q "^$CONDA_ENV_NAME "; then
    echo "Conda environment '$CONDA_ENV_NAME' not found."
    exit 1
fi

# Activate the conda environment
conda activate "$CONDA_ENV_NAME" || {
    echo "Failed to activate conda environment '$CONDA_ENV_NAME'"
    exit 1
}

# Check if salt is installed and importable
if ! python -c "import salt" 2>/dev/null; then
    echo "SALT is not installed in the current conda environment"
    exit 1
fi

echo "Environment setup successful. Conda environment '$CONDA_ENV_NAME' is ready."