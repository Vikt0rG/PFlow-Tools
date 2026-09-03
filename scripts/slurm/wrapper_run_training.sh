#!/bin/bash

usage() {
    echo "Usage: $0 -c <MODEL_CONFIG> [-b <BASE_CONFIG>]"
    echo "  -c | --config PATH         Path to model configuration file (Required)"
    echo "  -b | --base-config PATH     Path to base configuration file (Optional)"
    echo "  -h | --help                 Show this help message"
    exit 1
}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR=$(dirname "$SCRIPT_DIR")

CONFIG_BASE_PATH=$ROOT_DIR/salt-mpp-pflow/salt/configs/base.yaml
SLURM_SCRIPT="$SCRIPT_DIR/run_training.slurm"

# Parse CLI arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -h|--help) usage ;;
        -c|--config) CONFIG_PATH="$(realpath "$2")"; shift 2 ;;
        -b|--base-config) CONFIG_BASE_PATH="$(realpath "$2")"; shift 2 ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# Validate inputs
if [[ -z "$CONFIG_PATH" ]] || [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Valid --config file required."
    exit 1
fi

if [ ! -f "$CONFIG_BASE_PATH" ]; then
    echo "Error: Base config file not found: $CONFIG_BASE_PATH"
    exit 1
fi

if [ ! -f "$SLURM_SCRIPT" ]; then
    echo "Error: Slurm script not found: $SLURM_SCRIPT"
    exit 1
fi

# Export variables and submit the slurm script
export CONFIG_BASE_PATH
export CONFIG_PATH

echo "Submitting job with:"
echo "  Base Config : $CONFIG_BASE_PATH"
echo "  Model Config: $CONFIG_PATH"

sbatch --export=ALL "$SLURM_SCRIPT"
