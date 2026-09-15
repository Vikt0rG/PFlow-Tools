#!/bin/bash

usage() {
    echo "Wrapper script to submit a SALT training job to SLURM."
    echo ""
    echo "Usage: $0 <REQUIRED ARGUMENTS> [OPTIONAL ARGUMENTS]"
    echo ""
    echo "REQUIRED ARGUMENTS:"
    echo "  -c | --config [<PATH>...]   Path(s) to the model configuration file(s)"
    echo ""
    echo "OPTIONAL ARGUMENTS:"
    echo "  -e | --env NAME             Name of the conda environment to use.
                                        Default: salt"
    echo "  -h | --help                 Show this help message and exit"
    echo ""
    echo "EXAMPLES:"
    echo "  $0 -c /path/to/config.yaml"
    exit 1
}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SLURM_SCRIPT="$SCRIPT_DIR/run_training.slurm"

CONFIG_PATHS=()

# Parse CLI arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -h|--help) usage ;;
        -c|--config)
            shift
            if [[ "$#" -eq 0 || "$1" =~ ^- ]]; then
                echo "Error: --config requires at least one file path."
                usage
            fi
            while [[ "$#" -gt 0 && ! "$1" =~ ^- ]]; do
                if [ ! -f "$1" ]; then
                    echo "Error: Config file not found: $1"
                    exit 1
                fi
                CONFIG_PATHS+=("$(realpath "$1")")
                shift
            done
            ;;
        -e|--env)
            if [[ -z "$2" ]]; then
                echo "Error: --env requires a value."
                usage
            fi
            CONDA_ENV_NAME="$2"
            shift 2
            ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# Validate inputs
if [[ ${#CONFIG_PATHS[@]} -eq 0 ]]; then
    echo "Error: At least one --config file must be specified."
    exit 1
fi

if [ ! -f "$SLURM_SCRIPT" ]; then
    echo "Error: Slurm script not found: $SLURM_SCRIPT"
    exit 1
fi

# Build space-delimited string of -c flags: "-c /path/1 -c /path/2"
CONFIG_ARGS=""
for cfg in "${CONFIG_PATHS[@]}"; do
    CONFIG_ARGS="$CONFIG_ARGS -c $cfg"
done
CONFIG_ARGS="$(echo "$CONFIG_ARGS" | xargs)"

export CONFIG_ARGS
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-salt}" # Default to "salt" if not provided

echo "Submitting job with:"
echo "  Model Configs: ${CONFIG_PATHS[*]}"
echo "  Config Flag String: $CONFIG_ARGS"
echo "  Conda Environment: $CONDA_ENV_NAME"

sbatch --export=ALL "$SLURM_SCRIPT"