#!/bin/bash

usage() {
    echo "Wrapper script to submit a SALT evaluation job to SLURM."
    echo ""
    echo "Usage: $0 <REQUIRED ARGUMENTS> [OPTIONAL ARGUMENTS]"
    echo ""
    echo "REQUIRED ARGUMENTS:"
    echo "  -c | --config PATH          Path to model configuration file
                                        from the logs directory"
    echo ""
    echo "OPTIONAL ARGUMENTS:"
    echo "  -t | --test-file PATH       Path to the test data file"
    echo "  -d | --devices VALUE        Value passed to '--trainer.devices'
                                        (e.g. 0 for CPU, 1 for one GPU). Default: 1"
    echo "  -e | --env NAME             Name of the conda environment to use.
                                        Default: salt"
    echo "  -h | --help                 Show this help message and exit"
    echo ""
    echo "EXAMPLES:"
    echo "  $0 -c /path/to/config.yaml -t /path/to/test_file.h5 -d 1"
    exit 1
}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SLURM_SCRIPT="$SCRIPT_DIR/run_testing.slurm"

# Process CLI arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            ;;
        -c|--config)
            if [[ -z "$2" ]]; then
                echo "Error: --config requires a value."
                usage
            fi
            CONFIG_PATH="$2"
            shift 2
            ;;
        -t|--test-file)
            if [[ -z "$2" ]]; then
                echo "Error: --test-file requires a value."
                usage
            fi
            TEST_FILE_PATH="$2"
            shift 2
            ;;
        -d|--devices)
            if [[ -z "$2" ]]; then
                echo "Error: --devices requires a value."
                usage
            fi
            TRAINER_DEVICES="$2"
            shift 2
            ;;
        -e|--env)
            if [[ -z "$2" ]]; then
                echo "Error: --env requires a value."
                usage
            fi
            CONDA_ENV_NAME="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter passed: $1"
            usage
            ;;
    esac
done

# Validate CLI arguments
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found: $CONFIG_PATH"
    exit 1
fi

if [ ! -f "$SLURM_SCRIPT" ]; then
    echo "Error: Slurm script not found: $SLURM_SCRIPT"
    exit 1
fi

if [ ! -f "$TEST_FILE_PATH" ]; then
    TEST_FILE_PATH=$(python3 -c "
import yaml
with open('$CONFIG_PATH') as f:
    print(yaml.safe_load(f)['data']['test_file'])
") || {
        echo "Failed to extract testing data path from config"
        exit 1
    }
fi

# Verify test file exists
if [ ! -f "$TEST_FILE_PATH" ]; then
    echo "Error: Testing data file not found: $TEST_FILE_PATH"
    exit 1
fi

# Export variables and submit the slurm script
export CONFIG_PATH
export TEST_FILE_PATH
export TRAINER_DEVICES="${TRAINER_DEVICES:-1}"  # Default to GPU
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-salt}" # Default to "salt" if not provided

echo "Submitting job with:"
echo "  Config : $CONFIG_PATH"
echo "  Test file: $TEST_FILE_PATH"
echo "  Devices: $TRAINER_DEVICES"
echo "  Conda Environment: $CONDA_ENV_NAME"

sbatch --export=ALL "$SLURM_SCRIPT"