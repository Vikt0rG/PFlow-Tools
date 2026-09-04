#!/bin/bash

usage() {
    echo "Generic script to run SALT evaluation."
    echo ""
    echo "Usage: $0 <REQUIRED ARGUMENTS> [OPTIONAL ARGUMENTS]"
    echo ""
    echo "REQUIRED ARGUMENTS:"
    echo "  -c | --config PATH          Path to model configuration file
                                        (from comet logs directory)"
    echo ""
    echo "OPTIONAL ARGUMENTS:"
    echo "  -t | --test-file PATH       Path to the HDF5 test data file. If not 
                                        specified, inferred from the config file."
    echo "  -d | --devices BOOLEAN      Boolean value indicating whether to evaluate
                                        on GPU (1) or CPU (0). Default: 0"
    echo "  -h | --help                 Show this help message and exit"
    echo ""
    echo "EXAMPLES:"
    echo "  $0 -c /path/to/config.yaml -t /path/to/test_file.h5 -d 1"
    exit 1
}

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
            TEST_FILE_PATH="$2"
            shift 2
            ;;
        -d|--devices)
            TRAINER_DEVICES="$2"
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

TRAINER_DEVICES="${TRAINER_DEVICES:-0}"  # Default to CPU

# Run testing with SALT
salt test \
    -c "$CONFIG_PATH" \
    --data.test_file "$TEST_FILE_PATH" \
    --trainer.devices="$TRAINER_DEVICES"