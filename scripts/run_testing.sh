#!/bin/bash
usage() {
    echo "Script used to run SALT evaluation using predefined config and test file paths."
    echo "NOTE: Run check_env.sh first before running this script."
    echo ""
    echo "Usage: $0 <REQUIRED ARGUMENTS> [OPTIONAL ARGUMENTS]"
    echo ""
    echo "REQUIRED ARGUMENTS:"
    echo "  -c | --config PATH          Path to the configuration file"
    echo "  -t | --test-file PATH       Path to the test data file"
    echo "  -d | --devices BOOLEAN      Boolean value for trainer devices, e.g., 1 for using GPU"
    echo ""
    echo "OPTIONAL ARGUMENTS:"
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
            if [[ -z "$2" ]]; then
                echo "Error: --test-file requires a value."
                usage
            fi
            TEST_FILE="$2"
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
        *)
            echo "Unknown parameter passed: $1"
            usage
            ;;
    esac
done

# Verify argument files exist
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found: $CONFIG_PATH"
    exit 1
fi
if [ ! -f "$TEST_FILE" ]; then
    echo "Error: Test file not found: $TEST_FILE"
    exit 1
fi

# Run testing with SALT
salt test -c "$CONFIG_PATH" --data.test_file "$TEST_FILE" --trainer.devices="$TRAINER_DEVICES"