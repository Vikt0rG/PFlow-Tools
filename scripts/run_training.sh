#!/bin/bash
usage() {
    echo "Script used to run SALT training using predefined base config and a model config path."
    echo "NOTE: Run check_env.sh first before running this script."
    echo ""
    echo "Usage: $0 <REQUIRED ARGUMENTS> [OPTIONAL ARGUMENTS]"
    echo ""
    echo "REQUIRED ARGUMENTS:"
    echo "  -c | --config PATH          Path to the model configuration file"
    echo ""
    echo "OPTIONAL ARGUMENTS:"
    echo "  -b | --base-config PATH     Path to the base config file"
    echo "  -h | --help                 Show this help message and exit"
    echo ""
    echo "EXAMPLES:"
    echo "  $0 -c /path/to/config.yaml"
    echo "  $0 -c /path/to/config.yaml -b /path/to/base.yaml"
    exit 1
}

# Paths
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR=$(dirname "$SCRIPT_DIR")

CONFIG_BASE_PATH=$ROOT_DIR/salt-mpp-pflow/salt/configs/base.yaml

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
        -b|--base-config)
            if [[ -z "$2" ]]; then
                echo "Error: --base-config requires a value."
                usage
            fi
            CONFIG_BASE_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter passed: $1"
            usage
            ;;
    esac
done

if [[ -z "$CONFIG_PATH" ]]; then
    echo "Error: --config is required."
    usage
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found: $CONFIG_PATH"
    exit 1
fi
if [ ! -f "$CONFIG_BASE_PATH" ]; then
    echo "Error: Base config file not found: $CONFIG_BASE_PATH"
    exit 1
fi

# For some reason I needed data fetching but don't remember why??
# Fetch training data from config file
# DATA_PATH=$(python3 -c "
# import yaml; 
# with open('$CONFIG_PATH') as f: 
#     print(yaml.safe_load(f)['data']['train_file'])
# ") || {
#     echo "Failed to extract training data path from config"
#     exit 1
# }
# echo "Using training data: $DATA_PATH"
# 
# # Verify data file exists
# if [ ! -f "$DATA_PATH" ]; then
#     echo "Error: Training data file not found: $DATA_PATH"
#     exit 1
# fi

# Run training with SALT
salt fit -c "$CONFIG_BASE_PATH" -c "$CONFIG_PATH" --force