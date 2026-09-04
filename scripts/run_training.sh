#!/bin/bash

usage() {
    echo "Script used to run SALT training using predefined base config and
          a model config path."
    echo "NOTE: Run check_env.sh first before running this script."
    echo ""
    echo "Usage: $0 <REQUIRED ARGUMENTS> [OPTIONAL ARGUMENTS]"
    echo ""
    echo "REQUIRED ARGUMENTS:"
    echo "  -c | --config [<PATH>...]   Path(s) to the model configuration file(s)"
    echo ""
    echo "OPTIONAL ARGUMENTS:"
    echo "  -h | --help                 Show this help message and exit"
    echo ""
    echo "EXAMPLES:"
    echo "  $0 -c /path/to/config.yaml"
    exit 1
}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

CONFIG_PATHS=()

# Process CLI arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            ;;
        -c|--config)
            shift
            while [[ "$#" -gt 0 && ! "$1" =~ ^- ]]; do
                if [ ! -f "$1" ]; then
                    echo "Error: Config file not found: $1"
                    exit 1
                fi
                CONFIG_PATHS+=("$(realpath "$1")")
                shift
            done
            ;;
        *)
            echo "Unknown parameter passed: $1"
            usage
            ;;
    esac
done

# Validate CLI arguments
if [[ ${#CONFIG_PATHS[@]} -eq 0 ]]; then
    echo "Error: At least one --config file must be specified."
    usage
fi

CONFIG_ARGS=""
for cfg in "${CONFIG_PATHS[@]}"; do
    CONFIG_ARGS="$CONFIG_ARGS -c $cfg"
done
CONFIG_ARGS="$(echo "$CONFIG_ARGS" | xargs)"

# Run training with SALT
salt fit $CONFIG_ARGS --force