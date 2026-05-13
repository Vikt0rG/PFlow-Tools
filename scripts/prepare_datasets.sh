#!/bin/bash
usage() {
    echo "Script used to prepare datasets using a configuration file."
    echo "NOTE: Run check_env.sh first before running this script."
    echo ""
    echo "Usage: $0 <REQUIRED ARGUMENTS> [OPTIONAL ARGUMENTS]"
    echo ""
    echo "REQUIRED ARGUMENTS:"
    echo "  -c | --config PATH          Path to the configuration file"
    echo ""
    echo "OPTIONAL ARGUMENTS:"
    echo "  -h | --help                 Show this help message and exit"
    echo ""
    echo "EXAMPLES:"
    echo "  $0 -c /path/to/config.yaml"
    exit 1
}

# TODO: Implement data conversion from the .root to .h5

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

h5-prep -c "$CONFIG_PATH"