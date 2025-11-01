#!/bin/bash
# Run sec_agent module tests from project root

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get project root (parent of tests directory)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project root
cd "$PROJECT_ROOT"

# Run the specific test
if [ "$1" = "config" ]; then
    python src/sec_agent/tests/test_config_manager.py
else
    echo "Usage: ./run_sec_agent_tests.sh [config]"
    echo "Available tests:"
    echo "  config - Run config manager diagnostic test"
fi

