#!/bin/bash
# Run all VecSec tests

echo "🧪 Running VecSec Test Suite"
echo "============================"
echo ""

# Check if pytest is available
if command -v pytest &> /dev/null; then
    echo "Using pytest..."
    pytest tests/ -v
else
    echo "Using unittest..."
    python3 -m unittest discover tests/ -v
fi

echo ""
echo "✅ Tests complete!"

