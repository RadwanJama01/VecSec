#!/bin/bash
# VecSec Dependencies Installer

echo "🔧 Installing VecSec Dependencies..."
echo ""

# Install required packages
pip3 install python-dotenv requests numpy prometheus_client langchain langchain-core langchain-community langgraph --quiet

echo ""
echo "✅ Dependencies installed!"
echo ""
echo "Next steps:"
echo "  1. Run: python3 SHOWCASE_DEMO.py"
echo "  2. View metrics: cat learning_metrics.json"
echo ""

