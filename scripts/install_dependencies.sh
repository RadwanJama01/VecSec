#!/usr/bin/env bash
# VecSec Dependencies Installer
# Installs all dependencies needed for VecSec including real vector retrieval migration
# Matches CI dependencies for local development parity

set -e  # Exit on error

echo "🔧 Installing VecSec Dependencies..."
echo ""

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt not found!"
    echo "   Please run this script from the project root directory."
    exit 1
fi

# Auto-create and activate virtual environment if one isn't active
if [ -z "$VIRTUAL_ENV" ]; then
    echo "📦 Virtual environment not detected"
    if [ ! -d "venv" ]; then
        echo "   Creating virtual environment..."
        python3 -m venv venv
    fi
    echo "   Activating virtual environment..."
    source venv/bin/activate
    echo "   ✅ Virtual environment activated"
    echo ""
fi

# Upgrade pip first (matches CI)
echo "📦 Upgrading pip..."
python3 -m pip install --upgrade pip
echo ""

# Install base dependencies from requirements.txt
echo "📦 Installing base dependencies from requirements.txt..."
pip install -r requirements.txt
echo ""

# Install test & lint dependencies separately (matches CI workflow)
echo "🧪 Installing test & lint dependencies..."
pip install pytest pytest-cov ruff mypy
echo ""

# Alternative: Install core packages directly if requirements.txt fails
# Uncomment below if you prefer manual installation:
# echo "📦 Installing core packages..."
# pip3 install \
#     python-dotenv \
#     requests \
#     "numpy>=1.24.0,<2.0.0" \
#     prometheus-client \
#     "langchain>=0.1.20" \
#     "langchain-core>=1.0.0" \
#     "langchain-chroma>=0.1.3" \
#     "langgraph>=0.0.40" \
#     "chromadb>=0.5.3" \
#     "sentence-transformers>=2.2.0"

echo "✅ Dependencies installed successfully!"
echo ""
echo "📋 Installed packages:"
echo "   Core:"
echo "   - langchain & langchain-core (RAG framework)"
echo "   - langchain-chroma (ChromaDB vector store)"
echo "   - langgraph (RAG orchestration)"
echo "   - sentence-transformers (local embeddings)"
echo "   - chromadb (persistent vector storage)"
echo "   - python-dotenv (environment configuration)"
echo "   - prometheus-client (metrics)"
echo ""
echo "   Dev/CI:"
echo "   - pytest & pytest-cov (testing)"
echo "   - ruff (linting)"
echo "   - mypy (type checking)"
echo ""
echo "🧪 To verify installation:"
echo "   python3 -c 'from langchain_core.vectorstores import InMemoryVectorStore; print(\"✅ langchain-core OK\")'"
echo "   python3 -c 'from langchain_chroma import Chroma; print(\"✅ langchain-chroma OK\")'"
echo "   python3 -c 'from sentence_transformers import SentenceTransformer; print(\"✅ sentence-transformers OK\")'"
echo "   ruff --version && pytest --version && mypy --version"
echo ""
echo "🚀 Next steps:"
echo "  1. Run linting:"
echo "     ruff check src/ scripts/"
echo "  2. Run tests:"
echo "     pytest src/sec_agent/tests/ -v"
echo "  3. Set environment variables (optional):"
echo "     export USE_CHROMA=true  # Enable ChromaDB persistence (default: in-memory)"
echo "  4. Run integration test:"
echo "     ./scripts/test_chromadb.sh"
echo ""
echo "💡 Troubleshooting:"
echo "   If you see 'ModuleNotFoundError: No module named langchain_core.pydantic_v1':"
echo "   Run: ./scripts/upgrade_dependencies.sh"
echo ""

