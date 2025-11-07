#!/bin/bash
# VecSec Dependencies Installer
# Installs all dependencies needed for VecSec including real vector retrieval migration

set -e  # Exit on error

echo "🔧 Installing VecSec Dependencies..."
echo ""

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt not found!"
    echo "   Please run this script from the project root directory."
    exit 1
fi

# Install from requirements.txt (recommended method)
echo "📦 Installing from requirements.txt..."
pip3 install -r requirements.txt

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

echo ""
echo "✅ Dependencies installed!"
echo ""
echo "📋 Installed packages:"
echo "   - langchain & langchain-core (RAG framework)"
echo "   - langchain-chroma (ChromaDB vector store)"
echo "   - langgraph (RAG orchestration)"
echo "   - sentence-transformers (local embeddings)"
echo "   - chromadb (persistent vector storage)"
echo "   - python-dotenv (environment configuration)"
echo "   - prometheus-client (metrics)"
echo ""
echo "🧪 To verify installation:"
echo "   python3 -c 'from langchain_core.vectorstores import InMemoryVectorStore; print(\"✅ langchain-core OK\")'"
echo "   python3 -c 'from langchain_chroma import Chroma; print(\"✅ langchain-chroma OK\")'"
echo "   python3 -c 'from sentence_transformers import SentenceTransformer; print(\"✅ sentence-transformers OK\")'"
echo ""
echo "🚀 Next steps:"
echo "  1. If you have existing packages, upgrade them:"
echo "     ./scripts/upgrade_dependencies.sh  # Fixes langchain_core.pydantic_v1 issues"
echo "  2. Set environment variables (optional):"
echo "     export USE_CHROMA=true  # Enable ChromaDB persistence"
echo "     export USE_REAL_VECTOR_RETRIEVAL=true  # Enable real vector retrieval (default)"
echo "  3. Run tests:"
echo "     python3 src/sec_agent/tests/test_rag_orchestrator_migration.py"
echo "  4. Run integration test:"
echo "     ./scripts/test_chromadb.sh"
echo ""
echo "💡 Troubleshooting:"
echo "   If you see 'ModuleNotFoundError: No module named langchain_core.pydantic_v1':"
echo "   Run: ./scripts/upgrade_dependencies.sh"
echo ""

