#!/bin/bash
# Upgrade VecSec Dependencies
# Fixes langchain_core.pydantic_v1 compatibility issues

set -e  # Exit on error

echo "🔧 Upgrading VecSec Dependencies..."
echo ""

# Check if in virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: Not in a virtual environment"
    echo "   Consider activating your venv first:"
    echo "   source venv/bin/activate"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "📦 Upgrading langchain packages for compatibility..."
echo ""

# Upgrade langchain-core first (needed for pydantic_v1)
echo "1️⃣ Upgrading langchain-core to >= 0.2.0..."
pip3 install --upgrade "langchain-core>=0.2.0"

# Upgrade langgraph to compatible version
echo ""
echo "2️⃣ Upgrading langgraph to >= 0.2.0..."
pip3 install --upgrade "langgraph>=0.2.0"

# Install/upgrade all other dependencies
echo ""
echo "3️⃣ Installing/upgrading all dependencies from requirements.txt..."
pip3 install --upgrade -r requirements.txt

echo ""
echo "✅ Dependencies upgraded!"
echo ""
echo "🧪 Verifying installation..."
python3 << 'PYEOF'
import warnings
# Suppress deprecation warnings during verification (langgraph uses pydantic_v1 internally)
warnings.filterwarnings('ignore', category=DeprecationWarning, module='langchain_core')

try:
    from langchain_core.pydantic_v1 import BaseModel
    print("✅ langchain_core.pydantic_v1 is available")
    print("   (Note: Deprecation warning is from langgraph, not our code - safe to ignore)")
except ImportError as e:
    print(f"❌ langchain_core.pydantic_v1 NOT available: {e}")
    print("   Try: pip3 install --upgrade langchain-core>=0.2.0")
    exit(1)

try:
    import langgraph
    # langgraph may not have __version__ attribute in some versions
    try:
        version = langgraph.__version__
        print(f"✅ langgraph {version} installed")
    except AttributeError:
        print("✅ langgraph installed (version info not available)")
except ImportError as e:
    print(f"❌ langgraph NOT installed: {e}")
    exit(1)

print("\n✅ All dependencies verified!")
PYEOF

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Upgrade successful! You can now run:"
    echo "   python3 src/sec_agent/tests/test_rag_orchestrator_migration.py"
else
    echo ""
    echo "❌ Verification failed. Please check the errors above."
    exit 1
fi

