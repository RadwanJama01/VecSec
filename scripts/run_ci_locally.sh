#!/usr/bin/env bash
# Run GitHub Actions CI Pipeline Locally
# Replicates the CI workflow to test before pushing

set -e  # Exit on error

echo "🚀 Running GitHub Actions CI Pipeline Locally"
echo "=============================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track failures
FAILED=0

# ============================================================================
# 1. Lint & Type Check (matches CI lint job)
# ============================================================================
echo -e "${YELLOW}📋 Step 1: Lint & Type Check${NC}"
echo "-----------------------------------"

echo "Running ruff (linting)..."
if ruff check src/ scripts/ --output-format=github; then
    echo -e "${GREEN}✅ Ruff linting passed${NC}"
else
    echo -e "${RED}❌ Ruff linting failed${NC}"
    FAILED=1
fi

echo ""
echo "Running ruff (formatting check)..."
if ruff format --check src/ scripts/; then
    echo -e "${GREEN}✅ Ruff formatting check passed${NC}"
else
    echo -e "${RED}❌ Ruff formatting check failed${NC}"
    FAILED=1
fi

echo ""
echo "Running mypy (type checking)..."
if mypy . --ignore-missing-imports --no-strict-optional; then
    echo -e "${GREEN}✅ Mypy type checking passed${NC}"
else
    echo -e "${YELLOW}⚠️  Mypy type checking has warnings (non-blocking in CI)${NC}"
    # Don't fail on mypy errors since CI has continue-on-error: true
fi

echo ""
echo ""

# ============================================================================
# 2. Unit Tests (matches CI unit-tests job)
# ============================================================================
echo -e "${YELLOW}🧪 Step 2: Unit Tests${NC}"
echo "-----------------------------------"

export USE_REAL_VECTOR_RETRIEVAL=false
export LOG_LEVEL=WARNING

echo "Running unit tests with mock retrieval..."
if pytest src/sec_agent/tests/ \
    --maxfail=1 \
    --disable-warnings \
    --cov=src/sec_agent \
    --cov-report=term-missing \
    -v; then
    echo -e "${GREEN}✅ Unit tests passed${NC}"
else
    echo -e "${RED}❌ Unit tests failed${NC}"
    FAILED=1
fi

echo ""
echo ""

# ============================================================================
# 3. Integration Smoke Tests (matches CI integration-smoke job)
# ============================================================================
echo -e "${YELLOW}🔗 Step 3: Integration Smoke Tests${NC}"
echo "-----------------------------------"

export USE_REAL_VECTOR_RETRIEVAL=true
export USE_CHROMA=true
export CHROMA_PATH=./chroma_db_ci

echo "Running integration tests with real vector store (ChromaDB)..."
echo "Using CHROMA_PATH=${CHROMA_PATH}"

# Note: Integration tests may require ChromaDB setup
# If ChromaDB is not available, these tests will be skipped
if pytest src/sec_agent/tests/test_rag_orchestrator_migration.py \
    src/sec_agent/tests/test_metadata_generator_real.py \
    -v; then
    echo -e "${GREEN}✅ Integration tests passed${NC}"
else
    echo -e "${YELLOW}⚠️  Integration tests failed or skipped (may need ChromaDB setup)${NC}"
    # Integration tests are optional in CI, so don't fail the script
fi

echo ""
echo ""

# ============================================================================
# Summary
# ============================================================================
echo "=============================================="
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ All CI checks passed! Ready to push.${NC}"
    exit 0
else
    echo -e "${RED}❌ Some CI checks failed. Fix issues before pushing.${NC}"
    exit 1
fi

