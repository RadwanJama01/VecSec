# Test Commands Reference

## Quick Test Commands

### Run All Tests
```bash
# From project root
pytest src/sec_agent/tests/ -v
```

### Run Unit Tests Only (Fast, Mocked)
```bash
export LOG_LEVEL=WARNING
pytest src/sec_agent/tests/ \
  --maxfail=1 \
  --disable-warnings \
  --cov=src/sec_agent \
  --cov-report=term-missing \
  -v
```

### Run Integration Tests (Real Vector Store)
```bash
export USE_CHROMA=true
export CHROMA_PATH=./chroma_db_ci
pytest src/sec_agent/tests/test_rag_orchestrator.py \
  src/sec_agent/tests/test_metadata_generator.py \
  --maxfail=1 \
  --disable-warnings \
  -v
```

### Run Specific Test File
```bash
# Unit test for metadata generator
pytest src/sec_agent/tests/test_metadata_generator.py -v

# Unit test for orchestrator
pytest src/sec_agent/tests/test_rag_orchestrator.py -v

# Config tests
pytest src/sec_agent/tests/test_config_manager.py -v

# Policy tests
pytest src/sec_agent/tests/test_policy_manager.py -v

# Embeddings tests
pytest src/sec_agent/tests/test_embeddings_client.py -v

# ChromaDB integration tests
pytest src/sec_agent/tests/test_chroma_integration.py -v
```

### Run with Coverage
```bash
pytest src/sec_agent/tests/ \
  --cov=src/sec_agent \
  --cov-report=term-missing \
  --cov-report=html \
  -v
# Open htmlcov/index.html to view coverage report
```

### Run Tests Matching Pattern
```bash
# Run all tests with "metadata" in name
pytest src/sec_agent/tests/ -k metadata -v

# Run all tests with "orchestrator" in name
pytest src/sec_agent/tests/ -k orchestrator -v
```

## Test Categories

### Unit Tests (Fast, Isolated)
- `test_config_manager.py` - Configuration management
- `test_policy_manager.py` - Policy enforcement logic
- `test_embeddings_client.py` - Embedding client (mocked)
- `test_mock_llm.py` - Mock LLM functionality
- `test_metadata_generator.py` - Metadata generation (mocked vector store)
- `test_rag_orchestrator.py` - Orchestrator logic (mocked)

### Integration Tests (Slower, Real Components)
- `test_chroma_integration.py` - ChromaDB integration
- `test_sentence_transformers.py` - Sentence transformers integration
- Integration tests in `test_rag_orchestrator.py` (with real vector store)
- Integration tests in `test_metadata_generator.py` (with InMemoryVectorStore)

## CI Matching Commands

### Match CI Unit Tests Job
```bash
export LOG_LEVEL=WARNING
pytest src/sec_agent/tests/ \
  --maxfail=1 \
  --disable-warnings \
  --cov=src/sec_agent \
  --cov-report=term-missing \
  --cov-report=xml \
  -v
```

### Match CI Integration Tests Job
```bash
export USE_CHROMA=true
export CHROMA_PATH=./chroma_db_ci
export LOG_LEVEL=INFO
pytest src/sec_agent/tests/test_rag_orchestrator.py \
  src/sec_agent/tests/test_metadata_generator.py \
  --maxfail=1 \
  --disable-warnings \
  -v
```

## Troubleshooting

### Tests Fail with Import Errors
```bash
# Make sure you're in project root
cd /Users/dan/repos/VecSec

# Install dependencies
pip install -r requirements.txt
```

### ChromaDB Tests Fail
```bash
# Make sure ChromaDB is installed
pip install langchain-chroma chromadb

# Set environment variable
export USE_CHROMA=true
export CHROMA_PATH=./chroma_db_test
```

### Coverage Not Working
```bash
# Install coverage tools
pip install pytest-cov

# Run with coverage
pytest src/sec_agent/tests/ --cov=src/sec_agent --cov-report=term-missing
```

