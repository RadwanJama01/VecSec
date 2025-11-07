# Migration: Real Vector Retrieval Integration

## Overview

This document describes the migration from mock metadata generation to real vector store retrieval in the RAG Orchestrator. The migration includes feature flags, backward compatibility, logging, and comprehensive testing.

## Implementation Status

✅ **All requirements completed**

### 1. Orchestrator Initializes Vector Store Connection ✅

**Location**: `src/sec_agent/rag_orchestrator.py`

The `RAGOrchestrator` class receives the vector store via constructor injection:

```python
def __init__(self, vector_store, llm, prompt_template, threat_embedder=None, qwen_client=None):
    self.vector_store = vector_store
    # ... other initialization
```

The vector store is initialized in:
- `src/sec_agent/cli.py` (line 31)
- `src/Sec_Agent.py` (line 33)
- Any code that creates `RAGOrchestrator` instances

### 2. Calls New `generate_retrieval_metadata_real()` ✅

**Location**: `src/sec_agent/rag_orchestrator.py` (lines 91-95)

The orchestrator now calls the real retrieval function when the feature flag is enabled:

```python
retrieval_metadata = generate_retrieval_metadata_real(
    query_context, 
    tenant_id, 
    self.vector_store
)
```

### 3. Feature Flag for Gradual Rollout ✅

**Location**: `src/sec_agent/config.py`

**Feature Flag**: `USE_REAL_VECTOR_RETRIEVAL`

- **Default**: `true` (real retrieval enabled by default)
- **Environment Variable**: `USE_REAL_VECTOR_RETRIEVAL`
- **Config Schema**: Added to `CONFIG_SCHEMA` (line 158-163)
- **Constant**: `USE_REAL_VECTOR_RETRIEVAL` exported from `config.py` (line 378-381)

**Usage**:
```bash
# Enable real retrieval (default)
export USE_REAL_VECTOR_RETRIEVAL=true

# Disable real retrieval (use mock)
export USE_REAL_VECTOR_RETRIEVAL=false
```

**Implementation Logic** (`rag_orchestrator.py` lines 81-136):
- If `USE_REAL_VECTOR_RETRIEVAL=true` AND `vector_store is not None` → Use real retrieval
- If `USE_REAL_VECTOR_RETRIEVAL=false` → Use mock retrieval
- If `vector_store is None` → Fallback to mock retrieval
- If real retrieval fails → Fallback to mock retrieval

### 4. Backward Compatibility During Migration ✅

**Multiple fallback mechanisms**:

1. **Feature Flag Disabled**: Uses `generate_retrieval_metadata()` (mock)
2. **Vector Store None**: Falls back to mock metadata generator
3. **Real Retrieval Error**: Catches exceptions and falls back to mock
4. **API Contract**: Unchanged - `rag_with_rlsa()` signature identical

**Code** (`rag_orchestrator.py` lines 104-115, 116-136):
```python
try:
    retrieval_metadata = generate_retrieval_metadata_real(...)
except Exception as e:
    # Fallback to mock on error
    retrieval_metadata = generate_retrieval_metadata(...)
```

### 5. Logging for Monitoring Migration ✅

**Location**: `src/sec_agent/rag_orchestrator.py`

**Logging Events**:

1. **Real Retrieval Start** (INFO level, line 82-89):
   ```python
   logger.info(
       "Using real vector store for retrieval metadata",
       extra={
           "tenant_id": tenant_id,
           "migration": "real_vector_retrieval",
           "query_preview": query[:100]
       }
   )
   ```

2. **Real Retrieval Success** (DEBUG level, line 96-103):
   ```python
   logger.debug(
       f"Real vector retrieval successful: {len(retrieval_metadata)} results",
       extra={"tenant_id": tenant_id, "result_count": len(retrieval_metadata)}
   )
   ```

3. **Real Retrieval Fallback** (WARNING level, line 106-114):
   ```python
   logger.warning(
       f"Real vector retrieval failed, falling back to mock: {e}",
       exc_info=True,
       extra={
           "tenant_id": tenant_id,
           "migration": "real_vector_retrieval_fallback",
           "error_type": type(e).__name__
       }
   )
   ```

4. **Mock Retrieval (Feature Flag Disabled)** (DEBUG level, line 119-126):
   ```python
   logger.debug(
       "Feature flag disabled, using mock metadata generator",
       extra={"migration": "mock_metadata", "feature_flag": "USE_REAL_VECTOR_RETRIEVAL=false"}
   )
   ```

5. **Mock Retrieval (Vector Store None)** (WARNING level, line 128-135):
   ```python
   logger.warning(
       "Vector store is None, falling back to mock metadata generator",
       extra={"migration": "mock_metadata_fallback", "reason": "vector_store_none"}
   )
   ```

**Structured Logging Fields**:
- `migration`: Migration state identifier
- `tenant_id`: Tenant for filtering
- `result_count`: Number of results retrieved
- `error_type`: Exception type (on errors)
- `query_preview`: First 100 chars of query

### 6. No Breaking Changes to API Contract ✅

**API Contract Preserved**:

```python
def rag_with_rlsa(self, user_id, tenant_id, clearance, query, role="analyst"):
    # Signature unchanged
    # Return type unchanged (bool)
    # Behavior unchanged (True = allowed, False = blocked)
```

**Verification**:
- Function signature: ✅ Unchanged
- Return type: ✅ Still `bool`
- Parameters: ✅ All parameters identical
- Behavior: ✅ Same external behavior, internal implementation improved

## Testing

### Unit Tests ✅

**File**: `src/sec_agent/tests/test_rag_orchestrator_migration.py`

**Test Coverage**:

1. ✅ `test_feature_flag_defaults_to_true` - Verifies default is True
2. ✅ `test_feature_flag_respects_env_var` - Verifies env var parsing
3. ✅ `test_uses_real_retrieval_when_flag_enabled` - Real retrieval path
4. ✅ `test_uses_mock_retrieval_when_flag_disabled` - Mock retrieval path
5. ✅ `test_falls_back_to_mock_on_real_retrieval_error` - Error handling
6. ✅ `test_falls_back_to_mock_when_vector_store_none` - None handling
7. ✅ `test_logs_migration_info_when_using_real_retrieval` - Logging verification
8. ✅ `test_logs_migration_info_when_using_mock_retrieval` - Logging verification
9. ✅ `test_api_contract_unchanged` - API contract preservation
10. ✅ `test_passes_vector_store_to_real_retrieval` - Parameter passing

### Integration Tests ✅

**File**: `src/sec_agent/tests/test_rag_orchestrator_migration.py`

**Test Coverage**:

1. ✅ `test_integration_real_retrieval_with_inmemory_store` - Real vector store integration
2. ✅ `test_integration_backward_compatibility_mock_fallback` - Backward compatibility

## Migration Checklist

- [x] Feature flag added to config schema
- [x] Feature flag constant exported from config
- [x] Orchestrator checks feature flag
- [x] Real retrieval function called when enabled
- [x] Fallback to mock when flag disabled
- [x] Fallback to mock when vector_store is None
- [x] Fallback to mock on real retrieval errors
- [x] Migration logging implemented
- [x] API contract preserved
- [x] Unit tests created
- [x] Integration tests created

## Usage Examples

### Enable Real Retrieval (Default)
```python
from src.sec_agent.rag_orchestrator import RAGOrchestrator
from src.sec_agent.config import initialize_vector_store

# Vector store is initialized
vector_store = initialize_vector_store(embeddings)

# Orchestrator uses real retrieval by default
orchestrator = RAGOrchestrator(
    vector_store=vector_store,
    llm=llm,
    prompt_template=template
)

# Real retrieval will be used
result = orchestrator.rag_with_rlsa(...)
```

### Disable Real Retrieval (Use Mock)
```bash
export USE_REAL_VECTOR_RETRIEVAL=false
```

```python
# Mock retrieval will be used
result = orchestrator.rag_with_rlsa(...)
```

### Monitor Migration via Logs
```bash
# Enable logging to file
export LOG_FILE=vecsec.log
export LOG_LEVEL=DEBUG

# Check migration status
grep "migration" vecsec.log
```

## Rollout Strategy

1. **Phase 1**: Deploy with `USE_REAL_VECTOR_RETRIEVAL=true` (default)
   - Monitor logs for `real_vector_retrieval` events
   - Watch for `real_vector_retrieval_fallback` warnings

2. **Phase 2**: If issues occur, temporarily disable:
   ```bash
   export USE_REAL_VECTOR_RETRIEVAL=false
   ```
   See [Rollback Plan](./ROLLBACK_PLAN.md) for detailed rollback procedures.

3. **Phase 3**: Once stable, remove feature flag (future cleanup)
   - Remove `USE_REAL_VECTOR_RETRIEVAL` checks
   - Always use real retrieval
   - Remove mock fallback (or keep as emergency fallback)

## Rollback Plan

If issues are discovered, see the [Rollback Plan](./ROLLBACK_PLAN.md) for:
- Quick rollback procedures (feature flag)
- Full code rollback steps
- Rollback verification checklist
- When to rollback decision tree

## Files Modified

1. `src/sec_agent/config.py`
   - Added `USE_REAL_VECTOR_RETRIEVAL` to `CONFIG_SCHEMA`
   - Added `USE_REAL_VECTOR_RETRIEVAL` constant

2. `src/sec_agent/rag_orchestrator.py`
   - Added logging import
   - Added feature flag import
   - Updated `rag_with_rlsa()` to use feature flag
   - Added migration logging

3. `src/sec_agent/tests/test_rag_orchestrator_migration.py` (NEW)
   - Unit tests for feature flag behavior
   - Integration tests for real vector store
   - API contract verification tests

## Verification

To verify the migration is working:

```bash
# 1. Check feature flag is enabled (default)
python3 -c "from src.sec_agent.config import USE_REAL_VECTOR_RETRIEVAL; print(USE_REAL_VECTOR_RETRIEVAL)"
# Should output: True

# 2. Run integration test
python3 scripts/test_chromadb.sh

# 3. Check logs for migration events
grep -i "migration\|real vector" vecsec.log
```

## Summary

✅ All migration requirements have been implemented:
- Vector store connection initialized ✅
- Real retrieval function called ✅
- Feature flag for gradual rollout ✅
- Backward compatibility maintained ✅
- Migration logging implemented ✅
- API contract preserved ✅
- Unit and integration tests created ✅

The migration is **production-ready** and can be rolled out gradually using the feature flag.

