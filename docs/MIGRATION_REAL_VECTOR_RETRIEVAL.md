# Migration: Real Vector Retrieval Integration

> **⚠️ HISTORICAL DOCUMENT**  
> This document describes a completed migration that occurred in the past. The migration is **fully complete** and the feature flag has been removed. This document is kept for historical reference only.  
> **Current Status**: All code uses real vector store retrieval. No feature flag exists. Mock function has been removed.

## Overview

This document describes the completed migration from mock metadata generation to real vector store retrieval in the RAG Orchestrator. **The migration is complete** - the mock function has been removed and all code now uses real vector store retrieval.

## Implementation Status

✅ **Migration Complete** - Mock function removed, all code uses real vector retrieval

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

### 2. Uses `generate_retrieval_metadata()` with Vector Store ✅

**Location**: `src/sec_agent/rag_orchestrator.py`

The orchestrator calls the retrieval function with the vector store:

```python
retrieval_metadata = generate_retrieval_metadata(
    query_context, 
    tenant_id, 
    self.vector_store
)
```

**Note**: The function `generate_retrieval_metadata()` now requires a `vector_store` parameter. The mock function has been completely removed.

### 3. Migration Complete - Feature Flag Removed ✅

**Status**: ✅ **COMPLETE** - The feature flag `USE_REAL_VECTOR_RETRIEVAL` has been removed from the codebase. All code now uses real vector store retrieval. The mock function has been completely removed.

**Historical Implementation** (for reference - no longer applicable):
- Feature flag was used during migration for gradual rollout (now removed)
- Default was `true` (real retrieval enabled)
- Environment variable: `USE_REAL_VECTOR_RETRIEVAL` (no longer exists)
- **Constant**: `USE_REAL_VECTOR_RETRIEVAL` was exported from `config.py` (removed)

**Current Implementation**:
- All code always uses real vector store retrieval
- No feature flag exists
- If `vector_store is None` → Returns empty metadata (no fallback)
- If retrieval fails → Returns error metadata (no fallback)

### 4. Backward Compatibility During Migration ✅

**Error Handling** (Current Implementation):

1. **Vector Store None**: Returns empty metadata with warning log
2. **Retrieval Error**: Returns error metadata with error log
3. **API Contract**: Unchanged - `rag_with_rlsa()` signature identical

**Code** (`rag_orchestrator.py`):
```python
if self.vector_store is None:
    # Return empty metadata when vector store unavailable
    retrieval_metadata = [{"embedding_id": "emb-empty", ...}]
else:
    try:
        retrieval_metadata = generate_retrieval_metadata(
            query_context, tenant_id, self.vector_store
        )
    except Exception as e:
        # Return error metadata on failure
        retrieval_metadata = [{"embedding_id": "emb-error", ...}]
```

**Note**: The mock function has been completely removed. There is no fallback to mock metadata generation.

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

3. **Retrieval Error** (ERROR level):
   ```python
   logger.error(
       f"Vector retrieval failed: {e}",
       exc_info=logger.isEnabledFor(logging.DEBUG),
       extra={
           "tenant_id": tenant_id,
           "error_type": type(e).__name__
       }
   )
   ```

**Note**: The logging examples above (items 4-5) are historical and no longer apply. Mock fallback logging has been removed.

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

**File**: `src/sec_agent/tests/test_rag_orchestrator.py` (renamed from `test_rag_orchestrator_migration.py`)

**Test Coverage**:

1. ✅ `test_uses_vector_store_retrieval` - Vector store retrieval path
2. ✅ `test_handles_vector_store_error` - Error handling
3. ✅ `test_handles_none_vector_store` - None handling
4. ✅ `test_logs_retrieval_info` - Logging verification
5. ✅ `test_api_contract_unchanged` - API contract preservation

**Note**: Feature flag tests have been removed as the flag no longer exists.

### Integration Tests ✅

**File**: `src/sec_agent/tests/test_rag_orchestrator.py`

**Test Coverage**:

1. ✅ `test_integration_real_retrieval_with_inmemory_store` - Real vector store integration
2. ✅ `test_integration_handles_none_vector_store` - None vector store handling

## Migration Checklist

- [x] Feature flag added to config schema (removed after migration)
- [x] Feature flag constant exported from config (removed after migration)
- [x] Orchestrator checks feature flag (removed after migration)
- [x] Real retrieval function called when enabled
- [x] Fallback to mock when flag disabled (removed - no fallback)
- [x] Fallback to mock when vector_store is None (removed - returns empty metadata)
- [x] Fallback to mock on real retrieval errors (removed - returns error metadata)
- [x] Migration logging implemented
- [x] API contract preserved
- [x] Unit tests created
- [x] Integration tests created
- [x] **Feature flag removed** (cleanup complete)
- [x] **Mock function removed** (cleanup complete)

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

### Historical: Disable Real Retrieval (No Longer Available)
> **Note**: This section is historical. The feature flag has been removed. All code now uses real vector store retrieval.

```bash
# This no longer works - feature flag removed
# export USE_REAL_VECTOR_RETRIEVAL=false  # ❌ Flag doesn't exist
```

### Monitor Migration via Logs
```bash
# Enable logging to file
export LOG_FILE=vecsec.log
export LOG_LEVEL=DEBUG

# Check migration status
grep "migration" vecsec.log
```

## Rollout Strategy (Historical)

> **Note**: Rollout is complete. This section is kept for historical reference.

1. **Phase 1**: ✅ Deployed with `USE_REAL_VECTOR_RETRIEVAL=true` (default)
2. **Phase 2**: ✅ Rollout successful, no rollback needed
3. **Phase 3**: ✅ **COMPLETE** - Feature flag removed, always use real retrieval

## Rollback Plan (Historical)

> **Note**: Rollback plan is no longer applicable. The feature flag has been removed, so rollback to mock is not possible. This section is kept for historical reference only.

## Files Modified (Historical)

1. `src/sec_agent/config.py`
   - ✅ Added `USE_REAL_VECTOR_RETRIEVAL` to `CONFIG_SCHEMA` (removed in cleanup)
   - ✅ Added `USE_REAL_VECTOR_RETRIEVAL` constant (removed in cleanup)

2. `src/sec_agent/rag_orchestrator.py`
   - ✅ Added logging import
   - ✅ Removed feature flag import (cleanup)
   - ✅ Updated `rag_with_rlsa()` to always use real retrieval
   - ✅ Added migration logging (simplified after cleanup)

3. `src/sec_agent/tests/test_rag_orchestrator.py` (renamed from `test_rag_orchestrator_migration.py`)
   - ✅ Removed feature flag tests (cleanup)
   - ✅ Integration tests for real vector store
   - ✅ API contract verification tests

## Verification (Current)

To verify the system is working:

```bash
# 1. Feature flag no longer exists - all code uses real retrieval
# python3 -c "from src.sec_agent.config import USE_REAL_VECTOR_RETRIEVAL; print(USE_REAL_VECTOR_RETRIEVAL)"
# ❌ This will fail - flag doesn't exist

# 2. Run integration test
python3 scripts/test_chromadb.sh

# 3. Check logs for retrieval events
grep -i "vector retrieval\|vector store" vecsec.log
```

## Summary

✅ **Migration Complete** - All requirements implemented and cleanup finished:
- Vector store connection initialized ✅
- Real retrieval function called ✅
- Feature flag for gradual rollout ✅ (removed after successful rollout)
- Backward compatibility maintained ✅ (removed after successful rollout)
- Migration logging implemented ✅
- API contract preserved ✅
- Unit and integration tests created ✅
- **Feature flag removed** ✅ (cleanup complete)
- **Mock function removed** ✅ (cleanup complete)

The migration is **complete** and **production-ready**. All code uses real vector store retrieval with no fallback to mock.

