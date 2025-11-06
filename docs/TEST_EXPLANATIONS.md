# Test Suite Explanations

## Overview
This document explains what each test in `test_metadata_generator_real.py` does and why it's important.

---

## Unit Tests (Mocked Vector Store)

### 1. `test_connects_to_real_vector_store`
**Purpose**: Verifies the function actually calls the vector store's similarity search method.

**What it tests**:
- That `similarity_search_with_score()` is called on the vector store
- That the query text is passed correctly
- That it requests 5 results (k=5)
- That a tenant filter is applied (tenant_id in filter dict)

**Why it matters**: Ensures the function isn't just returning mock data - it's actually using the real vector store.

---

### 2. `test_performs_actual_similarity_search`
**Purpose**: Verifies that similarity scores are returned and preserved.

**What it tests**:
- Results include similarity scores
- Scores are floats
- Scores are in valid range [0, 1]
- Multiple results have different scores

**Why it matters**: Confirms real similarity search is happening, not just returning arbitrary values.

---

### 3. `test_returns_real_document_ids_and_content`
**Purpose**: Ensures the function returns actual document metadata, not placeholder data.

**What it tests**:
- Result structure includes all required fields (document_id, embedding_id, tenant_id, retrieval_score, content)
- Document IDs match what was stored
- Content matches what was stored
- Content is not empty

**Why it matters**: Verifies the function extracts and returns real document information from the vector store.

---

### 4. `test_implements_proper_tenant_filtering`
**Purpose**: Verifies tenant isolation - users only see their tenant's documents.

**What it tests**:
- Tenant filter is passed to vector store query
- Filter contains correct tenant_id
- Returned results have matching tenant_id

**Why it matters**: Critical for security - prevents data leakage between tenants.

---

### 5. `test_handles_errors_gracefully`
**Purpose**: Tests error handling and fallback mechanism.

**What it tests**:
- When vector store raises an exception, function doesn't crash
- Falls back to mock metadata generator
- Returns valid results even on error

**Why it matters**: Ensures system resilience - graceful degradation when vector store is unavailable.

---

### 6. `test_handles_empty_query`
**Purpose**: Tests behavior when query is empty but topics are provided.

**What it tests**:
- Empty query string is handled
- Topics are used to construct search query
- Query becomes "security finance" (topics joined)

**Why it matters**: Handles edge case where user provides topics but no query text.

---

### 7. `test_handles_no_results`
**Purpose**: Tests behavior when vector store returns no matching documents.

**What it tests**:
- Empty result set is handled
- Returns placeholder metadata with "doc-empty" ID
- Returns score of 0.0
- Content includes "No documents found"

**Why it matters**: Ensures function doesn't crash on empty results and provides meaningful feedback.

---

### 8. `test_converts_topics_from_string_to_list`
**Purpose**: Tests metadata conversion - ChromaDB stores topics as comma-separated strings, function converts back to lists.

**What it tests**:
- Topics stored as "security,access_control" (string)
- Topics returned as ["security", "access_control"] (list)
- Conversion handles comma-separated format

**Why it matters**: ChromaDB only accepts scalar metadata (strings), but application expects lists - this tests the conversion.

---

### 9. `test_handles_none_vector_store`
**Purpose**: Tests behavior when vector_store parameter is None.

**What it tests**:
- Function doesn't crash when vector_store is None
- Falls back to mock metadata generator
- Returns valid results

**Why it matters**: Handles edge case where vector store might not be initialized.

---

### 10. `test_real_retrieval_different_queries`
**Purpose**: Verifies that different queries return different documents (not just the same results every time).

**What it tests**:
- Query "security" returns security-related document as top result
- Query "finance" returns finance-related document as top result
- Query "marketing" returns marketing-related document as top result
- Top results differ for different queries
- Results are relevant to query (not random)

**Why it matters**: Ensures search is actually working - different queries should return different, relevant results.

---

### 11. `test_retrieval_scores_realistic`
**Purpose**: Verifies that similarity scores vary realistically based on actual similarity (not just sequential 0.9, 0.8, 0.7...).

**What it tests**:
- Scores vary (not all the same)
- Score differences include large gaps (> 0.1) - not just 0.1 increments
- Scores are in valid range [0, 1]
- Highest score is first (most similar document)
- Scores reflect actual similarity differences (high match: 0.87, medium: 0.72, low: 0.34)

**Why it matters**: Ensures scores are meaningful and reflect real similarity, not just mock sequential values.

---

### 12. `test_real_document_ids`
**Purpose**: Verifies that document IDs are realistic (not mock patterns like "emb-001", "doc-finance-001").

**What it tests**:
- Document IDs don't match simple mock patterns (emb-001, doc-001, etc.)
- IDs are meaningful (not placeholders like "doc-empty")
- IDs have reasonable length (> 5 characters)
- Uses realistic ID formats (UUIDs, timestamps, versioned names)

**Why it matters**: Ensures function works with production-like document IDs, not just test patterns.

---

## Integration Tests (Real Vector Store + Fake Embeddings)

### 13. `test_integration_with_real_vector_store`
**Purpose**: Tests with a real InMemoryVectorStore (not mocked) to verify end-to-end functionality.

**What it tests**:
- Function works with real vector store operations
- Returns results from actual vector store
- Document IDs match what was stored
- Tenant filtering works with real vector store

**Why it matters**: Catches issues that mocks might miss - tests real vector store integration.

---

### 14. `test_integration_tenant_filtering`
**Purpose**: Verifies tenant isolation works with real vector store operations.

**What it tests**:
- Tenant A search returns only tenant_a documents
- Tenant B search returns only tenant_b documents
- No cross-tenant leakage (no intersection between results)

**Why it matters**: Critical security test - ensures tenant isolation works in real scenarios.

---

### 15. `test_integration_topics_conversion`
**Purpose**: Tests topics string-to-list conversion with real vector store.

**What it tests**:
- Topics stored as comma-separated strings in vector store
- Topics returned as lists in results
- Conversion works correctly

**Why it matters**: Verifies metadata conversion works end-to-end with real vector store.

---

## Performance Benchmarks

### 16. `test_response_time_under_100ms`
**Purpose**: Ensures function responds quickly for typical queries.

**What it tests**:
- Response time for typical query is under 100ms
- Performance is acceptable for real-time use

**Why it matters**: Performance requirement - users expect fast responses.

---

### 17. `test_response_time_under_500ms_with_large_result_set`
**Purpose**: Tests performance with larger result sets (50 documents).

**What it tests**:
- Function handles larger result sets efficiently
- Response time stays under 500ms even with many results

**Why it matters**: Ensures performance doesn't degrade with larger datasets.

---

### 18. `test_handles_multiple_concurrent_calls`
**Purpose**: Tests performance under load (multiple sequential calls).

**What it tests**:
- Function can handle multiple calls efficiently
- Average response time per call is under 50ms
- No performance degradation with repeated calls

**Why it matters**: Ensures function performs well under typical usage patterns.

---

## Test Categories Summary

| Category | Test Count | Purpose |
|----------|------------|---------|
| **Connection & Search** | 3 tests | Verify vector store is used correctly |
| **Tenant Filtering** | 2 tests | Security - tenant isolation |
| **Error Handling** | 3 tests | Resilience - graceful degradation |
| **Data Format** | 2 tests | Metadata conversion (topics, IDs) |
| **Query Variation** | 1 test | Different queries return different results |
| **Score Realism** | 1 test | Scores reflect actual similarity |
| **Integration** | 3 tests | Real vector store operations |
| **Performance** | 3 tests | Response time benchmarks |
| **Total** | 18 tests | Complete coverage |

---

## Running Tests

```bash
# Run all tests
python3 src/sec_agent/tests/test_metadata_generator_real.py

# Run specific test class
python3 -m unittest src.sec_agent.tests.test_metadata_generator_real.TestGenerateRetrievalMetadataReal

# Run specific test
python3 -m unittest src.sec_agent.tests.test_metadata_generator_real.TestGenerateRetrievalMetadataReal.test_tenant_isolation
```

