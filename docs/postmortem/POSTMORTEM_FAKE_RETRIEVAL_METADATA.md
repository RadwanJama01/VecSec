# Postmortem: Fake Retrieval Metadata in generate_retrieval_metadata()

**Date:** 2025-11-13  
**Resolution Date:** 2025-11-13

---

## Summary

Retrieval pipeline produced synthetic metadata instead of real vector DB results, breaking RLS and semantic relevance across the system.

## Impact

All retrieval-augmented features were using fake scores and cross-tenant docs; invalidates audits and similarity checks; undermines trust.

## Root Cause

`generate_retrieval_metadata()` used hardcoded IDs and fabricated scores; no `vector_store.similarity_search()` call.

## Timeline

- **Issue first noticed:** 2025-11-07
- **Root cause confirmed:** 2025-11-08
- **Migration work:** 2025-11-09 through 2025-11-11
- **Final cleanup + resolution:** 2025-11-13

## Detection Failure Analysis

[To be filled in]

## Fix Applied

Integrated real vector store, replaced mock logic, added top-k parameter, enforced tenant scoping.

## Preventative Actions

Unit tests for real similarity calls, remove all mock code from production path, add staging alert for "zero real retrievals".

## Takeaways

[To be filled in]

