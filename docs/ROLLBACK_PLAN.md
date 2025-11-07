# Rollback Plan: Real Vector Retrieval Migration

This document provides a step-by-step rollback plan if issues are discovered with the real vector retrieval migration.

## Quick Rollback (Feature Flag)

The fastest way to rollback is to disable the feature flag. This immediately reverts to the mock metadata generator without code changes.

### Step 1: Disable Feature Flag

```bash
# Set environment variable
export USE_REAL_VECTOR_RETRIEVAL=false

# Or in .env file
echo "USE_REAL_VECTOR_RETRIEVAL=false" >> .env
```

### Step 2: Restart Application

Restart your application/service to pick up the new environment variable:

```bash
# If running as a service
sudo systemctl restart vecsec

# If running in Docker
docker-compose restart vecsec

# If running directly
# Kill and restart the process
```

### Step 3: Verify Rollback

Check logs to confirm mock retrieval is being used:

```bash
# Look for this log message
grep "Feature flag disabled, using mock metadata generator" /path/to/logs

# Or run a test query and check logs
python3 src/sec_agent/cli.py "test query" --tenant-id default_tenant
```

**Expected log output (mock retrieval):**
```
DEBUG - Feature flag disabled, using mock metadata generator
```

**NOT this (real retrieval):**
```
INFO - Using real vector store for retrieval metadata
```

### Step 4: Monitor

Monitor for 5-10 minutes to ensure:
- ✅ No errors in logs
- ✅ Queries are completing successfully
- ✅ Response times are acceptable
- ✅ No increase in error rates

## When to Rollback

Rollback immediately if you see:

1. **High Error Rates**
   - `Real vector retrieval failed, falling back to mock` appearing frequently
   - Application errors or crashes
   - 5xx HTTP errors increasing

2. **Performance Degradation**
   - Response times > 2x normal
   - Timeout errors
   - Resource exhaustion (CPU/memory)

3. **Data Issues**
   - Incorrect retrieval results
   - Missing documents
   - Tenant isolation failures

4. **Vector Store Issues**
   - ChromaDB connection failures
   - Database corruption
   - Embedding generation errors

## Rollback Verification Checklist

After rolling back, verify:

- [ ] Feature flag is set to `false`
- [ ] Logs show "Feature flag disabled, using mock metadata generator"
- [ ] No "Using real vector store" messages in logs
- [ ] Application is responding normally
- [ ] Error rates return to baseline
- [ ] Response times are acceptable
- [ ] No new errors in monitoring dashboards

## Full Code Rollback (If Needed)

If the feature flag rollback isn't sufficient, you can rollback the code changes:

### Option 1: Git Revert

```bash
# Find the migration commit
git log --oneline --grep="migration\|real vector"

# Revert the commit
git revert <commit-hash>

# Push to production
git push origin main
```

### Option 2: Manual Code Changes

If you need to manually revert, change `rag_orchestrator.py`:

**Before (with real retrieval):**
```python
if USE_REAL_VECTOR_RETRIEVAL and self.vector_store is not None:
    retrieval_metadata = generate_retrieval_metadata_real(...)
else:
    retrieval_metadata = generate_retrieval_metadata(...)
```

**After (mock only):**
```python
# Always use mock (rollback)
retrieval_metadata = generate_retrieval_metadata(query_context, tenant_id)
```

## Rollback Communication

When rolling back:

1. **Notify Team**
   - Post in team chat/email
   - Update incident tracker
   - Document reason for rollback

2. **Update Status**
   - Mark migration as "rolled back"
   - Update any tracking tickets
   - Note the reason in migration doc

3. **Post-Mortem**
   - Schedule post-mortem meeting
   - Document root cause
   - Plan fix before re-attempting

## Monitoring During Rollback

Watch these metrics:

1. **Error Rates**
   ```bash
   # Check error logs
   tail -f /var/log/vecsec/error.log | grep -i error
   ```

2. **Response Times**
   - Check Grafana/Prometheus dashboards
   - Look for p50, p95, p99 latency
   - Should return to baseline after rollback

3. **Application Health**
   - Health check endpoints
   - CPU/Memory usage
   - Request success rate

## Testing After Rollback

Run these tests to verify rollback worked:

```bash
# 1. Run migration tests (should show mock being used)
python3 src/sec_agent/tests/test_rag_orchestrator_migration.py

# 2. Check feature flag
python3 -c "from src.sec_agent.config import USE_REAL_VECTOR_RETRIEVAL; print(f'Flag: {USE_REAL_VECTOR_RETRIEVAL}')"
# Should output: Flag: False

# 3. Run integration test
python3 src/sec_agent/cli.py "test query" --tenant-id default_tenant
# Check logs for "Feature flag disabled"
```

## Re-Enabling After Fix

Once issues are fixed:

1. **Test in Staging**
   ```bash
   export USE_REAL_VECTOR_RETRIEVAL=true
   # Run full test suite
   ```

2. **Gradual Rollout**
   - Enable for 10% of traffic
   - Monitor for 1 hour
   - Increase to 50%
   - Monitor for 1 hour
   - Enable for 100%

3. **Verify Success**
   - Check logs show real retrieval
   - Monitor error rates
   - Verify performance metrics

## Emergency Contacts

- **On-Call Engineer**: [Contact Info]
- **Team Lead**: [Contact Info]
- **DevOps**: [Contact Info]

## Rollback Decision Tree

```
Issue Detected?
  ├─ Yes → Check Severity
  │   ├─ Critical (Errors/Crashes) → IMMEDIATE ROLLBACK
  │   ├─ High (Performance) → Rollback if > 2x slower
  │   └─ Low (Minor issues) → Monitor, fix without rollback
  └─ No → Continue monitoring
```

## Quick Reference

**Rollback Command:**
```bash
export USE_REAL_VECTOR_RETRIEVAL=false && systemctl restart vecsec
```

**Verify Rollback:**
```bash
grep "Feature flag disabled" /var/log/vecsec/app.log
```

**Re-enable:**
```bash
export USE_REAL_VECTOR_RETRIEVAL=true && systemctl restart vecsec
```

## Related Documents

- [Migration Documentation](./MIGRATION_REAL_VECTOR_RETRIEVAL.md)
- [Migration Evidence](./MIGRATION_EVIDENCE.md)
- [Vector Store Configuration](./VECTOR_STORE_CONFIG.md)
- [Monitoring Guide](./MONITORING.md)

