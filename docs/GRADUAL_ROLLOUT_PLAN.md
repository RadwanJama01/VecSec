# Gradual Rollout Plan: Real Vector Retrieval

## Overview

This document outlines the gradual rollout plan for enabling real vector retrieval in VecSec production environment. The rollout uses feature flags to enable safe, controlled deployment with quick rollback capability.

---

## Rollout Phases

### Phase 0: Pre-Deployment (Staging)
**Duration:** 1-2 days  
**Feature Flag:** `USE_REAL_VECTOR_RETRIEVAL=false`

**Objectives:**
- Deploy code to staging with feature flag OFF
- Verify system works with mock retrieval
- Run E2E test suite
- Validate monitoring dashboards

**Activities:**
1. Deploy latest code to staging
2. Verify feature flag is disabled
3. Run E2E tests: `pytest tests/test_e2e_validation.py -v`
4. Verify monitoring dashboards show mock retrieval metrics
5. Monitor for 24 hours

**Success Criteria:**
- ✅ All E2E tests pass
- ✅ No errors in logs
- ✅ Monitoring dashboards operational
- ✅ Performance within acceptable range

**Rollback Plan:**
- Feature flag already disabled, no action needed

---

### Phase 1: Internal Test Tenant
**Duration:** 2-3 days  
**Feature Flag:** `USE_REAL_VECTOR_RETRIEVAL=true` (internal tenant only)

**Objectives:**
- Enable real retrieval for internal test tenant only
- Validate functionality with real data
- Monitor for issues
- Gather performance metrics

**Activities:**
1. Enable feature flag for internal test tenant
   ```bash
   # Set tenant-specific feature flag
   export USE_REAL_VECTOR_RETRIEVAL=true
   export TEST_TENANT_ID=internal_test_tenant
   ```
2. Run E2E test suite with real retrieval
3. Monitor dashboards for real retrieval metrics
4. Check for errors or performance issues
5. Monitor for 24-48 hours

**Success Criteria:**
- ✅ Real retrieval working for test tenant
- ✅ No tenant data leakage
- ✅ Performance within SLA (< 2s average, < 3s P95)
- ✅ No critical errors
- ✅ Monitoring shows real retrievals

**Rollback Plan:**
```bash
# Disable feature flag
export USE_REAL_VECTOR_RETRIEVAL=false
systemctl restart vecsec
```

**Monitoring:**
- Check Grafana dashboard: "Real Vector Retrievals Over Time"
- Monitor alert: `VectorRetrievalFailure`
- Review logs for errors
- Track performance metrics

---

### Phase 2: 10% Traffic
**Duration:** 3-5 days  
**Feature Flag:** `USE_REAL_VECTOR_RETRIEVAL=true` (10% of tenants)

**Objectives:**
- Enable real retrieval for 10% of tenants
- Validate at scale
- Monitor performance and errors
- Gather production metrics

**Activities:**
1. Identify 10% of tenants for rollout
2. Enable feature flag for selected tenants
   ```bash
   # Tenant whitelist approach
   export USE_REAL_VECTOR_RETRIEVAL_TENANTS="tenant_a,tenant_b,tenant_c"
   ```
3. Deploy with tenant filtering
4. Monitor dashboards and alerts
5. Run E2E tests periodically
6. Monitor for 3-5 days

**Success Criteria:**
- ✅ Real retrieval working for 10% of tenants
- ✅ Performance within SLA
- ✅ Error rate < 1%
- ✅ No tenant data leakage
- ✅ No customer complaints

**Rollback Plan:**
```bash
# Remove tenants from whitelist or disable flag
export USE_REAL_VECTOR_RETRIEVAL_TENANTS=""
# OR
export USE_REAL_VECTOR_RETRIEVAL=false
systemctl restart vecsec
```

**Monitoring:**
- Track success rate: `rate(vecsec_vector_retrieval_success_total[5m])`
- Monitor error rate: `rate(vecsec_vector_retrieval_failures_total[5m])`
- Check performance: `histogram_quantile(0.95, vecsec_vector_retrieval_duration_seconds_bucket)`
- Review tenant isolation metrics

---

### Phase 3: 50% Traffic
**Duration:** 5-7 days  
**Feature Flag:** `USE_REAL_VECTOR_RETRIEVAL=true` (50% of tenants)

**Objectives:**
- Expand to 50% of tenants
- Validate at larger scale
- Monitor for scaling issues
- Continue performance validation

**Activities:**
1. Expand tenant whitelist to 50%
2. Deploy update
3. Monitor system resources (CPU, memory, disk)
4. Check for performance degradation
5. Monitor for 5-7 days

**Success Criteria:**
- ✅ Real retrieval working for 50% of tenants
- ✅ Performance stable (no degradation)
- ✅ Resource usage acceptable
- ✅ Error rate remains < 1%
- ✅ No scaling issues

**Rollback Plan:**
```bash
# Reduce to 10% or disable
export USE_REAL_VECTOR_RETRIEVAL_TENANTS="tenant_a,tenant_b"  # Back to 10%
# OR
export USE_REAL_VECTOR_RETRIEVAL=false
systemctl restart vecsec
```

**Monitoring:**
- System resource usage
- Vector store performance
- Error rates by tenant
- Performance trends

---

### Phase 4: 100% Traffic
**Duration:** Ongoing  
**Feature Flag:** `USE_REAL_VECTOR_RETRIEVAL=true` (all tenants)

**Objectives:**
- Enable real retrieval for all tenants
- Full production deployment
- Continuous monitoring
- Performance optimization

**Activities:**
1. Enable feature flag for all tenants
   ```bash
   export USE_REAL_VECTOR_RETRIEVAL=true
   # Remove tenant whitelist restriction
   ```
2. Deploy to production
3. Monitor continuously
4. Optimize performance as needed
5. Document lessons learned

**Success Criteria:**
- ✅ Real retrieval working for all tenants
- ✅ Performance within SLA
- ✅ Stable error rates
- ✅ No critical issues
- ✅ Customer satisfaction maintained

**Rollback Plan:**
```bash
# Full rollback if needed
export USE_REAL_VECTOR_RETRIEVAL=false
systemctl restart vecsec

# Or partial rollback for specific tenants
export USE_REAL_VECTOR_RETRIEVAL_TENANTS="tenant_a,tenant_b"  # Exclude problematic tenants
```

**Monitoring:**
- Continuous monitoring of all metrics
- Weekly performance reports
- Monthly review meetings

---

## Rollback Procedures

### Quick Rollback (< 5 minutes)

**Scenario:** Critical issue detected (data leakage, high error rate, performance degradation)

**Steps:**
1. Disable feature flag immediately
   ```bash
   export USE_REAL_VECTOR_RETRIEVAL=false
   systemctl restart vecsec
   ```
2. Verify rollback
   ```bash
   grep "Feature flag disabled" /path/to/logs/*.log
   ```
3. Notify team
4. Investigate issue
5. Fix and re-test before re-enabling

### Partial Rollback

**Scenario:** Issue affecting specific tenants

**Steps:**
1. Remove affected tenants from whitelist
   ```bash
   export USE_REAL_VECTOR_RETRIEVAL_TENANTS="tenant_a,tenant_b"  # Exclude problematic tenants
   systemctl restart vecsec
   ```
2. Monitor remaining tenants
3. Fix issue for excluded tenants
4. Re-add tenants after validation

### Gradual Rollback

**Scenario:** Performance issues requiring gradual reduction

**Steps:**
1. Reduce percentage (e.g., 50% → 25% → 10%)
2. Monitor at each level
3. Identify root cause
4. Fix and gradually re-enable

---

## Monitoring & Validation

### Daily Checks (During Rollout)

1. **Dashboard Review**
   - Check Grafana: "Real Vector Retrievals Over Time"
   - Verify success rate > 99%
   - Check error rates

2. **Alert Review**
   - `VectorRetrievalFailure`: Should not fire
   - `VectorRetrievalTimeout`: Should not fire
   - `VectorStoreDown`: Should not fire

3. **Performance Metrics**
   - Average response time < 2s
   - P95 response time < 3s
   - Error rate < 1%

4. **Log Review**
   - Check for errors: `grep -i error /path/to/logs/*.log | tail -20`
   - Check for fallbacks: `grep "falling back to mock" /path/to/logs/*.log`

### Weekly Validation

1. **E2E Test Suite**
   ```bash
   pytest tests/test_e2e_validation.py -v
   ```

2. **Performance Report**
   - Generate performance report (see `PERFORMANCE_REPORT_TEMPLATE.md`)
   - Review SLA compliance
   - Identify optimization opportunities

3. **Security Audit**
   - Verify tenant isolation
   - Check for data leakage
   - Review access logs

---

## Success Metrics

### Performance Metrics
- ✅ Average response time < 2.0s
- ✅ P95 response time < 3.0s
- ✅ Error rate < 1%
- ✅ Vector retrieval success rate > 99%

### Reliability Metrics
- ✅ Uptime > 99.9%
- ✅ No critical outages
- ✅ MTTR < 5 minutes

### Security Metrics
- ✅ Zero tenant data leakage incidents
- ✅ All RLS violations blocked
- ✅ 100% tenant isolation

### Business Metrics
- ✅ No customer complaints
- ✅ No service degradation
- ✅ Improved search relevance (if applicable)

---

## Communication Plan

### Stakeholder Updates

**Daily (During Active Rollout):**
- Status update to team
- Metrics summary
- Any issues or concerns

**Weekly:**
- Progress report
- Performance summary
- Next steps

**Phase Completion:**
- Phase summary
- Lessons learned
- Go/No-go decision for next phase

### Escalation

**Level 1:** On-call engineer (all issues)  
**Level 2:** Team lead (critical issues > 15 min)  
**Level 3:** Management (extended outages, data leakage)

---

## Risk Mitigation

### Identified Risks

1. **Performance Degradation**
   - Risk: Response times exceed SLA
   - Mitigation: Monitor closely, rollback if needed
   - Contingency: Optimize or scale resources

2. **Data Leakage**
   - Risk: Tenant data isolation failure
   - Mitigation: Comprehensive testing, gradual rollout
   - Contingency: Immediate rollback, security review

3. **Vector Store Failure**
   - Risk: ChromaDB connection issues
   - Mitigation: Health checks, fallback to mock
   - Contingency: Automatic fallback, manual rollback

4. **Scaling Issues**
   - Risk: System cannot handle load
   - Mitigation: Gradual rollout, resource monitoring
   - Contingency: Scale resources, reduce percentage

---

## Timeline

| Phase | Duration | Start Date | End Date | Status |
|-------|----------|------------|----------|--------|
| Phase 0: Staging | 1-2 days | [Date] | [Date] | ⏳ Pending |
| Phase 1: Internal Test | 2-3 days | [Date] | [Date] | ⏳ Pending |
| Phase 2: 10% Traffic | 3-5 days | [Date] | [Date] | ⏳ Pending |
| Phase 3: 50% Traffic | 5-7 days | [Date] | [Date] | ⏳ Pending |
| Phase 4: 100% Traffic | Ongoing | [Date] | - | ⏳ Pending |

**Total Estimated Duration:** 11-17 days (excluding Phase 4)

---

## Sign-off

**Plan Created By:** [Name]  
**Reviewed By:** [Name]  
**Approved By:** [Name]  
**Date:** [YYYY-MM-DD]

---

## Appendix

### Feature Flag Configuration

**Environment Variables:**
```bash
# Enable/disable real retrieval
USE_REAL_VECTOR_RETRIEVAL=true|false

# Tenant whitelist (optional, for gradual rollout)
USE_REAL_VECTOR_RETRIEVAL_TENANTS="tenant_a,tenant_b"

# Vector store configuration
USE_CHROMA=true
CHROMA_PATH=./chroma_db
```

### Monitoring Queries

```promql
# Real retrieval success rate
rate(vecsec_vector_retrieval_success_total[5m]) / (rate(vecsec_vector_retrieval_success_total[5m]) + rate(vecsec_vector_retrieval_failures_total[5m]))

# Vector retrieval latency (p95)
histogram_quantile(0.95, vecsec_vector_retrieval_duration_seconds_bucket)

# Error rate
rate(vecsec_vector_retrieval_failures_total[5m])
```

### Test Commands

```bash
# Run E2E tests
pytest tests/test_e2e_validation.py -v

# Check feature flag
python3 -c "from src.sec_agent.config import USE_REAL_VECTOR_RETRIEVAL; print(USE_REAL_VECTOR_RETRIEVAL)"

# Test vector store connection
python3 scripts/test_vector_store_connection.py
```

