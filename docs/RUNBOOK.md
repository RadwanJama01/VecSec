# VecSec Operations Runbook

## Overview
This runbook provides step-by-step procedures for common operational issues with VecSec, particularly related to real vector retrieval.

---

## Table of Contents
1. [Vector Retrieval Failures](#vector-retrieval-failures)
2. [Performance Degradation](#performance-degradation)
3. [Tenant Data Leakage Concerns](#tenant-data-leakage-concerns)
4. [Monitoring Dashboard Issues](#monitoring-dashboard-issues)
5. [Feature Flag Management](#feature-flag-management)
6. [Vector Store Connectivity](#vector-store-connectivity)
7. [Alert Response Procedures](#alert-response-procedures)

---

## Vector Retrieval Failures

### Symptoms
- High error rates in logs: "Real vector retrieval failed, falling back to mock"
- Alert: `VectorRetrievalFailure` firing
- Users experiencing degraded functionality
- Metrics showing `vecsec_vector_retrieval_failures_total` increasing

### Diagnosis Steps

1. **Check Vector Store Status**
   ```bash
   # Check if ChromaDB is running
   docker ps | grep chroma
   
   # Or check connection
   python3 -c "from src.sec_agent.config import initialize_vector_store; vs = initialize_vector_store(); print('Connected' if vs else 'Failed')"
   ```

2. **Check Logs**
   ```bash
   # Look for retrieval errors
   grep -i "vector retrieval failed" /path/to/logs/*.log | tail -20
   
   # Check for connection errors
   grep -i "connection\|timeout\|refused" /path/to/logs/*.log | tail -20
   ```

3. **Check Metrics**
   ```bash
   # Query Prometheus
   curl 'http://localhost:9090/api/v1/query?query=rate(vecsec_vector_retrieval_failures_total[5m])'
   ```

### Resolution Steps

**Option 1: Quick Rollback (Recommended for Critical Issues)**
```bash
# Disable real retrieval, use mock
export USE_REAL_VECTOR_RETRIEVAL=false

# Restart service
systemctl restart vecsec
# OR
docker-compose restart vecsec-agent
```

**Option 2: Fix Vector Store Connection**
```bash
# Check ChromaDB path
echo $CHROMA_PATH

# Verify ChromaDB files exist
ls -la $CHROMA_PATH

# Restart ChromaDB if using Docker
docker-compose restart chromadb

# Test connection
python3 scripts/test_vector_store_connection.py
```

**Option 3: Check Resource Limits**
```bash
# Check disk space
df -h $CHROMA_PATH

# Check memory
free -h

# Check ChromaDB process
ps aux | grep chroma
```

### Verification
```bash
# Verify feature flag
python3 -c "from src.sec_agent.config import USE_REAL_VECTOR_RETRIEVAL; print(f'Real retrieval: {USE_REAL_VECTOR_RETRIEVAL}')"

# Check logs for successful retrieval
grep "Using real vector store" /path/to/logs/*.log | tail -5

# Verify metrics improving
curl 'http://localhost:9090/api/v1/query?query=rate(vecsec_vector_retrieval_success_total[5m])'
```

---

## Performance Degradation

### Symptoms
- Response times > 5 seconds
- Alert: `VectorRetrievalTimeout` or `HighInferenceLatency` firing
- User complaints about slow queries
- P95 latency exceeding SLA (3s)

### Diagnosis Steps

1. **Check Current Performance**
   ```bash
   # Query Prometheus for latency
   curl 'http://localhost:9090/api/v1/query?query=histogram_quantile(0.95, vecsec_vector_retrieval_duration_seconds_bucket)'
   ```

2. **Check System Resources**
   ```bash
   # CPU usage
   top -p $(pgrep -f vecsec)
   
   # Memory usage
   ps aux | grep vecsec
   
   # Disk I/O
   iostat -x 1 5
   ```

3. **Check Vector Store Performance**
   ```bash
   # ChromaDB query time
   time python3 -c "
   from src.sec_agent.config import initialize_vector_store
   vs = initialize_vector_store()
   results = vs.similarity_search('test query', k=5)
   print(f'Found {len(results)} results')
   "
   ```

### Resolution Steps

**Option 1: Optimize Vector Store**
```bash
# Reindex ChromaDB if needed
# (ChromaDB handles this automatically, but can be forced)
python3 -c "
from src.sec_agent.config import initialize_vector_store
vs = initialize_vector_store()
# Force reindex by adding a document
vs.add_documents([Document(page_content='reindex trigger', metadata={})])
"
```

**Option 2: Scale Resources**
```bash
# Increase ChromaDB memory
export CHROMA_MEMORY_LIMIT=4G

# Or in docker-compose.yml
# chromadb:
#   mem_limit: 4g
```

**Option 3: Enable Caching**
```bash
# Check if caching is enabled
grep -i cache /path/to/config

# Enable response caching if available
```

**Option 4: Temporary Rollback**
```bash
# If performance is critical, temporarily use mock
export USE_REAL_VECTOR_RETRIEVAL=false
systemctl restart vecsec
```

### Verification
```bash
# Monitor performance metrics
watch -n 5 'curl -s "http://localhost:9090/api/v1/query?query=histogram_quantile(0.95, vecsec_vector_retrieval_duration_seconds_bucket)" | jq .data.result[0].value[1]'
```

---

## Tenant Data Leakage Concerns

### Symptoms
- User reports seeing data from another tenant
- Security audit flags potential leakage
- Logs show cross-tenant access attempts

### Diagnosis Steps

1. **Check Tenant Isolation Logs**
   ```bash
   # Look for cross-tenant violations
   grep "cross_tenant_violation" /path/to/logs/*.log | tail -20
   ```

2. **Run Tenant Isolation Tests**
   ```bash
   # Run E2E tenant isolation test
   pytest tests/test_e2e_validation.py::TestE2EValidation::test_e2e_tenant_isolation -v
   
   # Run unit tests
   pytest src/sec_agent/tests/test_metadata_generator_real.py -k tenant -v
   ```

3. **Check Vector Store Filters**
   ```bash
   # Verify tenant filtering in queries
   python3 -c "
   from src.sec_agent.metadata_generator import generate_retrieval_metadata_real
   from src.sec_agent.config import initialize_vector_store
   
   vs = initialize_vector_store()
   results = generate_retrieval_metadata_real(
       {'query': 'test'},
       'tenant_a',
       vs
   )
   # Verify all results have tenant_id='tenant_a'
   for r in results:
       assert r.get('tenant_id') == 'tenant_a', f'Leakage detected: {r}'
   print('Tenant isolation verified')
   "
   ```

### Resolution Steps

**Immediate Action:**
```bash
# If leakage confirmed, disable real retrieval immediately
export USE_REAL_VECTOR_RETRIEVAL=false
systemctl restart vecsec

# Notify security team
# Review access logs
```

**Investigation:**
1. Review recent code changes to tenant filtering
2. Check vector store metadata for tenant_id correctness
3. Audit all queries for proper tenant filtering
4. Review RLS enforcer logic

**Fix:**
1. Correct tenant filtering in metadata generator
2. Add additional validation in RLS enforcer
3. Re-run all tenant isolation tests
4. Gradually re-enable with monitoring

### Verification
```bash
# Run comprehensive tenant isolation tests
pytest tests/test_e2e_validation.py::TestE2EValidation::test_e2e_no_tenant_data_leakage -v

# Check metrics for violations
curl 'http://localhost:9090/api/v1/query?query=vecsec_rlsa_violations_total{violation_type="cross_tenant_violation"}'
```

---

## Monitoring Dashboard Issues

### Symptoms
- Grafana dashboard not showing data
- Metrics not appearing
- Prometheus not scraping

### Diagnosis Steps

1. **Check Prometheus Status**
   ```bash
   # Check if Prometheus is running
   docker ps | grep prometheus
   # OR
   systemctl status prometheus
   
   # Check Prometheus targets
   curl http://localhost:9090/api/v1/targets
   ```

2. **Check Metrics Endpoint**
   ```bash
   # Check if metrics are being exported
   curl http://localhost:9091/metrics | head -20
   ```

3. **Check Grafana Status**
   ```bash
   # Check if Grafana is running
   docker ps | grep grafana
   # OR
   systemctl status grafana
   
   # Check Grafana datasource
   curl -u admin:vecsec_admin http://localhost:3000/api/datasources
   ```

### Resolution Steps

**Prometheus Not Scraping:**
```bash
# Restart Prometheus
docker-compose restart prometheus
# OR
systemctl restart prometheus

# Reload Prometheus config
curl -X POST http://localhost:9090/-/reload
```

**Metrics Not Exported:**
```bash
# Check if metrics exporter is running
ps aux | grep metrics_exporter

# Start metrics exporter
python3 src/metrics_exporter.py &

# Or ensure it's started in application
```

**Grafana Dashboard Missing:**
```bash
# Import dashboard
# 1. Go to http://localhost:3000
# 2. Dashboards → Import
# 3. Upload monitoring/grafana/provisioning/dashboards/dashboard.json

# Or restart Grafana to auto-provision
docker-compose restart grafana
```

### Verification
```bash
# Check Prometheus targets are UP
curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | select(.health=="up")'

# Check metrics are available
curl -s http://localhost:9091/metrics | grep vecsec_vector_retrieval
```

---

## Feature Flag Management

### Enable Real Retrieval
```bash
# Set environment variable
export USE_REAL_VECTOR_RETRIEVAL=true

# Or in .env file
echo "USE_REAL_VECTOR_RETRIEVAL=true" >> .env

# Restart service
systemctl restart vecsec
```

### Disable Real Retrieval (Rollback)
```bash
# Set environment variable
export USE_REAL_VECTOR_RETRIEVAL=false

# Or in .env file
sed -i 's/USE_REAL_VECTOR_RETRIEVAL=true/USE_REAL_VECTOR_RETRIEVAL=false/' .env

# Restart service
systemctl restart vecsec
```

### Check Current Status
```bash
# Check feature flag value
python3 -c "from src.sec_agent.config import USE_REAL_VECTOR_RETRIEVAL; print(f'Real retrieval: {USE_REAL_VECTOR_RETRIEVAL}')"

# Check logs
grep "Feature flag" /path/to/logs/*.log | tail -5
```

---

## Vector Store Connectivity

### Symptoms
- Connection timeouts
- "Vector store connection lost" errors
- Alert: `VectorStoreDown` firing

### Diagnosis Steps

1. **Test Connection**
   ```bash
   python3 scripts/test_vector_store_connection.py
   ```

2. **Check ChromaDB Path**
   ```bash
   echo $CHROMA_PATH
   ls -la $CHROMA_PATH
   ```

3. **Check Network Connectivity**
   ```bash
   # If using remote ChromaDB
   ping <chromadb-host>
   telnet <chromadb-host> <port>
   ```

### Resolution Steps

**Local ChromaDB:**
```bash
# Check if path exists and is writable
mkdir -p $CHROMA_PATH
chmod 755 $CHROMA_PATH

# Reinitialize if corrupted
rm -rf $CHROMA_PATH/*
python3 -c "from src.sec_agent.config import initialize_vector_store, initialize_sample_documents; vs = initialize_vector_store(); initialize_sample_documents(vs)"
```

**Remote ChromaDB:**
```bash
# Check connection string
echo $CHROMA_CONNECTION_STRING

# Test connection
python3 -c "import chromadb; client = chromadb.Client(); print('Connected')"
```

---

## Alert Response Procedures

### VectorRetrievalFailure Alert

**Severity:** Critical  
**Action Required:** Immediate

1. Check vector store connectivity (see [Vector Store Connectivity](#vector-store-connectivity))
2. Review error logs
3. If unable to fix quickly, rollback to mock retrieval
4. Notify team of issue

### VectorRetrievalTimeout Alert

**Severity:** Warning  
**Action Required:** Within 1 hour

1. Check system performance (see [Performance Degradation](#performance-degradation))
2. Review recent changes
3. Consider scaling resources
4. Monitor for escalation

### VectorStoreDown Alert

**Severity:** Critical  
**Action Required:** Immediate

1. Check vector store service status
2. Restart if needed
3. Verify data integrity
4. Rollback if service cannot be restored quickly

---

## Emergency Contacts

- **On-Call Engineer:** [Contact Info]
- **Security Team:** [Contact Info]
- **DevOps Team:** [Contact Info]

## Escalation Path

1. **Level 1:** On-call engineer (all alerts)
2. **Level 2:** Team lead (critical alerts > 15 min)
3. **Level 3:** Security team (data leakage concerns)
4. **Level 4:** Management (extended outages)

---

## Quick Reference Commands

```bash
# Check system status
systemctl status vecsec

# View logs
tail -f /path/to/logs/vecsec.log

# Check metrics
curl http://localhost:9091/metrics | grep vecsec_vector_retrieval

# Rollback to mock
export USE_REAL_VECTOR_RETRIEVAL=false && systemctl restart vecsec

# Run E2E tests
pytest tests/test_e2e_validation.py -v

# Check feature flag
python3 -c "from src.sec_agent.config import USE_REAL_VECTOR_RETRIEVAL; print(USE_REAL_VECTOR_RETRIEVAL)"
```

