# VecSec Monitoring System

## Overview
This setup provides **Prometheus + Grafana** monitoring for your VecSec security system, including BaseTen API usage, inference metrics, and model training data.

## Quick Start

### 1. Start Monitoring Stack

**Option A: With Docker (Recommended)**
```bash
# Install Docker if needed
# macOS: brew install docker
# Then start services
docker-compose -f docker-compose.monitoring.yml up -d

# Check status
docker-compose -f docker-compose.monitoring.yml ps
```

**Option B: Without Docker (Standalone)**
```bash
# Install Prometheus directly
brew install prometheus grafana

# Or download from official sites:
# Prometheus: https://prometheus.io/download/
# Grafana: https://grafana.com/grafana/download

# Start Prometheus
prometheus --config.file=monitoring/prometheus.yml

# Start Grafana
grafana-server --config=/etc/grafana/grafana.ini

# Or just run metrics exporter
python3 metrics_exporter.py
```

### 2. Access Dashboards
- **Grafana**: http://localhost:3000
  - Username: `admin`
  - Password: `vecsec_admin`

- **Prometheus**: http://localhost:9090

### 3. Import Dashboards
After logging into Grafana:
1. Go to Dashboards → Import
2. Use the dashboard JSON files from `monitoring/dashboards/`

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    VecSec Security System                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Sec_Agent  │  │ Evil_Agent   │  │  Legitimate  │     │
│  │              │  │              │  │    Agent    │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬──────┘     │
│         │                  │                  │            │
│         └──────────────────┼──────────────────┘            │
│                            │                               │
│                            ▼                               │
│                   ┌─────────────────┐                      │
│                   │  Metrics Exporter│                     │
│                   │  (Port 9091)     │                      │
│                   └────────┬────────┘                      │
│                            │                               │
└────────────────────────────┼───────────────────────────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │    Prometheus        │
                   │    (Port 9090)       │
                   └──────────┬───────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │    Grafana          │
                   │    (Port 3000)      │
                   └─────────────────────┘
```

## Metrics Collected

### 1. BaseTen API Metrics
- `vecsec_baseten_api_calls_total` - Total API calls
- `vecsec_baseten_api_calls_failed_total` - Failed API calls
- `vecsec_baseten_api_duration_seconds` - API call duration
- `vecsec_baseten_credits_remaining` - Estimated API credits remaining

### 2. Inference Metrics
- `vecsec_inference_total` - Total inference requests
- `vecsec_inference_duration_seconds` - Inference time
- `vecsec_inference_success_total` - Successful inferences
- `vecsec_inference_errors_total` - Failed inferences

### 3. Security Metrics
- `vecsec_attacks_blocked_total` - Attacks blocked by type
- `vecsec_threats_detected_total` - Threats detected by type
- `vecsec_semantic_similarity_score` - Semantic threat scores
- `vecsec_rlsa_violations_total` - RLSA policy violations

### 4. Training Metrics
- `vecsec_patterns_learned_total` - Threat patterns learned
- `vecsec_embedding_generation_total` - Embeddings generated
- `vecsec_model_updates_total` - Model update events

## Adding Metrics to Your Code

### Example: Tracking BaseTen API Calls

```python
from prometheus_client import Counter, Histogram, Gauge

# Counters
baseten_api_calls = Counter('vecsec_baseten_api_calls_total', 
    'Total BaseTen API calls', ['status'])
baseten_credits = Gauge('vecsec_baseten_credits_remaining', 
    'Estimated BaseTen credits remaining')

# Histograms
inference_duration = Histogram('vecsec_inference_duration_seconds',
    'Time spent on inference', 
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0])

# In your code:
baseten_api_calls.labels(status='success').inc()
baseten_credits.set(estimated_credits)
inference_duration.observe(time_elapsed)
```

### Expose Metrics Endpoint

Add to `Sec_Agent.py`:
```python
from prometheus_client import make_wsgi_app
from wsgiref.simple_server import make_server

# Expose metrics on port 9091
metrics_app = make_wsgi_app()

def metrics_server():
    server = make_server('0.0.0.0', 9091, metrics_app)
    server.serve_forever()

# Run in background thread
import threading
threading.Thread(target=metrics_server, daemon=True).start()
```

## Grafana Dashboards

### 1. BaseTen Usage Dashboard
- API call rate over time
- Credit consumption trends
- Average response time
- Error rate percentage

### 2. Security Metrics Dashboard
- Attacks blocked by type (pie chart)
- Threat detection rate over time
- Semantic similarity scores
- RLSA violations by category

### 3. Performance Dashboard
- Inference latency (p50, p95, p99)
- Throughput (requests/second)
- Error rates
- System health status

### 4. Training Analytics
- Patterns learned over time
- Embedding generation rate
- Model update frequency
- Similarity threshold distribution

## Creating Custom Dashboards

1. Go to Grafana → Dashboards → New Dashboard
2. Add Panels:
   - Query: Use PromQL queries
   - Visualization: Choose charts
3. Example Queries:

```promql
# API Call Rate
rate(vecsec_baseten_api_calls_total[5m])

# Attack Blocking Rate
rate(vecsec_attacks_blocked_total{type="prompt_injection"}[5m])

# Average Inference Time
vecsec_inference_duration_seconds

# Credit Consumption
vecsec_baseten_credits_remaining
```

## Sandbox Environment

### Training and Testing BaseTen Models

```bash
# Set up environment
export BASETEN_MODEL_ID=your_model_id
export BASETEN_API_KEY=your_api_key

# Run tests with monitoring
python3 Good_Vs_Evil.py --test-type blind --blind-tests 100

# Check credit usage in Grafana
# Open: BaseTen Usage Dashboard
```

### Credit Management

Monitor credit usage:
```python
# In Sec_Agent.py, track credits after each API call
from prometheus_client import Gauge

baseten_credits = Gauge('vecsec_baseten_credits_remaining')

# After BaseTen API call
estimated_credits = get_remaining_credits()  # Implement this
baseten_credits.set(estimated_credits)
```

## Data Storage

### Prometheus Retention
By default, Prometheus keeps data for **15 days**. To extend:

```yaml
# In docker-compose.monitoring.yml
prometheus:
  command:
    - '--storage.tsdb.retention.time=90d'  # Keep 90 days
```

### Long-Term Storage (Optional)
For longer retention, configure remote write:

```yaml
remote_write:
  - url: 'https://your-storage-backend/api/v1/write'
```

## Alerting Rules

Create `monitoring/prometheus/vecsec_rules.yml`:

```yaml
groups:
  - name: vecsec_alerts
    interval: 30s
    rules:
      - alert: HighAPIErrorRate
        expr: rate(vecsec_baseten_api_calls_failed_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "BaseTen API error rate is high"

      - alert: LowCreditsRemaining
        expr: vecsec_baseten_credits_remaining < 1000
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "BaseTen credits running low"

      - alert: HighInferenceLatency
        expr: histogram_quantile(0.95, vecsec_inference_duration_seconds) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Inference latency is high"
```

## Testing

### 1. Generate Test Data
```bash
# Run comprehensive tests
python3 Good_Vs_Evil.py --test-type blind --blind-tests 1000

# This will generate metrics in Prometheus
```

### 2. View Metrics
```bash
# Query Prometheus directly
curl http://localhost:9090/api/v1/query?query=vecsec_baseten_api_calls_total
```

### 3. Export Metrics
```bash
# Export metrics to CSV
python3 export_metrics.py --start-time 2025-01-01 --end-time 2025-01-02
```

## Troubleshooting

### Prometheus not scraping
```bash
# Check logs
docker logs vecsec_prometheus

# Verify config
docker exec vecsec_prometheus promtool check config /etc/prometheus/prometheus.yml
```

### Metrics not appearing
1. Check if metrics endpoint is accessible:
   ```bash
   curl http://vecsec_agent:9091/metrics
   ```

2. Verify job config in Prometheus:
   - Targets → vecsec-agent should show UP

### Grafana not showing data
1. Check datasource: Configuration → Data Sources → Test
2. Verify time range in dashboard
3. Check if Prometheus has data for that time range

## Cost Monitoring

### Track BaseTen Credit Usage

```python
# Add to Sec_Agent.py
import requests

def track_credit_usage():
    """Track BaseTen API credit usage"""
    try:
        # Get usage from BaseTen API
        response = requests.get(
            f"https://api.baseten.co/v1/models/{model_id}/usage",
            headers={"Authorization": f"Api-Key {api_key}"}
        )
        usage = response.json()
        
        # Update metrics
        baseten_credits.set(usage.get('credits_remaining', 0))
        baseten_api_calls.labels(status='success').inc()
    except:
        baseten_api_calls.labels(status='failed').inc()
```

## Useful Commands

```bash
# Start everything
docker-compose -f docker-compose.monitoring.yml up -d

# Stop everything
docker-compose -f docker-compose.monitoring.yml down

# View logs
docker logs vecsec_prometheus -f
docker logs vecsec_grafana -f
docker logs vecsec_agent -f

# Restart services
docker-compose -f docker-compose.monitoring.yml restart

# Clean up
docker-compose -f docker-compose.monitoring.yml down -v
```

## Next Steps

1. ✅ Start monitoring stack
2. ✅ Add metrics exporter to Sec_Agent.py
3. ✅ Import Grafana dashboards
4. ✅ Set up alerting rules
5. ✅ Track BaseTen credit usage
6. ✅ Export metrics for analysis

## Dashboard Access
- Grafana: http://localhost:3000 (admin/vecsec_admin)
- Prometheus: http://localhost:9090
- Metrics: http://localhost:9091/metrics

