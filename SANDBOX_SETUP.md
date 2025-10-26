# VecSec Sandbox Environment Setup

## Overview
Complete sandbox environment for training/testing BaseTen models with Prometheus + Grafana monitoring.

## Quick Start

### 1. Setup Environment Variables
```bash
# Create .env file
cat > .env << EOF
# BaseTen Configuration
BASETEN_MODEL_ID=your_model_id_here
BASETEN_API_KEY=your_api_key_here

# Optional API Keys
GOOGLE_API_KEY=your_google_key
OPENAI_API_KEY=your_openai_key
EOF
```

### 2. Start Monitoring (Optional)
```bash
# Option A: Use Docker (requires Docker installed)
docker-compose -f docker-compose.monitoring.yml up -d

# Option B: Use standalone services
# Install: brew install prometheus grafana
# Then start them separately

# Option C: Just use metrics exporter (simplest)
python3 metrics_exporter.py &
```

### 3. Run Tests and Generate Data
```bash
# Run comprehensive tests
python3 Good_Vs_Evil.py --test-type blind --blind-tests 100

# This will:
# - Generate test data
# - Track BaseTen API usage
# - Learn attack patterns
# - Export metrics

# View metrics in Prometheus
# Open: http://localhost:9090

# View dashboards in Grafana
# Open: http://localhost:3000 (admin/vecsec_admin)
```

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│              VecSec Sandbox Environment                  │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  ┌─────────────────────────────────────────────────┐    │
│  │         Training & Testing Layer                │    │
│  │                                                   │    │
│  │  ┌──────────────┐  ┌──────────────┐           │    │
│  │  │ Evil_Agent   │  │ Legitimate   │           │    │
│  │  │ (Attack Gen) │  │ Agent        │           │    │
│  │  └──────┬───────┘  └──────┬───────┘           │    │
│  │         │                   │                  │    │
│  │         └─────┬─────────────┘                  │    │
│  │               ▼                                 │    │
│  │         ┌──────────────────┐                   │    │
│  │         │  Good_Vs_Evil    │                   │    │
│  │         │  (Test Runner)   │                   │    │
│  │         └──────┬───────────┘                   │    │
│  └─────────────────┼──────────────────────────────│    │
│                    │                                │    │
├────────────────────┼────────────────────────────────┤    │
│                    │                                │    │
│  ┌─────────────────▼──────────────────────────────┐│    │
│  │         BaseTen Qwen3 Inference Layer          ││    │
│  │                                                  ││    │
│  │  ┌──────────────────────────────────────────┐  ││    │
│  │  │   Sec_Agent.py                          │  ││    │
│  │  │  - Semantic Threat Detection            │  ││    │
│  │  │  - Pattern-Based Detection              │  ││    │
│  │  │  - RLSA Enforcement                     │  ││    │
│  │  │  - Learning from Attacks                │  ││    │
│  │  └────┬────────────────────────────┬───────┘  ││    │
│  │       │                            │          ││    │
│  └───────┼────────────────────────────┼──────────┘│    │
│          │                            │          │    │
└──────────┼────────────────────────────┼──────────────┘    │
           │                            │                    │
           ▼                            ▼                    │
┌───────────────────┐      ┌──────────────────────┐         │
│  BaseTen API      │      │  Metrics Exporter    │         │
│  (Qwen3 8B)       │      │  (Port 9091)         │         │
└───────────────────┘      └────────────┬─────────┘         │
                                        │                   │
                                        ▼                   │
                              ┌──────────────────┐          │
                              │   Prometheus     │          │
                              │   (Port 9090)    │          │
                              └────────┬─────────┘          │
                                        │                   │
                                        ▼                   │
                              ┌──────────────────┐          │
                              │    Grafana       │          │
                              │    (Port 3000)   │          │
                              └──────────────────┘          │
```

## Components

### 1. Training/Testing Layer
- **Evil_Agent.py**: Generates malicious attack patterns
- **Legitimate_Agent.py**: Generates legitimate queries
- **Good_Vs_Evil.py**: Runs comprehensive blind tests

### 2. Inference Layer (Sec_Agent.py)
- Pattern-based detection
- Semantic threat detection (BaseTen Qwen3)
- RLSA enforcement
- Learning from attacks

### 3. Monitoring Layer
- **Prometheus**: Collects and stores metrics
- **Grafana**: Visualizes metrics
- **Metrics Exporter**: Exports metrics for scraping

## Usage Examples

### Training BaseTen Model

```bash
# 1. Run tests to generate training data
python3 Good_Vs_Evil.py --test-type blind --blind-tests 1000

# 2. Monitor credit usage
# Open Grafana → BaseTen Usage Dashboard

# 3. Check learned patterns
python3 -c "
from Sec_Agent import threat_embedder
print(f'Learned {len(threat_embedder.learned_patterns)} patterns')
"
```

### Tracking API Costs

```python
# Add to Sec_Agent.py
from metrics_exporter import metrics_exporter
import time

def track_baseten_api_call(prompt, model="qwen3-8b"):
    start = time.time()
    
    try:
        # Call BaseTen API
        embedding = qwen_client.get_embedding(prompt)
        duration = time.time() - start
        
        # Track metrics
        metrics_exporter.track_baseten_call("success", model, duration)
        
        # Update credits (you'll need to implement this)
        remaining_credits = estimate_remaining_credits()
        metrics_exporter.set_baseten_credits(remaining_credits)
        
        return embedding
    except Exception as e:
        duration = time.time() - start
        metrics_exporter.track_baseten_call("failed", model, duration)
        raise
```

### Monitoring System Health

```bash
# Check if services are running
curl http://localhost:9090/api/v1/query?query=up{job="vecsec-agent"}

# Check credit usage
curl http://localhost:9090/api/v1/query?query=vecsec_baseten_credits_remaining

# Check attack blocking rate
curl http://localhost:9090/api/v1/query?query=rate(vecsec_attacks_blocked_total[5m])
```

## Monitoring Queries

### PromQL Examples

```promql
# API call rate per minute
rate(vecsec_baseten_api_calls_total[1m])

# Attack blocking rate
rate(vecsec_attacks_blocked_total[5m])

# Average inference time (p50)
histogram_quantile(0.5, vecsec_inference_duration_seconds_bucket)

# Error rate
rate(vecsec_baseten_api_calls_failed_total[5m]) / rate(vecsec_baseten_api_calls_total[5m])

# Credits remaining
vecsec_baseten_credits_remaining

# Patterns learned over last hour
increase(vecsec_patterns_learned_total[1h])
```

## Cost Management

### Track Credits

```python
# In Sec_Agent.py
from metrics_exporter import metrics_exporter

# After each BaseTen API call
cost_estimate = 0.001  # Rough estimate per call
credits_remaining -= cost_estimate
metrics_exporter.set_baseten_credits(max(0, credits_remaining))
```

### Set Alerts

When credits get low, Grafana will alert you based on the rules in `vecsec_rules.yml`.

## Dashboard URLs

- **Grafana**: http://localhost:3000
  - Username: `admin`
  - Password: `vecsec_admin`

- **Prometheus**: http://localhost:9090

- **Metrics Endpoint**: http://localhost:9091/metrics

## Exporting Data

### Export Metrics to CSV

```python
from prometheus_client import CollectorRegistry, Gauge, generate_latest
import csv

def export_metrics(start_time, end_time):
    # Query Prometheus API
    # Export to CSV for analysis
    pass
```

### Backup Learned Patterns

```bash
# Save learned patterns
python3 -c "
from Sec_Agent import threat_embedder
import json

patterns = [{
    'query': p['query'],
    'user_context': p['user_context'],
    'was_blocked': p.get('was_blocked', True),
    'timestamp': p.get('timestamp')
} for p in threat_embedder.learned_patterns]

with open('learned_patterns_backup.json', 'w') as f:
    json.dump(patterns, f, indent=2)

print(f'Saved {len(patterns)} patterns')
"
```

## Troubleshooting

### BaseTen API Issues

```bash
# Check if credentials are set
echo $BASETEN_API_KEY

# Test API connectivity
curl -X POST "https://api.baseten.co/v1/models/${BASETEN_MODEL_ID}/predict" \
  -H "Authorization: Api-Key ${BASETEN_API_KEY}" \
  -d '{"inputs": "test"}'
```

### Metrics Not Appearing

```bash
# Check if metrics endpoint is accessible
curl http://localhost:9091/metrics | grep vecsec

# Check Prometheus targets
open http://localhost:9090/targets
```

### High Latency

```bash
# Check inference times in Prometheus
open http://localhost:9090/graph?g0.expr=histogram_quantile(0.95,%20vecsec_inference_duration_seconds_bucket)
```

## Next Steps

1. ✅ Set up BaseTen account and get API key
2. ✅ Configure .env file
3. ✅ Start monitoring stack
4. ✅ Run tests to generate training data
5. ✅ Monitor credit usage
6. ✅ Set up Grafana dashboards
7. ✅ Configure alerting rules

## Cost Estimates

- BaseTen Qwen3: ~$0.01 per 1000 embeddings
- For 10,000 test queries: ~$100
- With monitoring: Track exactly what you're spending!

## Support

- BaseTen: https://docs.baseten.co
- Prometheus: https://prometheus.io/docs
- Grafana: https://grafana.com/docs

