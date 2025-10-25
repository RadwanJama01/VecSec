# Flask Proxy Server

A flexible HTTP proxy server built with Flask that can forward requests to any target URL while providing security controls and logging.

## Features

- **Universal Proxy**: Forward any HTTP method (GET, POST, PUT, DELETE, etc.) to any target URL
- **Security Controls**: URL validation, host allowlisting, and path blocking
- **Request/Response Forwarding**: Preserves headers and request body
- **Error Handling**: Comprehensive error handling with appropriate HTTP status codes
- **Logging**: Request logging for monitoring and debugging
- **Health Check**: Built-in health check endpoint
- **Configuration**: Runtime configuration viewing

## Installation

### Using uv (Recommended)

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
   pip install flask requests werkzeug
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

## Configuration

You can modify the following variables in `app.py`:

- `DEFAULT_TARGET_URL`: Default target when no target is specified
- `ALLOWED_HOSTS`: List of allowed hostnames (empty = allow all)
- `BLOCKED_PATHS`: List of paths to block

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
