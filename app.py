from flask import Flask, request, Response, jsonify
import requests
import logging
from urllib.parse import urljoin, urlparse
import json

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DEFAULT_TARGET_URL = "https://httpbin.org"  # Default target for testing
ALLOWED_HOSTS = []  # Empty list means allow all hosts
BLOCKED_PATHS = []  # Paths to block

def is_url_safe(url):
    """Check if the URL is safe to proxy to"""
    try:
        parsed = urlparse(url)
        # Only allow HTTP and HTTPS
        if parsed.scheme not in ['http', 'https']:
            return False
        
        # Check if host is in allowed hosts (if specified)
        if ALLOWED_HOSTS and parsed.hostname not in ALLOWED_HOSTS:
            return False
            
        return True
    except Exception:
        return False

def should_block_path(path):
    """Check if the path should be blocked"""
    for blocked in BLOCKED_PATHS:
        if path.startswith(blocked):
            return True
    return False

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS'])
def proxy(path):
    """Main proxy endpoint that forwards requests to target URL"""
    
    # Get target URL from query parameter or use default
    target_url = request.args.get('target', DEFAULT_TARGET_URL)
    
    # Construct full target URL
    if path:
        target_url = urljoin(target_url.rstrip('/') + '/', path)
    else:
        target_url = target_url.rstrip('/')
    
    # Add query parameters (excluding 'target')
    query_params = dict(request.args)
    query_params.pop('target', None)
    
    if query_params:
        separator = '&' if '?' in target_url else '?'
        query_string = '&'.join([f"{k}={v}" for k, v in query_params.items()])
        target_url = f"{target_url}{separator}{query_string}"
    
    # Security checks
    if not is_url_safe(target_url):
        return jsonify({'error': 'Unsafe URL'}), 400
    
    if should_block_path(path):
        return jsonify({'error': 'Path blocked'}), 403
    
    try:
        # Prepare headers (exclude host and connection headers)
        headers = dict(request.headers)
        headers_to_exclude = ['host', 'connection', 'content-length']
        for header in headers_to_exclude:
            headers.pop(header, None)
        
        # Get request data
        data = None
        if request.method in ['POST', 'PUT', 'PATCH']:
            if request.is_json:
                data = json.dumps(request.get_json())
                headers['Content-Type'] = 'application/json'
            else:
                data = request.get_data()
        
        # Make the proxied request
        logger.info(f"Proxying {request.method} {request.url} -> {target_url}")
        
        response = requests.request(
            method=request.method,
            url=target_url,
            headers=headers,
            data=data,
            allow_redirects=False,
            timeout=30
        )
        
        # Prepare response headers (exclude hop-by-hop headers)
        response_headers = dict(response.headers)
        hop_by_hop_headers = [
            'connection', 'keep-alive', 'proxy-authenticate',
            'proxy-authorization', 'te', 'trailers', 'transfer-encoding', 'upgrade'
        ]
        for header in hop_by_hop_headers:
            response_headers.pop(header, None)
        
        # Create Flask response
        flask_response = Response(
            response.content,
            status=response.status_code,
            headers=response_headers
        )
        
        return flask_response
        
    except requests.exceptions.Timeout:
        logger.error(f"Timeout while proxying to {target_url}")
        return jsonify({'error': 'Request timeout'}), 504
    except requests.exceptions.ConnectionError:
        logger.error(f"Connection error while proxying to {target_url}")
        return jsonify({'error': 'Connection failed'}), 502
    except Exception as e:
        logger.error(f"Error while proxying to {target_url}: {str(e)}")
        return jsonify({'error': 'Internal proxy error'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'flask-proxy'})

@app.route('/config')
def config():
    """Configuration endpoint"""
    return jsonify({
        'default_target': DEFAULT_TARGET_URL,
        'allowed_hosts': ALLOWED_HOSTS,
        'blocked_paths': BLOCKED_PATHS
    })

if __name__ == '__main__':
    print("Flask Proxy Server")
    print("=================")
    print(f"Default target URL: {DEFAULT_TARGET_URL}")
    print("Usage examples:")
    print("  GET  /?target=https://api.github.com/users/octocat")
    print("  POST /?target=https://httpbin.org/post")
    print("  GET  /health")
    print("  GET  /config")
    print("\nStarting server on http://localhost:8080")
    
    app.run(host='0.0.0.0', port=8080, debug=True)
