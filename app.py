from flask import Flask, request, Response, jsonify
import requests
import logging
from urllib.parse import urljoin, urlparse
import json
from malware_bert import MalwareBERTDetector, ThreatLevel

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DEFAULT_TARGET_URL = "https://httpbin.org"  # Default target for testing
ALLOWED_HOSTS = []  # Empty list means allow all hosts
BLOCKED_PATHS = []  # Paths to block

# Initialize Malware-BERT detector
try:
    malware_detector = MalwareBERTDetector()
    logger.info("Malware-BERT detector initialized successfully")
except Exception as e:
    logger.warning(f"Failed to initialize Malware-BERT detector: {e}")
    malware_detector = None

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

def analyze_for_malware(content, content_type="text"):
    """Analyze content for malware using Malware-BERT"""
    if not malware_detector:
        return None
    
    try:
        # Convert content to string if needed
        if isinstance(content, bytes):
            content = content.decode('utf-8', errors='ignore')
        elif not isinstance(content, str):
            content = str(content)
        
        # Analyze the content
        result = malware_detector.detect_malware(content, use_ml=True)
        
        # Log suspicious activity
        if result.threat_level != ThreatLevel.CLEAN:
            logger.warning(f"Malware detected: {result.threat_level.value} - {result.indicators}")
        
        return {
            'threat_level': result.threat_level.value,
            'confidence': result.confidence,
            'risk_score': result.risk_score,
            'indicators': result.indicators,
            'patterns_found': result.patterns_found[:10]  # Limit to first 10 patterns
        }
    except Exception as e:
        logger.error(f"Malware analysis failed: {e}")
        return None

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
            
            # Analyze request data for malware
            if data:
                malware_analysis = analyze_for_malware(data)
                if malware_analysis and malware_analysis['threat_level'] == 'malicious':
                    logger.error(f"Blocking malicious request: {malware_analysis}")
                    return jsonify({
                        'error': 'Request blocked due to malicious content',
                        'malware_analysis': malware_analysis
                    }), 403
        
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
        
        # Analyze response content for malware
        response_analysis = None
        if response.content:
            response_analysis = analyze_for_malware(response.content)
            if response_analysis and response_analysis['threat_level'] == 'malicious':
                logger.error(f"Blocking malicious response: {response_analysis}")
                return jsonify({
                    'error': 'Response blocked due to malicious content',
                    'malware_analysis': response_analysis
                }), 403
        
        # Create Flask response
        flask_response = Response(
            response.content,
            status=response.status_code,
            headers=response_headers
        )
        
        # Add malware analysis to response headers if available
        if response_analysis:
            flask_response.headers['X-Malware-Analysis'] = json.dumps({
                'threat_level': response_analysis['threat_level'],
                'risk_score': response_analysis['risk_score']
            })
        
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
        'blocked_paths': BLOCKED_PATHS,
        'malware_detection_enabled': malware_detector is not None
    })

@app.route('/analyze', methods=['POST'])
def analyze_malware():
    """Analyze text content for malware"""
    if not malware_detector:
        return jsonify({'error': 'Malware detection not available'}), 503
    
    try:
        data = request.get_json()
        if not data or 'content' not in data:
            return jsonify({'error': 'Content field required'}), 400
        
        content = data['content']
        use_ml = data.get('use_ml', True)
        
        result = malware_detector.detect_malware(content, use_ml=use_ml)
        
        return jsonify({
            'threat_level': result.threat_level.value,
            'confidence': result.confidence,
            'risk_score': result.risk_score,
            'indicators': result.indicators,
            'patterns_found': result.patterns_found,
            'analysis_method': 'ml' if use_ml else 'pattern'
        })
        
    except Exception as e:
        logger.error(f"Malware analysis failed: {e}")
        return jsonify({'error': 'Analysis failed'}), 500

@app.route('/scan', methods=['POST'])
def scan_file():
    """Scan file content for malware"""
    if not malware_detector:
        return jsonify({'error': 'Malware detection not available'}), 503
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        content = file.read()
        result = malware_detector.detect_malware(content.decode('utf-8', errors='ignore'))
        
        return jsonify({
            'filename': file.filename,
            'threat_level': result.threat_level.value,
            'confidence': result.confidence,
            'risk_score': result.risk_score,
            'indicators': result.indicators,
            'patterns_found': result.patterns_found
        })
        
    except Exception as e:
        logger.error(f"File scan failed: {e}")
        return jsonify({'error': 'File scan failed'}), 500

@app.route('/patterns')
def get_patterns():
    """Get available detection patterns"""
    if not malware_detector:
        return jsonify({'error': 'Malware detection not available'}), 503
    
    patterns = {
        'shell_commands': list(malware_detector.pattern_detector.shell_commands.keys()),
        'encoded_patterns': list(malware_detector.pattern_detector.encoded_patterns.keys()),
        'suspicious_urls': malware_detector.pattern_detector.suspicious_urls,
        'script_patterns': len(malware_detector.pattern_detector.script_patterns)
    }
    
    return jsonify(patterns)

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
