"""
Comprehensive DDoS Protection and Rate Limiting System for Flask Proxy
=====================================================================

This module provides production-ready DDoS protection including:
- Per-IP and per-endpoint rate limiting
- Connection rate limiting and concurrent connection limits
- Request size limits and slowloris attack prevention
- IP allowlist/blocklist management
- Redis-based distributed rate limiting
- Admin API for monitoring and management

Author: VecSec Team
"""

import os
import time
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import ipaddress

from flask import Flask, request, jsonify, Response, g
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import redis
from werkzeug.exceptions import TooManyRequests, RequestEntityTooLarge

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class IPStats:
    """Statistics for an IP address"""
    ip: str
    request_count: int = 0
    connection_count: int = 0
    last_request: float = 0
    first_request: float = 0
    blocked_until: Optional[float] = None
    violation_count: int = 0
    total_bytes: int = 0

@dataclass
class RateLimitConfig:
    """Configuration for rate limiting"""
    # Per-IP limits
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    
    # Connection limits
    max_concurrent_connections: int = 10
    max_connections_per_minute: int = 100
    
    # Request size limits
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    max_header_size: int = 8192  # 8KB
    
    # Timeout limits
    request_timeout: int = 30
    slow_request_threshold: int = 10  # seconds
    
    # Blocking configuration
    block_duration: int = 300  # 5 minutes
    max_violations_before_block: int = 5
    
    # Redis configuration
    redis_url: str = "redis://localhost:6379/0"
    redis_enabled: bool = False

class IPManager:
    """Manages IP allowlist/blocklist and statistics"""
    
    def __init__(self):
        self.allowlist: set = set()
        self.blocklist: set = set()
        self.ip_stats: Dict[str, IPStats] = {}
        self.lock = threading.RLock()
        
        # Load from environment
        self._load_from_env()
    
    def _load_from_env(self):
        """Load allowlist/blocklist from environment variables"""
        # Allowlist
        allowlist_env = os.getenv('DDOS_ALLOWLIST', '')
        if allowlist_env:
            for ip in allowlist_env.split(','):
                ip = ip.strip()
                if self._is_valid_ip(ip):
                    self.allowlist.add(ip)
        
        # Blocklist
        blocklist_env = os.getenv('DDOS_BLOCKLIST', '')
        if blocklist_env:
            for ip in blocklist_env.split(','):
                ip = ip.strip()
                if self._is_valid_ip(ip):
                    self.blocklist.add(ip)
    
    def _is_valid_ip(self, ip: str) -> bool:
        """Validate IP address"""
        try:
            ipaddress.ip_address(ip)
            return True
        except ValueError:
            return False
    
    def is_allowed(self, ip: str) -> bool:
        """Check if IP is allowed (not blocked and not in blocklist)"""
        with self.lock:
            if ip in self.allowlist:
                return True
            if ip in self.blocklist:
                return False
            
            # Check if IP is temporarily blocked
            if ip in self.ip_stats:
                stats = self.ip_stats[ip]
                if stats.blocked_until and time.time() < stats.blocked_until:
                    return False
            
            return True
    
    def block_ip(self, ip: str, duration: int = 300):
        """Block an IP for specified duration"""
        with self.lock:
            if ip not in self.ip_stats:
                self.ip_stats[ip] = IPStats(ip=ip)
            
            self.ip_stats[ip].blocked_until = time.time() + duration
            logger.warning(f"IP {ip} blocked for {duration} seconds")
    
    def unblock_ip(self, ip: str):
        """Unblock an IP"""
        with self.lock:
            if ip in self.ip_stats:
                self.ip_stats[ip].blocked_until = None
                logger.info(f"IP {ip} unblocked")
    
    def add_to_allowlist(self, ip: str):
        """Add IP to permanent allowlist"""
        with self.lock:
            if self._is_valid_ip(ip):
                self.allowlist.add(ip)
                self.blocklist.discard(ip)
                logger.info(f"IP {ip} added to allowlist")
    
    def add_to_blocklist(self, ip: str):
        """Add IP to permanent blocklist"""
        with self.lock:
            if self._is_valid_ip(ip):
                self.blocklist.add(ip)
                self.allowlist.discard(ip)
                logger.info(f"IP {ip} added to blocklist")
    
    def get_stats(self, ip: str) -> Optional[IPStats]:
        """Get statistics for an IP"""
        with self.lock:
            return self.ip_stats.get(ip)
    
    def update_stats(self, ip: str, request_size: int = 0):
        """Update IP statistics"""
        with self.lock:
            current_time = time.time()
            
            if ip not in self.ip_stats:
                self.ip_stats[ip] = IPStats(
                    ip=ip,
                    first_request=current_time
                )
            
            stats = self.ip_stats[ip]
            stats.request_count += 1
            stats.last_request = current_time
            stats.total_bytes += request_size
    
    def get_all_stats(self) -> Dict[str, IPStats]:
        """Get all IP statistics"""
        with self.lock:
            return self.ip_stats.copy()

class ConnectionTracker:
    """Tracks concurrent connections per IP"""
    
    def __init__(self, max_connections: int = 10):
        self.max_connections = max_connections
        self.connections: Dict[str, int] = defaultdict(int)
        self.connection_times: Dict[str, deque] = defaultdict(lambda: deque())
        self.lock = threading.RLock()
    
    def add_connection(self, ip: str) -> bool:
        """Add a connection for IP. Returns True if allowed, False if limit exceeded"""
        with self.lock:
            current_time = time.time()
            
            # Clean old connections (older than 1 minute)
            if ip in self.connection_times:
                while (self.connection_times[ip] and 
                       current_time - self.connection_times[ip][0] > 60):
                    self.connection_times[ip].popleft()
                    self.connections[ip] = max(0, self.connections[ip] - 1)
            
            # Check if we can add new connection
            if self.connections[ip] >= self.max_connections:
                return False
            
            self.connections[ip] += 1
            self.connection_times[ip].append(current_time)
            return True
    
    def remove_connection(self, ip: str):
        """Remove a connection for IP"""
        with self.lock:
            if self.connections[ip] > 0:
                self.connections[ip] -= 1
    
    def get_connection_count(self, ip: str) -> int:
        """Get current connection count for IP"""
        with self.lock:
            return self.connections[ip]

class DDoSProtection:
    """Main DDoS protection class"""
    
    def __init__(self, app: Flask = None, config: RateLimitConfig = None):
        self.app = app
        self.config = config or RateLimitConfig()
        self.ip_manager = IPManager()
        self.connection_tracker = ConnectionTracker(self.config.max_concurrent_connections)
        self.redis_client = None
        
        # Initialize Redis if enabled
        if self.config.redis_enabled:
            try:
                self.redis_client = redis.from_url(self.config.redis_url)
                self.redis_client.ping()
                logger.info("Redis connection established for distributed rate limiting")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Using in-memory rate limiting.")
                self.config.redis_enabled = False
        
        # Initialize Flask-Limiter
        self.limiter = Limiter(
            app=app,
            key_func=get_remote_address,
            storage_uri=self.config.redis_url if self.config.redis_enabled else "memory://",
            default_limits=[
                f"{self.config.requests_per_minute} per minute",
                f"{self.config.requests_per_hour} per hour",
                f"{self.config.requests_per_day} per day"
            ]
        )
        
        if app:
            self.init_app(app)
    
    def init_app(self, app: Flask):
        """Initialize Flask app with DDoS protection"""
        self.app = app
        
        # Add middleware
        app.before_request(self._before_request)
        app.after_request(self._after_request)
        app.teardown_request(self._teardown_request)
        
        # Add error handlers
        app.errorhandler(TooManyRequests)(self._handle_rate_limit_exceeded)
        app.errorhandler(RequestEntityTooLarge)(self._handle_request_too_large)
        
        # Add admin routes
        self._add_admin_routes()
    
    def _before_request(self):
        """Middleware to run before each request"""
        client_ip = get_remote_address()
        
        # Check if IP is allowed
        if not self.ip_manager.is_allowed(client_ip):
            logger.warning(f"Blocked request from {client_ip}")
            return jsonify({
                'error': 'Access denied',
                'reason': 'IP blocked',
                'retry_after': 60
            }), 403
        
        # Check connection limits
        if not self.connection_tracker.add_connection(client_ip):
            logger.warning(f"Connection limit exceeded for {client_ip}")
            return jsonify({
                'error': 'Too many concurrent connections',
                'retry_after': 60
            }), 429
        
        # Check request size
        content_length = request.content_length or 0
        if content_length > self.config.max_request_size:
            logger.warning(f"Request too large from {client_ip}: {content_length} bytes")
            return jsonify({
                'error': 'Request too large',
                'max_size': self.config.max_request_size,
                'retry_after': 60
            }), 413
        
        # Check header size
        total_header_size = sum(len(k) + len(v) for k, v in request.headers)
        if total_header_size > self.config.max_header_size:
            logger.warning(f"Headers too large from {client_ip}: {total_header_size} bytes")
            return jsonify({
                'error': 'Headers too large',
                'max_size': self.config.max_header_size,
                'retry_after': 60
            }), 413
        
        # Store request start time for slowloris detection
        g.request_start_time = time.time()
        g.client_ip = client_ip
        
        # Update IP statistics
        self.ip_manager.update_stats(client_ip, content_length)
    
    def _after_request(self, response: Response) -> Response:
        """Middleware to run after each request"""
        if hasattr(g, 'client_ip'):
            # Check for slow requests (slowloris detection)
            if hasattr(g, 'request_start_time'):
                request_duration = time.time() - g.request_start_time
                if request_duration > self.config.slow_request_threshold:
                    logger.warning(f"Slow request from {g.client_ip}: {request_duration:.2f}s")
            
            # Add rate limit headers
            response.headers['X-RateLimit-Limit'] = str(self.config.requests_per_minute)
            response.headers['X-RateLimit-Remaining'] = str(
                max(0, self.config.requests_per_minute - 
                    self.ip_manager.get_stats(g.client_ip).request_count if 
                    self.ip_manager.get_stats(g.client_ip) else 0)
            )
        
        return response
    
    def _teardown_request(self, exception=None):
        """Cleanup after request"""
        if hasattr(g, 'client_ip'):
            self.connection_tracker.remove_connection(g.client_ip)
    
    def _handle_rate_limit_exceeded(self, e):
        """Handle rate limit exceeded errors"""
        client_ip = get_remote_address()
        
        # Increment violation count
        stats = self.ip_manager.get_stats(client_ip)
        if stats:
            stats.violation_count += 1
            
            # Block IP if too many violations
            if stats.violation_count >= self.config.max_violations_before_block:
                self.ip_manager.block_ip(client_ip, self.config.block_duration)
        
        retry_after = getattr(e, 'retry_after', 60)
        
        response = jsonify({
            'error': 'Rate limit exceeded',
            'retry_after': retry_after,
            'message': 'Too many requests. Please slow down.'
        })
        response.status_code = 429
        response.headers['Retry-After'] = str(retry_after)
        
        return response
    
    def _handle_request_too_large(self, e):
        """Handle request too large errors"""
        response = jsonify({
            'error': 'Request too large',
            'max_size': self.config.max_request_size,
            'message': 'Request body exceeds maximum allowed size'
        })
        response.status_code = 413
        response.headers['Retry-After'] = '60'
        
        return response
    
    def _add_admin_routes(self):
        """Add admin API routes"""
        
        @self.app.route('/admin/ddos/stats')
        def admin_stats():
            """Get DDoS protection statistics"""
            stats = self.ip_manager.get_all_stats()
            return jsonify({
                'total_ips': len(stats),
                'blocked_ips': len([s for s in stats.values() if s.blocked_until]),
                'allowlist_size': len(self.ip_manager.allowlist),
                'blocklist_size': len(self.ip_manager.blocklist),
                'redis_enabled': self.config.redis_enabled,
                'config': asdict(self.config)
            })
        
        @self.app.route('/admin/ddos/ips')
        def admin_ips():
            """Get IP statistics"""
            stats = self.ip_manager.get_all_stats()
            return jsonify({
                ip: asdict(stat) for ip, stat in stats.items()
            })
        
        @self.app.route('/admin/ddos/block/<ip>', methods=['POST'])
        def admin_block_ip(ip):
            """Block an IP"""
            duration = request.json.get('duration', self.config.block_duration) if request.is_json else self.config.block_duration
            self.ip_manager.block_ip(ip, duration)
            return jsonify({'message': f'IP {ip} blocked for {duration} seconds'})
        
        @self.app.route('/admin/ddos/unblock/<ip>', methods=['POST'])
        def admin_unblock_ip(ip):
            """Unblock an IP"""
            self.ip_manager.unblock_ip(ip)
            return jsonify({'message': f'IP {ip} unblocked'})
        
        @self.app.route('/admin/ddos/allowlist/<ip>', methods=['POST'])
        def admin_add_allowlist(ip):
            """Add IP to allowlist"""
            self.ip_manager.add_to_allowlist(ip)
            return jsonify({'message': f'IP {ip} added to allowlist'})
        
        @self.app.route('/admin/ddos/blocklist/<ip>', methods=['POST'])
        def admin_add_blocklist(ip):
            """Add IP to blocklist"""
            self.ip_manager.add_to_blocklist(ip)
            return jsonify({'message': f'IP {ip} added to blocklist'})
        
        @self.app.route('/admin/ddos/config', methods=['GET', 'POST'])
        def admin_config():
            """Get or update configuration"""
            if request.method == 'POST':
                if request.is_json:
                    new_config = request.get_json()
                    for key, value in new_config.items():
                        if hasattr(self.config, key):
                            setattr(self.config, key, value)
                    return jsonify({'message': 'Configuration updated'})
            
            return jsonify(asdict(self.config))

def create_ddos_protection(app: Flask = None) -> DDoSProtection:
    """Factory function to create DDoS protection instance"""
    config = RateLimitConfig()
    
    # Load configuration from environment variables
    config.requests_per_minute = int(os.getenv('DDOS_REQUESTS_PER_MINUTE', config.requests_per_minute))
    config.requests_per_hour = int(os.getenv('DDOS_REQUESTS_PER_HOUR', config.requests_per_hour))
    config.requests_per_day = int(os.getenv('DDOS_REQUESTS_PER_DAY', config.requests_per_day))
    config.max_concurrent_connections = int(os.getenv('DDOS_MAX_CONNECTIONS', config.max_concurrent_connections))
    config.max_request_size = int(os.getenv('DDOS_MAX_REQUEST_SIZE', config.max_request_size))
    config.max_header_size = int(os.getenv('DDOS_MAX_HEADER_SIZE', config.max_header_size))
    config.request_timeout = int(os.getenv('DDOS_REQUEST_TIMEOUT', config.request_timeout))
    config.slow_request_threshold = int(os.getenv('DDOS_SLOW_REQUEST_THRESHOLD', config.slow_request_threshold))
    config.block_duration = int(os.getenv('DDOS_BLOCK_DURATION', config.block_duration))
    config.max_violations_before_block = int(os.getenv('DDOS_MAX_VIOLATIONS', config.max_violations_before_block))
    config.redis_url = os.getenv('DDOS_REDIS_URL', config.redis_url)
    config.redis_enabled = os.getenv('DDOS_REDIS_ENABLED', 'false').lower() == 'true'
    
    return DDoSProtection(app, config)

# Example usage and testing
if __name__ == "__main__":
    from flask import Flask
    
    app = Flask(__name__)
    
    # Create DDoS protection
    ddos_protection = create_ddos_protection(app)
    
    @app.route('/')
    def index():
        return jsonify({'message': 'Hello World'})
    
    @app.route('/test')
    def test():
        return jsonify({'message': 'Test endpoint'})
    
    print("DDoS Protection System")
    print("=====================")
    print("Admin endpoints:")
    print("  GET  /admin/ddos/stats")
    print("  GET  /admin/ddos/ips")
    print("  POST /admin/ddos/block/<ip>")
    print("  POST /admin/ddos/unblock/<ip>")
    print("  POST /admin/ddos/allowlist/<ip>")
    print("  POST /admin/ddos/blocklist/<ip>")
    print("  GET/POST /admin/ddos/config")
    print("\nStarting server on http://localhost:8080")
    
    app.run(host='0.0.0.0', port=8080, debug=True)
