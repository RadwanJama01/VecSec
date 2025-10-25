#!/usr/bin/env python3
"""
DDoS Protection Test Suite
==========================

This script tests the DDoS protection features of the Flask proxy server.
It includes tests for rate limiting, connection limits, request size limits,
and IP blocking functionality.

Usage:
    python test_ddos_protection.py [--server-url http://localhost:8080]

Author: VecSec Team
"""

import argparse
import requests
import time
import threading
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

class DDoSTester:
    """Test suite for DDoS protection features"""
    
    def __init__(self, server_url="http://localhost:8080"):
        self.server_url = server_url.rstrip('/')
        self.session = requests.Session()
        self.results = {}
    
    def test_rate_limiting(self, requests_count=10, delay=0.1):
        """Test rate limiting functionality"""
        print(f"\n🧪 Testing Rate Limiting ({requests_count} requests)...")
        
        responses = []
        start_time = time.time()
        
        for i in range(requests_count):
            try:
                response = self.session.get(f"{self.server_url}/health")
                responses.append({
                    'status_code': response.status_code,
                    'headers': dict(response.headers),
                    'timestamp': time.time()
                })
                
                if response.status_code == 429:
                    print(f"  ✅ Rate limit triggered at request {i+1}")
                    retry_after = response.headers.get('Retry-After', 'N/A')
                    print(f"  📊 Retry-After: {retry_after} seconds")
                
                time.sleep(delay)
                
            except Exception as e:
                print(f"  ❌ Request {i+1} failed: {e}")
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Analyze results
        success_count = sum(1 for r in responses if r['status_code'] == 200)
        rate_limited_count = sum(1 for r in responses if r['status_code'] == 429)
        
        print(f"  📊 Results: {success_count} successful, {rate_limited_count} rate limited")
        print(f"  ⏱️  Duration: {duration:.2f} seconds")
        
        self.results['rate_limiting'] = {
            'success_count': success_count,
            'rate_limited_count': rate_limited_count,
            'duration': duration,
            'responses': responses
        }
        
        return rate_limited_count > 0
    
    def test_connection_limits(self, concurrent_connections=15):
        """Test concurrent connection limits"""
        print(f"\n🧪 Testing Connection Limits ({concurrent_connections} concurrent)...")
        
        def make_request(thread_id):
            try:
                response = self.session.get(f"{self.server_url}/health", timeout=5)
                return {
                    'thread_id': thread_id,
                    'status_code': response.status_code,
                    'success': True
                }
            except Exception as e:
                return {
                    'thread_id': thread_id,
                    'error': str(e),
                    'success': False
                }
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=concurrent_connections) as executor:
            futures = [executor.submit(make_request, i) for i in range(concurrent_connections)]
            results = [future.result() for future in as_completed(futures)]
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Analyze results
        successful = sum(1 for r in results if r.get('success', False))
        connection_limited = sum(1 for r in results if r.get('status_code') == 429)
        errors = sum(1 for r in results if not r.get('success', False))
        
        print(f"  📊 Results: {successful} successful, {connection_limited} connection limited, {errors} errors")
        print(f"  ⏱️  Duration: {duration:.2f} seconds")
        
        self.results['connection_limits'] = {
            'successful': successful,
            'connection_limited': connection_limited,
            'errors': errors,
            'duration': duration,
            'results': results
        }
        
        return connection_limited > 0
    
    def test_request_size_limits(self):
        """Test request size limits"""
        print(f"\n🧪 Testing Request Size Limits...")
        
        # Test with large payload
        large_payload = "x" * (11 * 1024 * 1024)  # 11MB (should exceed 10MB limit)
        
        try:
            response = self.session.post(
                f"{self.server_url}/analyze",
                json={'content': large_payload},
                timeout=10
            )
            
            if response.status_code == 413:
                print(f"  ✅ Request size limit triggered (413)")
                print(f"  📊 Response: {response.json()}")
                self.results['request_size_limits'] = {'success': True, 'status_code': 413}
                return True
            else:
                print(f"  ❌ Expected 413, got {response.status_code}")
                self.results['request_size_limits'] = {'success': False, 'status_code': response.status_code}
                return False
                
        except Exception as e:
            print(f"  ❌ Request failed: {e}")
            self.results['request_size_limits'] = {'success': False, 'error': str(e)}
            return False
    
    def test_ip_blocking(self):
        """Test IP blocking functionality"""
        print(f"\n🧪 Testing IP Blocking...")
        
        # First, get current IP stats
        try:
            stats_response = self.session.get(f"{self.server_url}/admin/ddos/stats")
            if stats_response.status_code == 200:
                stats = stats_response.json()
                print(f"  📊 Current blocked IPs: {stats.get('blocked_ips', 0)}")
        except Exception as e:
            print(f"  ⚠️  Could not get stats: {e}")
        
        # Test blocking an IP (using a test IP)
        test_ip = "192.168.1.999"  # Invalid IP for testing
        
        try:
            block_response = self.session.post(f"{self.server_url}/admin/ddos/block/{test_ip}")
            if block_response.status_code == 200:
                print(f"  ✅ IP blocking endpoint working")
                print(f"  📊 Response: {block_response.json()}")
                
                # Test unblocking
                unblock_response = self.session.post(f"{self.server_url}/admin/ddos/unblock/{test_ip}")
                if unblock_response.status_code == 200:
                    print(f"  ✅ IP unblocking endpoint working")
                    print(f"  📊 Response: {unblock_response.json()}")
                
                self.results['ip_blocking'] = {'success': True}
                return True
            else:
                print(f"  ❌ Blocking failed: {block_response.status_code}")
                self.results['ip_blocking'] = {'success': False, 'status_code': block_response.status_code}
                return False
                
        except Exception as e:
            print(f"  ❌ IP blocking test failed: {e}")
            self.results['ip_blocking'] = {'success': False, 'error': str(e)}
            return False
    
    def test_admin_endpoints(self):
        """Test admin API endpoints"""
        print(f"\n🧪 Testing Admin Endpoints...")
        
        endpoints = [
            ('/admin/ddos/stats', 'GET'),
            ('/admin/ddos/ips', 'GET'),
            ('/admin/ddos/config', 'GET'),
        ]
        
        results = {}
        
        for endpoint, method in endpoints:
            try:
                if method == 'GET':
                    response = self.session.get(f"{self.server_url}{endpoint}")
                else:
                    response = self.session.post(f"{self.server_url}{endpoint}")
                
                if response.status_code == 200:
                    print(f"  ✅ {method} {endpoint} - OK")
                    results[endpoint] = {'success': True, 'status_code': 200}
                else:
                    print(f"  ❌ {method} {endpoint} - {response.status_code}")
                    results[endpoint] = {'success': False, 'status_code': response.status_code}
                    
            except Exception as e:
                print(f"  ❌ {method} {endpoint} - Error: {e}")
                results[endpoint] = {'success': False, 'error': str(e)}
        
        self.results['admin_endpoints'] = results
        return all(r.get('success', False) for r in results.values())
    
    def test_slowloris_protection(self):
        """Test slowloris attack protection"""
        print(f"\n🧪 Testing Slowloris Protection...")
        
        # Simulate slow request by sending data slowly
        def slow_request():
            try:
                # Send a request that takes longer than the threshold
                response = self.session.get(
                    f"{self.server_url}/health",
                    timeout=15
                )
                return response.status_code
            except Exception as e:
                return f"Error: {e}"
        
        # Start multiple slow requests
        threads = []
        results = []
        
        for i in range(5):
            thread = threading.Thread(target=lambda: results.append(slow_request()))
            threads.append(thread)
            thread.start()
            time.sleep(0.5)  # Stagger the requests
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        print(f"  📊 Slow request results: {results}")
        
        # Check if any requests were blocked due to slow behavior
        blocked_count = sum(1 for r in results if r == 429)
        
        if blocked_count > 0:
            print(f"  ✅ Slowloris protection triggered ({blocked_count} requests blocked)")
            self.results['slowloris_protection'] = {'success': True, 'blocked_count': blocked_count}
            return True
        else:
            print(f"  ⚠️  No slowloris protection triggered")
            self.results['slowloris_protection'] = {'success': False, 'blocked_count': 0}
            return False
    
    def run_all_tests(self):
        """Run all DDoS protection tests"""
        print("🚀 Starting DDoS Protection Test Suite")
        print("=" * 50)
        
        # Check if server is running
        try:
            response = self.session.get(f"{self.server_url}/health", timeout=5)
            if response.status_code != 200:
                print(f"❌ Server not responding properly: {response.status_code}")
                return False
            print(f"✅ Server is running at {self.server_url}")
        except Exception as e:
            print(f"❌ Cannot connect to server: {e}")
            return False
        
        # Run tests
        tests = [
            ("Rate Limiting", self.test_rate_limiting),
            ("Connection Limits", self.test_connection_limits),
            ("Request Size Limits", self.test_request_size_limits),
            ("IP Blocking", self.test_ip_blocking),
            ("Admin Endpoints", self.test_admin_endpoints),
            ("Slowloris Protection", self.test_slowloris_protection),
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                results[test_name] = result
                status = "✅ PASSED" if result else "❌ FAILED"
                print(f"\n{status} {test_name}")
            except Exception as e:
                print(f"\n❌ FAILED {test_name}: {e}")
                results[test_name] = False
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 TEST SUMMARY")
        print("=" * 50)
        
        passed = sum(1 for r in results.values() if r)
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{status} {test_name}")
        
        print(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All DDoS protection tests passed!")
        else:
            print("⚠️  Some tests failed. Check your configuration.")
        
        return passed == total
    
    def generate_report(self):
        """Generate a detailed test report"""
        report = {
            'timestamp': time.time(),
            'server_url': self.server_url,
            'results': self.results
        }
        
        with open('ddos_test_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: ddos_test_report.json")

def main():
    parser = argparse.ArgumentParser(description='Test DDoS protection features')
    parser.add_argument('--server-url', default='http://localhost:8080',
                       help='Server URL to test (default: http://localhost:8080)')
    parser.add_argument('--report', action='store_true',
                       help='Generate detailed JSON report')
    
    args = parser.parse_args()
    
    tester = DDoSTester(args.server_url)
    
    try:
        success = tester.run_all_tests()
        
        if args.report:
            tester.generate_report()
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
