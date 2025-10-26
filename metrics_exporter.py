#!/usr/bin/env python3
"""
VecSec Metrics Exporter for Prometheus
Exports BaseTen API usage, inference metrics, and training data
"""

from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import threading
from typing import Dict
import os

# BaseTen API Metrics
baseten_api_calls = Counter('vecsec_baseten_api_calls_total',
    'Total BaseTen API calls',
    ['status', 'model'])
baseten_api_errors = Counter('vecsec_baseten_api_calls_failed_total',
    'Total failed BaseTen API calls',
    ['error_type'])
baseten_api_duration = Histogram('vecsec_baseten_api_duration_seconds',
    'BaseTen API call duration',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0])
baseten_credits = Gauge('vecsec_baseten_credits_remaining',
    'Estimated BaseTen credits remaining')

# Inference Metrics
inference_total = Counter('vecsec_inference_total',
    'Total inference requests',
    ['status'])
inference_duration = Histogram('vecsec_inference_duration_seconds',
    'Inference duration',
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0])
inference_queue_size = Gauge('vecsec_inference_queue_size',
    'Number of requests in inference queue')

# Security Metrics
attacks_blocked = Counter('vecsec_attacks_blocked_total',
    'Attacks blocked by type',
    ['attack_type'])
threats_detected = Counter('vecsec_threats_detected_total',
    'Threats detected by type',
    ['threat_type'])
rlsa_violations = Counter('vecsec_rlsa_violations_total',
    'RLSA policy violations',
    ['violation_type'])

# Semantic Detection Metrics
semantic_threats = Counter('vecsec_semantic_threats_total',
    'Semantic similarity threats detected')
semantic_similarity = Histogram('vecsec_semantic_similarity_score',
    'Semantic threat similarity scores',
    buckets=[0.0, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0])

# Training Metrics
patterns_learned = Counter('vecsec_patterns_learned_total',
    'Threat patterns learned',
    ['pattern_type'])
embeddings_generated = Counter('vecsec_embedding_generation_total',
    'Embeddings generated')
model_updates = Counter('vecsec_model_updates_total',
    'Model update events')

# System Health
system_health = Gauge('vecsec_system_health',
    'Overall system health (1=healthy, 0=unhealthy)')
active_sandboxes = Gauge('vecsec_active_sandboxes',
    'Number of active sandboxes')

class MetricsExporter:
    """Export metrics for VecSec monitoring"""
    
    def __init__(self, port: int = 9091):
        self.port = port
        self.server_started = False
    
    def start_server(self):
        """Start the metrics server"""
        if not self.server_started:
            start_http_server(self.port)
            self.server_started = True
            print(f"✅ Metrics exporter started on port {self.port}")
    
    def track_baseten_call(self, status: str, model: str, duration: float):
        """Track BaseTen API call"""
        baseten_api_calls.labels(status=status, model=model).inc()
        baseten_api_duration.observe(duration)
    
    def track_baseten_error(self, error_type: str):
        """Track BaseTen API error"""
        baseten_api_errors.labels(error_type=error_type).inc()
    
    def set_baseten_credits(self, credits: float):
        """Update BaseTen credits remaining"""
        baseten_credits.set(credits)
    
    def track_inference(self, status: str, duration: float):
        """Track inference request"""
        inference_total.labels(status=status).inc()
        inference_duration.observe(duration)
    
    def track_attack_blocked(self, attack_type: str):
        """Track blocked attack"""
        attacks_blocked.labels(attack_type=attack_type).inc()
    
    def track_threat_detected(self, threat_type: str):
        """Track detected threat"""
        threats_detected.labels(threat_type=threat_type).inc()
    
    def track_rlsa_violation(self, violation_type: str):
        """Track RLSA violation"""
        rlsa_violations.labels(violation_type=violation_type).inc()
    
    def track_semantic_threat(self, similarity: float):
        """Track semantic threat detection"""
        semantic_threats.inc()
        semantic_similarity.observe(similarity)
    
    def track_pattern_learned(self, pattern_type: str):
        """Track learned pattern"""
        patterns_learned.labels(pattern_type=pattern_type).inc()
    
    def track_embedding_generated(self):
        """Track embedding generation"""
        embeddings_generated.inc()
    
    def set_system_health(self, healthy: bool):
        """Set system health status"""
        system_health.set(1 if healthy else 0)
    
    def set_active_sandboxes(self, count: int):
        """Update active sandbox count"""
        active_sandboxes.set(count)

# Global metrics exporter instance
metrics_exporter = MetricsExporter()

def start_metrics_exporter(port: int = 9091):
    """Start metrics exporter in background"""
    metrics_exporter.start_server()
    print(f"📊 Prometheus metrics available at http://localhost:{port}/metrics")

if __name__ == "__main__":
    # Start metrics server
    start_metrics_exporter(9091)
    
    # Demo: Generate some sample metrics
    print("Generating sample metrics...")
    for i in range(10):
        metrics_exporter.track_baseten_call("success", "qwen3-8b", 0.25)
        metrics_exporter.track_inference("success", 0.15)
        metrics_exporter.track_attack_blocked("prompt_injection")
        metrics_exporter.set_system_health(True)
        time.sleep(1)
    
    print("Sample metrics generated. Server running on port 9091")
    print("Prometheus will scrape metrics from http://localhost:9091/metrics")
    
    # Keep server running
    while True:
        time.sleep(1)

