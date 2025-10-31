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
import json
from datetime import datetime

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

# Request & Performance Metrics
request_total = Counter('vecsec_requests_total',
    'Total requests processed',
    ['status'])
request_duration = Histogram('vecsec_request_duration_seconds',
    'Request processing duration',
    buckets=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0])
files_processed = Counter('vecsec_files_processed_total',
    'Total files processed',
    ['action'])  # action: 'blocked' or 'approved'
requests_per_second = Gauge('vecsec_requests_per_second',
    'Current requests per second')

# Detection & Accuracy Metrics
detection_accuracy = Gauge('vecsec_detection_accuracy',
    'Overall detection accuracy (0-1)')
false_positives = Counter('vecsec_false_positives_total',
    'False positive detections')
false_negatives = Counter('vecsec_false_negatives_total',
    'False negative detections')
true_positives = Counter('vecsec_true_positives_total',
    'True positive detections')

# Rate Metrics
block_rate = Gauge('vecsec_block_rate',
    'Percentage of blocked requests (0-100)')
approval_rate = Gauge('vecsec_approval_rate',
    'Percentage of approved requests (0-100)')

# System Health
system_health = Gauge('vecsec_system_health',
    'Overall system health (1=healthy, 0=unhealthy)')
system_uptime = Gauge('vecsec_system_uptime_seconds',
    'System uptime in seconds')
active_sandboxes = Gauge('vecsec_active_sandboxes',
    'Number of active sandboxes')

class MetricsExporter:
    """Export metrics for VecSec monitoring"""
    
    def __init__(self, port: int = 9091, persist_file: str = "data/training/vecsec_metrics.json"):
        self.port = port
        self.persist_file = persist_file
        self.server_started = False
        self.metrics_snapshot = {}
        self.load_metrics()
    
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
    
    def track_learning_event(self, event_data: Dict):
        """Track learning event for continuous learning"""
        patterns_learned.labels(pattern_type=event_data.get('attack_type', 'unknown')).inc()
        model_updates.labels(status='success').inc()
    
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
    
    def track_request(self, status: str, duration: float):
        """Track request processing"""
        request_total.labels(status=status).inc()
        request_duration.observe(duration)
    
    def track_file_processed(self, action: str):
        """Track file processing (blocked or approved)"""
        files_processed.labels(action=action).inc()
        self._update_rates()
    
    def track_detection_result(self, is_accurate: bool, is_blocked: bool, is_threat: bool):
        """Track detection accuracy"""
        if is_accurate:
            if is_threat and is_blocked:
                true_positives.inc()
        else:
            if is_blocked and not is_threat:
                false_positives.inc()
            elif not is_blocked and is_threat:
                false_negatives.inc()
        self._calculate_accuracy()
    
    def _calculate_accuracy(self):
        """Calculate overall detection accuracy"""
        # Get total positive and negative results
        # Accuracy = (TP + TN) / (TP + TN + FP + FN)
        tp = self._get_counter_value('vecsec_true_positives_total')
        fp = self._get_counter_value('vecsec_false_positives_total')
        fn = self._get_counter_value('vecsec_false_negatives_total')
        
        if tp + fp + fn > 0:
            accuracy = tp / (tp + fp + fn)
            detection_accuracy.set(accuracy)
    
    def _update_rates(self):
        """Update block and approval rates"""
        blocked = self._get_counter_value('vecsec_files_processed_total', {'action': 'blocked'})
        approved = self._get_counter_value('vecsec_files_processed_total', {'action': 'approved'})
        total = blocked + approved
        
        if total > 0:
            block_rate.set((blocked / total) * 100)
            approval_rate.set((approved / total) * 100)
    
    def _get_counter_value(self, metric_name: str, labels: Dict = None):
        """Helper to get counter value"""
        try:
            from prometheus_client import REGISTRY
            for collector in REGISTRY.collectors:
                if hasattr(collector, '_name') and collector._name == metric_name:
                    if labels:
                        # For labeled metrics
                        for metric in collector.collect()[0].samples:
                            if metric.labels == labels:
                                return metric.value
                    else:
                        # For unlabeled metrics
                        return collector.collect()[0].samples[0].value
            return 0
        except:
            return 0
    
    def set_requests_per_second(self, rps: float):
        """Update requests per second"""
        requests_per_second.set(rps)
    
    def update_uptime(self, seconds: float):
        """Update system uptime"""
        system_uptime.set(seconds)
    
    def save_metrics(self):
        """Save metrics snapshot to JSON file for persistence"""
        snapshot = {
            "timestamp": datetime.utcnow().isoformat(),
            "detection_accuracy": self._get_gauge_value('vecsec_detection_accuracy'),
            "block_rate": self._get_gauge_value('vecsec_block_rate'),
            "approval_rate": self._get_gauge_value('vecsec_approval_rate'),
            "system_health": self._get_gauge_value('vecsec_system_health'),
            "system_uptime": self._get_gauge_value('vecsec_system_uptime_seconds'),
            "requests_per_second": self._get_gauge_value('vecsec_requests_per_second'),
            "total_requests_allowed": self._get_counter_value('vecsec_requests_total', {'status': 'allowed'}),
            "total_requests_blocked": self._get_counter_value('vecsec_requests_total', {'status': 'blocked'}),
            "files_blocked": self._get_counter_value('vecsec_files_processed_total', {'action': 'blocked'}),
            "files_approved": self._get_counter_value('vecsec_files_processed_total', {'action': 'approved'}),
            "attacks_blocked": self._get_counter_value('vecsec_attacks_blocked_total'),
            "threats_detected": self._get_counter_value('vecsec_threats_detected_total'),
            "rlsa_violations": self._get_counter_value('vecsec_rlsa_violations_total'),
            "patterns_learned": self._get_counter_value('vecsec_patterns_learned_total'),
            "embeddings_generated": self._get_counter_value('vecsec_embedding_generation_total'),
            "model_updates": self._get_counter_value('vecsec_model_updates_total')
        }
        
        self.metrics_snapshot = snapshot
        
        try:
            with open(self.persist_file, 'w') as f:
                json.dump(snapshot, f, indent=2)
        except Exception as e:
            print(f"⚠️  Could not save metrics to {self.persist_file}: {e}")
    
    def load_metrics(self):
        """Load metrics snapshot from JSON file"""
        try:
            if os.path.exists(self.persist_file):
                with open(self.persist_file, 'r') as f:
                    self.metrics_snapshot = json.load(f)
                print(f"✅ Loaded persisted metrics from {self.persist_file}")
        except Exception as e:
            print(f"⚠️  Could not load metrics from {self.persist_file}: {e}")
            self.metrics_snapshot = {}
    
    def _get_gauge_value(self, metric_name: str):
        """Helper to get gauge value"""
        try:
            from prometheus_client import REGISTRY
            for collector in REGISTRY.collectors:
                if hasattr(collector, '_name') and collector._name == metric_name:
                    return collector.collect()[0].samples[0].value
            return 0
        except:
            return 0
    
    def auto_save(self, interval: int = 30):
        """Automatically save metrics every N seconds"""
        def _save_loop():
            while True:
                time.sleep(interval)
                self.save_metrics()
        
        thread = threading.Thread(target=_save_loop, daemon=True)
        thread.start()
        print(f"✅ Auto-save enabled (every {interval}s)")

# Global metrics exporter instance
metrics_exporter = MetricsExporter()

def start_metrics_exporter(port: int = 9091):
    """Start metrics exporter in background"""
    metrics_exporter.start_server()
    metrics_exporter.auto_save(interval=30)  # Auto-save every 30 seconds
    print(f"📊 Prometheus metrics available at http://localhost:{port}/metrics")
    print(f"💾 Metrics persisting to: {metrics_exporter.persist_file}")

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

