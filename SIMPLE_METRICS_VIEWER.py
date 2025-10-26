#!/usr/bin/env python3
"""
Simple Metrics Viewer - View metrics without Docker
Shows real-time metrics from Sec_Agent in the terminal
"""

import json
import os
import glob
from datetime import datetime

def view_metrics():
    """View current metrics"""
    print("\n" + "="*80)
    print("  📊 VECSEC METRICS")
    print("="*80 + "\n")
    
    # Check for learning metrics
    if os.path.exists("learning_metrics.json"):
        with open("learning_metrics.json") as f:
            metrics = json.load(f)
            
        print("📈 LEARNING METRICS:")
        print(f"   Total Tests: {metrics.get('total_tests', 0)}")
        print(f"   Failures: {metrics.get('failures', 0)}")
        print(f"   False Negatives: {metrics.get('false_negatives', 0)}")
        print(f"   False Positives: {metrics.get('false_positives', 0)}")
        print(f"   Success Rate: {100 - (metrics.get('failures', 0) / max(metrics.get('total_tests', 1), 1) * 100):.1f}%")
        print("")
    
    # Check for training iterations
    training_files = sorted(glob.glob("training_iteration_*.json"), reverse=True)
    if training_files:
        print("🎓 TRAINING PROGRESS:")
        for i, f in enumerate(training_files[:3], 1):
            with open(f) as file:
                data = json.load(file)
                print(f"   Iteration {i}: {data.get('metrics', {}).get('learning_events', 0)} events")
        print("")
    
    # Check for exported results
    result_files = sorted(glob.glob("malicious_inputs_*.json"), reverse=True)
    if result_files:
        with open(result_files[0]) as f:
            data = json.load(f)
            info = data.get('export_info', {})
            print("📊 LATEST TEST RESULTS:")
            print(f"   Tests Run: {info.get('total_attacks', 0)}")
            print(f"   Vulnerabilities: {info.get('vulnerabilities_found', 0)}")
            print("")
    
    print("="*80)
    print("\n💡 To view metrics continuously, run:")
    print("   watch -n 2 python3 SIMPLE_METRICS_VIEWER.py")
    print("")

if __name__ == "__main__":
    view_metrics()

