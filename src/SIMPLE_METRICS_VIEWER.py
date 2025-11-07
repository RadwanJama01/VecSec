#!/usr/bin/env python3
"""
Simple Metrics Viewer - View metrics without Docker
Shows real-time metrics from Sec_Agent in the terminal
"""

import glob
import json
import os


def view_metrics():
    """View current metrics"""
    print("\n" + "=" * 80)
    print("  📊 VECSEC METRICS")
    print("=" * 80 + "\n")

    # Check for learning metrics (try data/training first, then root)
    metrics_paths = [
        "data/training/vecsec_metrics.json",
        "vecsec_metrics.json",
        "learning_metrics.json",
    ]
    for path in metrics_paths:
        if os.path.exists(path):
            with open(path) as f:
                metrics = json.load(f)
                print("📈 LEARNING METRICS:")
                print(f"   Total Tests: {metrics.get('total_tests', 0)}")
                print(f"   Failures: {metrics.get('failures', 0)}")
                print(f"   False Negatives: {metrics.get('false_negatives', 0)}")
                print(f"   False Positives: {metrics.get('false_positives', 0)}")
                print(
                    f"   Success Rate: {100 - (metrics.get('failures', 0) / max(metrics.get('total_tests', 1), 1) * 100):.1f}%"
                )
                print("")
                break

    # Check for training iterations
    training_paths = ["data/training/training_iteration_*.json", "training_iteration_*.json"]
    training_files = []
    for pattern in training_paths:
        training_files.extend(glob.glob(pattern))
    training_files = sorted(set(training_files), reverse=True)

    if training_files:
        print("🎓 TRAINING PROGRESS:")
        for i, f in enumerate(training_files[:3], 1):
            with open(f) as file:
                data = json.load(file)
                print(
                    f"   Iteration {i}: {data.get('metrics', {}).get('learning_events', 0)} events"
                )
        print("")

    # Check for exported results
    attack_paths = ["data/attacks/malicious_inputs_*.json", "malicious_inputs_*.json"]
    result_files = []
    for pattern in attack_paths:
        result_files.extend(glob.glob(pattern))
    result_files = sorted(set(result_files), reverse=True)

    if result_files:
        with open(result_files[0]) as f:
            data = json.load(f)
            info = data.get("export_info", {})
            print("📊 LATEST TEST RESULTS:")
            print(f"   Tests Run: {info.get('total_attacks', 0)}")
            print(f"   Vulnerabilities: {info.get('vulnerabilities_found', 0)}")
            print("")

    print("=" * 80)
    print("\n💡 To view metrics continuously, run:")
    print("   watch -n 2 python3 SIMPLE_METRICS_VIEWER.py")
    print("")


if __name__ == "__main__":
    view_metrics()
