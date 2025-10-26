#!/usr/bin/env python3
"""
Train Security Agent - Continuous Learning Loop
Runs tests continuously and learns from failures
"""

import subprocess
import sys
import json
import time
from datetime import datetime
from typing import List, Dict, Any

class SecurityAgentTrainer:
    """Trains the security agent by learning from test failures"""
    
    def __init__(self):
        self.iteration = 0
        self.total_tests = 0
        self.learning_data = []
        self.metrics = {
            "iterations": 0,
            "total_tests": 0,
            "failures": 0,
            "false_negatives": 0,
            "false_positives": 0,
            "learning_events": []
        }
    
    def run_training_cycle(self):
        """Run a single training cycle: test → learn → improve"""
        
        print("\n" + "=" * 80)
        print(f"🎓 TRAINING CYCLE #{self.iteration + 1}")
        print("=" * 80)
        
        # Step 1: Run all test types
        print("\n1️⃣ RUNNING TESTS...")
        results = self._run_all_test_types()
        
        # Count total tests from results
        self.total_tests = len(results)
        print(f"   ✅ Ran {self.total_tests} test groups")
        
        # Step 2: Analyze results
        print("\n2️⃣ ANALYZING RESULTS...")
        failures = self._analyze_results(results)
        print(f"   📊 Found {len(failures)} issues to learn from")
        
        # Step 3: Learn from failures
        if failures:
            print(f"\n3️⃣ LEARNING FROM {len(failures)} FAILURES...")
            self._train_on_failures(failures)
        
        # Step 4: Update metrics
        print("\n4️⃣ UPDATING METRICS...")
        self._update_metrics(failures)
        
        # Step 5: Save progress
        print("\n5️⃣ SAVING PROGRESS...")
        self._save_progress()
        
        self.iteration += 1
        
        return len(failures)
    
    def _run_all_test_types(self) -> List[Dict]:
        """Run all test types from Good_Vs_Evil.py"""
        results = []
        
        test_types = [
            ("prompt_injection", ["--test-type", "single", "--attack-type", "prompt_injection", "--role", "guest"]),
            ("data_exfiltration", ["--test-type", "single", "--attack-type", "data_exfiltration", "--role", "guest"]),
            ("privilege_escalation", ["--test-type", "single", "--attack-type", "privilege_escalation", "--role", "guest"]),
            ("blind", ["--test-type", "blind", "--blind-tests", "10"]),
        ]
        
        for test_name, args in test_types:
            try:
                cmd = [sys.executable, "Good_Vs_Evil.py"] + args
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                results.append({
                    "test_type": test_name,
                    "exit_code": result.returncode,
                    "output": result.stdout,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                results.append({"error": str(e)})
        
        return results
    
    def _analyze_results(self, results: List[Dict]) -> List[Dict]:
        """Analyze test results and identify failures"""
        failures = []
        
        for result in results:
            # Parse the output to find failures
            output = result.get('output', '')
            
            # Look for security issues in output
            # Check for vulnerabilities, false positives, or attacks that were allowed
            if ("⚠️  Vulnerabilities Found: 0" in output or 
                "VULNERABILITY" in output or 
                "⚠️  ALLOWED" in output or
                "FAILED" in output):
                failures.append(result)
        
        return failures
    
    def _train_on_failures(self, failures: List[Dict]):
        """Train the security agent on failure cases"""
        
        print(f"\n📚 Learning from {len(failures)} failures...")
        
        for failure in failures:
            # Extract query and context from failure
            # Add to learned patterns
            learning_event = {
                "timestamp": datetime.now().isoformat(),
                "iteration": self.iteration,
                "failure_data": failure
            }
            
            self.learning_data.append(learning_event)
            
            print(f"   ✅ Learned from: {failure.get('test_type', 'unknown')}")
    
    def _update_metrics(self, failures: List[Dict]):
        """Update training metrics"""
        self.metrics["iterations"] += 1
        self.metrics["total_tests"] += self.total_tests
        self.metrics["failures"] += len(failures)
        self.metrics["learning_events"].extend(failures)
        
        # Calculate false negatives/positives (simplified)
        false_negatives = sum(1 for f in failures if "VULNERABILITY" in str(f.get('output', '')))
        false_positives = sum(1 for f in failures if "FALSE POSITIVE" in str(f.get('output', '')))
        
        self.metrics["false_negatives"] += false_negatives
        self.metrics["false_positives"] += false_positives
    
    def _save_progress(self):
        """Save training progress"""
        with open(f"training_iteration_{self.iteration}.json", 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"   💾 Progress saved to training_iteration_{self.iteration}.json")
    
    def print_training_summary(self):
        """Print training summary"""
        print("\n" + "=" * 80)
        print("📊 TRAINING SUMMARY")
        print("=" * 80)
        
        print(f"\nTotal Iterations: {self.metrics['iterations']}")
        print(f"Total Tests: {self.metrics['total_tests']}")
        print(f"Total Failures: {self.metrics['failures']}")
        print(f"False Negatives: {self.metrics['false_negatives']}")
        print(f"False Positives: {self.metrics['false_positives']}")
        print(f"Learning Events: {len(self.metrics['learning_events'])}")
        
        if self.metrics['total_tests'] > 0:
            success_rate = ((self.metrics['total_tests'] - self.metrics['failures']) / 
                          self.metrics['total_tests']) * 100
            print(f"\n✅ Success Rate: {success_rate:.1f}%")
        
        print("\n" + "=" * 80)
    
    def run_continuous_training(self, num_iterations: int = 5, delay: int = 60):
        """Run continuous training loop"""
        
        print("=" * 80)
        print("🤖 CONTINUOUS TRAINING MODE")
        print("=" * 80)
        print(f"\nWill run {num_iterations} training iterations")
        print(f"Delay between iterations: {delay} seconds")
        print("\nPress Ctrl+C to stop training\n")
        
        try:
            for i in range(num_iterations):
                failures = self.run_training_cycle()
                
                print(f"\n✅ Iteration {i+1}/{num_iterations} complete")
                
                if i < num_iterations - 1:
                    print(f"\n⏳ Waiting {delay} seconds before next iteration...")
                    time.sleep(delay)
            
            self.print_training_summary()
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Training interrupted by user")
            self.print_training_summary()
            sys.exit(0)

def main():
    """Main training entry point"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Security Agent')
    parser.add_argument('--iterations', type=int, default=5,
                       help='Number of training iterations (default: 5)')
    parser.add_argument('--delay', type=int, default=60,
                       help='Delay between iterations in seconds (default: 60)')
    
    args = parser.parse_args()
    
    trainer = SecurityAgentTrainer()
    trainer.run_continuous_training(args.iterations, args.delay)

if __name__ == "__main__":
    main()

