#!/usr/bin/env python3
"""
Quick Start Training Script
One command to train your security agent
"""

import sys
import subprocess

def main():
    """Quick start training"""
    
    print("""
╔════════════════════════════════════════════════════════════════╗
║           VecSec Continuous Learning - Quick Start             ║
╚════════════════════════════════════════════════════════════════╝

This will:
  1️⃣  Run security tests
  2️⃣  Learn from failures  
  3️⃣  Improve detection
  4️⃣  Save metrics

Starting in 3 seconds... (Press Ctrl+C to cancel)
""")
    
    import time
    time.sleep(3)
    
    # Run training
    try:
        subprocess.run([
            sys.executable, 
            "train_security_agent.py",
            "--iterations", "5",
            "--delay", "30"
        ], check=True)
    except KeyboardInterrupt:
        print("\n\n⚠️  Training cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTry running manually:")
        print("   python3 train_security_agent.py --iterations 5")
        sys.exit(1)

if __name__ == "__main__":
    main()

