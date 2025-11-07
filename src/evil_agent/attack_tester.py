"""
Attack Tester - Integration Testing
Tests generated attacks against the security agent system
"""

import subprocess
import sys


def test_attack_against_agent(
    query: str, tenant_id: str, clearance: str, agent_path: str = "src/Sec_Agent.py"
):
    """Test the generated attack against the security agent system"""

    try:
        cmd = [
            sys.executable,
            agent_path,
            query,
            "--tenant-id",
            tenant_id,
            "--clearance",
            clearance,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        print(f"Query: {query}")
        print(f"Exit code: {result.returncode}")
        print(f"Response: {result.stdout}")

        if result.stderr:
            print(f"Errors: {result.stderr}")

    except subprocess.TimeoutExpired:
        print("❌ Test timed out - agent may be vulnerable to DoS")
    except Exception as e:
        print(f"❌ Test failed: {e}")
