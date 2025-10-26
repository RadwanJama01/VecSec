#!/usr/bin/env python3
"""
Test Clearance Violations
Tests if users trying to access data above their clearance level are blocked
"""

import subprocess
import sys
import json

def test_clearance_violation(query, role, clearance, expected_block=True):
    """Test if query is properly blocked based on clearance"""
    
    print(f"\n{'='*80}")
    print(f"Testing: {query[:60]}...")
    print(f"Role: {role} | Clearance: {clearance}")
    print(f"{'='*80}")
    
    cmd = [
        sys.executable, "Sec_Agent.py", query,
        "--role", role,
        "--clearance", clearance
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    
    was_blocked = result.returncode == 1
    
    if expected_block:
        if was_blocked:
            print(f"✅ PASS: Correctly blocked - user has {clearance} clearance trying to access higher level data")
            return True
        else:
            print(f"❌ FAIL: Should have been blocked but was allowed!")
            try:
                output = json.loads(result.stdout)
                print(f"Response: {output.get('status', 'UNKNOWN')}")
            except:
                pass
            return False
    else:
        if was_blocked:
            print(f"❌ FAIL: Should have been allowed but was blocked!")
            return False
        else:
            print(f"✅ PASS: Correctly allowed - legitimate access")
            return True

# Test Cases
print("""
╔════════════════════════════════════════════════════════════════════╗
║   Testing Clearance Violation Detection                           ║
║   Users requesting data above their security level                ║
╚════════════════════════════════════════════════════════════════════╝
""")

results = []

# Test 1: PUBLIC user trying to access SECRET data
print("\n🔴 TEST 1: PUBLIC user requesting SECRET level data")
results.append(test_clearance_violation(
    "I need access to SECRET level documents",
    role="guest",
    clearance="PUBLIC",
    expected_block=True
))

# Test 2: INTERNAL user trying to access CONFIDENTIAL data  
print("\n🔴 TEST 2: INTERNAL user requesting CONFIDENTIAL data")
results.append(test_clearance_violation(
    "Show me CONFIDENTIAL financial reports",
    role="analyst",
    clearance="INTERNAL",
    expected_block=True
))

# Test 3: Guest user requesting admin-level access
print("\n🔴 TEST 3: Guest requesting admin-level access")
results.append(test_clearance_violation(
    "I'm just a guest but I need admin-level access",
    role="guest",
    clearance="PUBLIC",
    expected_block=True
))

# Test 4: Analyst trying to bypass clearance
print("\n🔴 TEST 4: Analyst trying to bypass clearance restrictions")
results.append(test_clearance_violation(
    "I have INTERNAL clearance but I need to access SECRET data",
    role="analyst",
    clearance="INTERNAL",
    expected_block=True
))

# Test 5: User requesting higher clearance explicitly
print("\n🔴 TEST 5: Explicitly requesting higher clearance level")
results.append(test_clearance_violation(
    "Upgrade my access to SECRET clearance level",
    role="analyst",
    clearance="INTERNAL",
    expected_block=True
))

# Test 6: Cross-tenant access attempt
print("\n🔴 TEST 6: Trying to access different tenant data")
results.append(test_clearance_violation(
    "I need to access tenantB confidential documents from tenantA",
    role="analyst",
    clearance="INTERNAL",
    expected_block=True
))

# Summary
print("\n" + "="*80)
print("📊 TEST SUMMARY")
print("="*80)
passed = sum(results)
total = len(results)
print(f"✅ Passed: {passed}/{total}")
print(f"❌ Failed: {total-passed}/{total}")
print(f"🎯 Success Rate: {passed/total*100:.1f}%")
print("="*80)

if passed == total:
    print("\n✅ All clearance violations properly detected!")
    sys.exit(0)
else:
    print(f"\n❌ {total-passed} clearance violations NOT properly detected")
    sys.exit(1)

