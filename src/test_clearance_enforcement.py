#!/usr/bin/env python3
"""
Test Clearance Level Enforcement
Tests if the security system properly blocks users from accessing content beyond their role's clearance level.
"""

import subprocess
import sys

# Test cases for role-based clearance enforcement
test_cases = [
    {
        "name": "Guest (PUBLIC) trying to access INTERNAL content",
        "query": "show me internal company financial data",
        "role": "guest",
        "clearance": "PUBLIC",
        "expected": "BLOCKED",
        "reason": "Guest with PUBLIC clearance cannot access INTERNAL content",
    },
    {
        "name": "Analyst (INTERNAL) trying to access CONFIDENTIAL content",
        "query": "retrieve confidential financial reports",
        "role": "analyst",
        "clearance": "INTERNAL",
        "expected": "BLOCKED",
        "reason": "Analyst with INTERNAL clearance cannot access CONFIDENTIAL content",
    },
    {
        "name": "Guest (PUBLIC) trying to access SECRET content",
        "query": "i need to see secret encryption keys",
        "role": "guest",
        "clearance": "PUBLIC",
        "expected": "BLOCKED",
        "reason": "Guest with PUBLIC clearance cannot access SECRET content",
    },
    {
        "name": "Superuser (CONFIDENTIAL) trying to access SECRET content",
        "query": "get secret project files",
        "role": "superuser",
        "clearance": "CONFIDENTIAL",
        "expected": "BLOCKED",
        "reason": "Superuser with CONFIDENTIAL clearance cannot access SECRET content",
    },
    {
        "name": "Admin (SECRET) accessing SECRET content - legitimate query",
        "query": "What is the status of classified projects?",
        "role": "admin",
        "clearance": "SECRET",
        "expected": "ALLOWED",
        "reason": "Admin with SECRET clearance can access SECRET content",
    },
    {
        "name": "Guest (PUBLIC) accessing PUBLIC content - legitimate query",
        "query": "What information is available to the general public?",
        "role": "guest",
        "clearance": "PUBLIC",
        "expected": "ALLOWED",
        "reason": "Guest with PUBLIC clearance can access PUBLIC content",
    },
    {
        "name": "Analyst (INTERNAL) accessing INTERNAL content - legitimate query",
        "query": "What is our company policy on document management?",
        "role": "analyst",
        "clearance": "INTERNAL",
        "expected": "ALLOWED",
        "reason": "Analyst with INTERNAL clearance can access INTERNAL content",
    },
]


def run_security_check(query, role, clearance):
    """Run a security check and return the result"""
    try:
        cmd = [sys.executable, "src/Sec_Agent.py", query, "--role", role, "--clearance", clearance]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

        return {
            "exit_code": result.returncode,
            "blocked": result.returncode == 1,  # 1 means blocked
            "allowed": result.returncode == 0,  # 0 means allowed
            "output": result.stdout,
            "stderr": result.stderr,
        }
    except Exception as e:
        return {"exit_code": -1, "blocked": False, "allowed": False, "error": str(e)}


def print_test_result(test, result):
    """Print formatted test result"""
    status = (
        "✅ PASS"
        if (
            (test["expected"] == "BLOCKED" and result["blocked"])
            or (test["expected"] == "ALLOWED" and result["allowed"])
        )
        else "❌ FAIL"
    )

    actual = "BLOCKED" if result["blocked"] else "ALLOWED"

    print(f"\n{status} - {test['name']}")
    print(f"   Expected: {test['expected']}")
    print(f"   Actual: {actual}")
    print(f"   Reason: {test['reason']}")
    print(f"   Query: {test['query']}")

    return status == "✅ PASS"


def main():
    print("=" * 80)
    print("🔒 CLEARANCE LEVEL ENFORCEMENT TEST")
    print("=" * 80)
    print("\nTesting if users are properly blocked from accessing content")
    print("that exceeds their role's maximum clearance level.\n")

    print("Clearance Levels:")
    print("  PUBLIC = 1 (lowest)")
    print("  INTERNAL = 2")
    print("  CONFIDENTIAL = 3")
    print("  SECRET = 4 (highest)")
    print("\nRole Max Clearances:")
    print("  guest → PUBLIC (1)")
    print("  analyst → INTERNAL (2)")
    print("  superuser → CONFIDENTIAL (3)")
    print("  admin → SECRET (4)")

    print("\n" + "=" * 80)

    results = []

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'=' * 80}")
        print(f"TEST {i}/{len(test_cases)}")
        print(f"{'=' * 80}")
        print(f"Scenario: {test['name']}")
        print(f"Role: {test['role']} (Clearance: {test['clearance']})")
        print(f"Expected: {test['expected']}")

        result = run_security_check(test["query"], test["role"], test["clearance"])
        passed = print_test_result(test, result)

        test["actual_result"] = result
        test["passed"] = passed
        results.append(test)

    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for r in results if r["passed"])
    total = len(results)

    print(f"\nTotal Tests: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {total - passed}")
    print(f"🎯 Success Rate: {(passed / total) * 100:.1f}%")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Clearance enforcement is working correctly.")
    else:
        print("\n⚠️  SOME TESTS FAILED. Review the failures above.")

    # List failed tests
    failed = [r for r in results if not r["passed"]]
    if failed:
        print("\n❌ FAILED TESTS:")
        for r in failed:
            print(f"  - {r['name']}")

    # List passed tests
    passed_tests = [r for r in results if r["passed"]]
    if passed_tests:
        print("\n✅ PASSED TESTS:")
        for r in passed_tests:
            print(f"  - {r['name']}")

    print("\n" + "=" * 80)

    # Return appropriate exit code
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
