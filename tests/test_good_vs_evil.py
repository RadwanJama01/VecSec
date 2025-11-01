#!/usr/bin/env python3
"""
Unit tests for Good_Vs_Evil.py

Tests for security testing framework
"""

import unittest
import sys
import os
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from Good_Vs_Evil import SecurityTester


class TestSecurityTester(unittest.TestCase):
    """Test SecurityTester class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.tester = SecurityTester()
    
    def test_tester_initialization(self):
        """Test that SecurityTester initializes correctly"""
        self.assertEqual(len(self.tester.test_results), 0)
        self.assertEqual(len(self.tester.vulnerabilities_found), 0)
    
    @patch('src.Good_Vs_Evil.subprocess.run')
    def test_run_agent_test_handles_blocked(self, mock_run):
        """Test that _run_agent_test handles blocked requests"""
        mock_result = MagicMock()
        mock_result.returncode = 1  # Blocked
        mock_result.stdout = "BLOCKED"
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        
        result = self.tester._run_agent_test(
            query="test query",
            tenant_id="tenantA",
            clearance="INTERNAL",
            role="analyst"
        )
        
        self.assertEqual(result["exit_code"], 1)
        self.assertFalse(result["success"])
        self.assertIn("elapsed_time", result)
    
    @patch('src.Good_Vs_Evil.subprocess.run')
    def test_run_agent_test_handles_allowed(self, mock_run):
        """Test that _run_agent_test handles allowed requests"""
        mock_result = MagicMock()
        mock_result.returncode = 0  # Allowed
        mock_result.stdout = "ALLOWED"
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        
        result = self.tester._run_agent_test(
            query="test query",
            tenant_id="tenantA",
            clearance="INTERNAL",
            role="analyst"
        )
        
        self.assertEqual(result["exit_code"], 0)
        self.assertTrue(result["success"])
    
    @patch('src.Good_Vs_Evil.subprocess.run')
    def test_run_agent_test_handles_timeout(self, mock_run):
        """Test that _run_agent_test handles timeouts"""
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired("cmd", 30)
        
        result = self.tester._run_agent_test(
            query="test query",
            tenant_id="tenantA",
            clearance="INTERNAL",
            role="analyst"
        )
        
        self.assertEqual(result["exit_code"], -1)
        self.assertFalse(result["success"])
        self.assertTrue(result.get("timeout", False))
    
    def test_generate_report_creates_valid_structure(self):
        """Test that generate_report creates valid structure"""
        # Add some test results
        self.tester.test_results = [
            {
                "attack_type": "prompt_injection",
                "security_status": "BLOCKED",
                "vulnerability_detected": False
            },
            {
                "attack_type": "jailbreak",
                "security_status": "BLOCKED",
                "vulnerability_detected": False
            }
        ]
        
        report = self.tester.generate_report()
        
        self.assertIn("summary", report)
        self.assertIn("detailed_results", report)
        self.assertEqual(report["summary"]["total_tests"], 2)


if __name__ == '__main__':
    unittest.main()

