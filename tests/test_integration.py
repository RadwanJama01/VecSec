#!/usr/bin/env python3
"""
Integration tests for VecSec

Tests that verify components work together correctly
"""

import unittest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestAttackGenerationToDetection(unittest.TestCase):
    """Test that generated attacks are properly detected"""
    
    def test_prompt_injection_generated_and_detected(self):
        """Test that Evil_Agent generates prompt injection that Sec_Agent detects"""
        from Evil_Agent import generate_attack
        from Sec_Agent import extract_query_context
        
        # Generate prompt injection attack
        attack = generate_attack(
            user_id="test",
            tenant_id="tenantA",
            clearance="INTERNAL",
            attack_type="prompt_injection",
            role="analyst"
        )
        
        query = attack["example"]["query"]
        
        # Check if Sec_Agent would detect it
        context = extract_query_context(query)
        
        # Should detect prompt injection
        self.assertIn("prompt_injection", context["detected_threats"])
    
    def test_legitimate_query_not_detected_as_threat(self):
        """Test that legitimate queries are not detected as threats"""
        from Legitimate_Agent import generate_legitimate_operation
        from Sec_Agent import extract_query_context
        
        # Generate legitimate operation
        operation = generate_legitimate_operation(
            user_id="test",
            tenant_id="tenantA",
            role="analyst",
            operation_type="data_retrieval"
        )
        
        query = operation["example"]["query"]
        
        # Check if Sec_Agent would detect it as threat
        context = extract_query_context(query)
        
        # Should NOT be detected as threat
        # Note: Some legitimate queries might contain words that trigger detection
        # This is acceptable - the full RLS check will allow them if appropriate
        threats = context.get("detected_threats", [])
        
        # Check that it's not always_blocked threats (prompt_injection, jailbreak, etc.)
        always_blocked = ["prompt_injection", "jailbreak", "obfuscation", "privilege_escalation"]
        detected_always_blocked = [t for t in threats if t in always_blocked]
        
        # Legitimate queries should not trigger always-blocked threats
        # (May still trigger role-dependent threats like "data_exfiltration" keyword)
        self.assertEqual(len(detected_always_blocked), 0)


if __name__ == '__main__':
    unittest.main()

