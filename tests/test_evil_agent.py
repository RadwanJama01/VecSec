#!/usr/bin/env python3
"""
Unit tests for Evil_Agent.py

Tests for attack generation and privilege escalation
"""

import os
import sys
import unittest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from Evil_Agent import (
    ATTACK_TYPES,
    generate_attack,
    generate_batch,
    generate_privilege_escalation_attack,
)


class TestAttackGeneration(unittest.TestCase):
    """Test attack generation functions"""
    
    def test_generate_attack_creates_valid_structure(self):
        """Test that generate_attack returns valid structure"""
        attack = generate_attack(
            user_id="test_user",
            tenant_id="tenantA",
            clearance="INTERNAL",
            attack_type="prompt_injection",
            role="analyst"
        )
        
        self.assertIn("example", attack)
        self.assertIn("metadata", attack)
        self.assertEqual(attack["metadata"]["attack_type"], "prompt_injection")
        self.assertIn("query", attack["example"])
    
    def test_generate_attack_uses_role(self):
        """Test that generate_attack uses provided role"""
        attack = generate_attack(
            user_id="test_user",
            tenant_id="tenantA",
            clearance="INTERNAL",
            attack_type="jailbreak",
            role="guest"
        )
        
        self.assertEqual(attack["example"]["role"], "guest")
    
    def test_all_attack_types_generatable(self):
        """Test that all attack types can be generated"""
        for attack_type in ATTACK_TYPES.keys():
            attack = generate_attack(
                user_id="test",
                tenant_id="tenantA",
                clearance="INTERNAL",
                attack_type=attack_type,
                role="analyst"
            )
            self.assertEqual(attack["metadata"]["attack_type"], attack_type)
    
    def test_generate_batch_creates_multiple(self):
        """Test that generate_batch creates multiple attacks"""
        batch = generate_batch(
            user_id="test",
            tenant_id="tenantA",
            clearance="INTERNAL",
            attack_types=["prompt_injection", "jailbreak"],
            count_per_type=2,
            role="analyst"
        )
        
        self.assertEqual(len(batch), 4)  # 2 types * 2 per type
        attack_types = [a["metadata"]["attack_type"] for a in batch]
        self.assertEqual(attack_types.count("prompt_injection"), 2)
        self.assertEqual(attack_types.count("jailbreak"), 2)


class TestPrivilegeEscalation(unittest.TestCase):
    """Test privilege escalation attack generation"""
    
    def test_privilege_escalation_has_role(self):
        """Test that privilege escalation includes role"""
        attack = generate_privilege_escalation_attack(
            user_id="test",
            tenant_id="tenantA"
        )
        
        self.assertIn("role", attack["example"])
        self.assertIn("clearance", attack["example"])
    
    def test_privilege_escalation_scenarios(self):
        """Test that escalation scenarios are valid"""
        # Generate multiple to test different scenarios
        scenarios_seen = set()
        for _ in range(20):
            attack = generate_privilege_escalation_attack(
                user_id="test",
                tenant_id="tenantA"
            )
            role = attack["example"]["role"]
            clearance = attack["example"]["clearance"]
            scenarios_seen.add((role, clearance))
        
        # Should have multiple different scenarios
        self.assertGreater(len(scenarios_seen), 1)
    
    def test_privilege_escalation_targets_higher_clearance(self):
        """Test that escalation attempts target higher clearance"""
        attack = generate_privilege_escalation_attack(
            user_id="test",
            tenant_id="tenantA"
        )
        
        query = attack["example"]["query"]
        clearance = attack["example"]["clearance"]
        
        # Query should mention trying to access higher clearance
        # This is a simple check - might need refinement
        if clearance == "PUBLIC":
            # Should try to access INTERNAL, CONFIDENTIAL, or SECRET
            self.assertTrue(
                any(level in query.upper() 
                    for level in ["INTERNAL", "CONFIDENTIAL", "SECRET"])
            )


if __name__ == '__main__':
    unittest.main()

