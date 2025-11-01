#!/usr/bin/env python3
"""
Unit tests for Legitimate_Agent.py

Tests for legitimate query generation and false positive testing
"""

import unittest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from Legitimate_Agent import (
    generate_legitimate_operation,
    generate_legitimate_batch,
    ROLE_CLEARANCE_MAPPING,
    LEGITIMATE_OPERATIONS
)


class TestLegitimateOperation(unittest.TestCase):
    """Test legitimate operation generation"""
    
    def test_generate_legitimate_operation_structure(self):
        """Test that operation has correct structure"""
        operation = generate_legitimate_operation(
            user_id="test_user",
            tenant_id="tenantA",
            role="analyst"
        )
        
        self.assertIn("example", operation)
        self.assertIn("metadata", operation)
        self.assertEqual(operation["example"]["role"], "analyst")
        self.assertIn("query", operation["example"])
    
    def test_role_maps_to_correct_clearance(self):
        """Test that roles map to appropriate clearance levels"""
        for role, expected_clearance in ROLE_CLEARANCE_MAPPING.items():
            operation = generate_legitimate_operation(
                user_id="test",
                tenant_id="tenantA",
                role=role
            )
            
            self.assertEqual(operation["example"]["clearance"], expected_clearance)
    
    def test_legitimate_queries_match_role(self):
        """Test that generated queries are appropriate for role"""
        operation = generate_legitimate_operation(
            user_id="test",
            tenant_id="tenantA",
            role="admin",
            operation_type="data_retrieval"
        )
        
        query = operation["example"]["query"]
        # Admin queries should be about admin-level operations
        # This is a basic check - queries should exist
        self.assertIsInstance(query, str)
        self.assertGreater(len(query), 0)
    
    def test_guest_queries_are_public_level(self):
        """Test that guest queries are PUBLIC-level"""
        operation = generate_legitimate_operation(
            user_id="test",
            tenant_id="tenantA",
            role="guest"
        )
        
        self.assertEqual(operation["example"]["clearance"], "PUBLIC")
        query = operation["example"]["query"]
        # Guest queries should be about public operations
        self.assertIsInstance(query, str)


class TestLegitimateBatch(unittest.TestCase):
    """Test batch legitimate operation generation"""
    
    def test_generate_batch_creates_multiple(self):
        """Test that batch generation creates multiple operations"""
        batch = generate_legitimate_batch(
            user_id="test",
            tenant_id="tenantA",
            role="analyst",
            count_per_type=2
        )
        
        # Should have 2 operations per type, 3 types = 6 total
        expected_count = len(LEGITIMATE_OPERATIONS) * 2
        self.assertEqual(len(batch), expected_count)
    
    def test_batch_all_same_role(self):
        """Test that all operations in batch have same role"""
        batch = generate_legitimate_batch(
            user_id="test",
            tenant_id="tenantA",
            role="superuser",
            count_per_type=3
        )
        
        for operation in batch:
            self.assertEqual(operation["example"]["role"], "superuser")
    
    def test_batch_operations_appropriate_clearance(self):
        """Test that batch operations have appropriate clearance for role"""
        batch = generate_legitimate_batch(
            user_id="test",
            tenant_id="tenantA",
            role="analyst",
            count_per_type=2
        )
        
        for operation in batch:
            clearance = operation["example"]["clearance"]
            self.assertEqual(clearance, "INTERNAL")  # Analyst = INTERNAL


class TestFalsePositiveTesting(unittest.TestCase):
    """Test that legitimate operations can be used for FP testing"""
    
    def test_legitimate_queries_should_be_allowed(self):
        """Test that legitimate queries are designed to be allowed"""
        operation = generate_legitimate_operation(
            user_id="test",
            tenant_id="tenantA",
            role="analyst",
            operation_type="data_retrieval"
        )
        
        # Metadata should indicate it's legitimate
        metadata = operation["metadata"]
        self.assertEqual(metadata["config"]["severity"], "LEGITIMATE")
        self.assertEqual(metadata["config"]["authorization_level"], "APPROPRIATE")
        self.assertEqual(metadata["config"]["risk_level"], "LOW")


if __name__ == '__main__':
    unittest.main()

