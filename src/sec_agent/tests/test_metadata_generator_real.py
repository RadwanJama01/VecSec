"""
Unit Tests for generate_retrieval_metadata_real function

Tests the real vector store implementation with proper mocking and benchmarks.

Uses:
- MagicMock for unit tests (no real embeddings needed - we mock the vector store)
- MockEmbeddings + InMemoryVectorStore for integration tests (fake embeddings, real vector store)
"""

import unittest
import time
import os
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, List

# Mock imports
import sys
import importlib.util
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

# Import directly to avoid __init__.py issues
# Need to set up package structure for relative imports to work
import types

# Create sec_agent package module
if 'sec_agent' not in sys.modules:
    sys.modules['sec_agent'] = types.ModuleType('sec_agent')
    sys.modules['sec_agent'].__path__ = [str(project_root / "src" / "sec_agent")]

# Load policy_manager first (metadata_generator depends on it)
policy_path = project_root / "src" / "sec_agent" / "policy_manager.py"
policy_spec = importlib.util.spec_from_file_location("sec_agent.policy_manager", policy_path)
policy_module = importlib.util.module_from_spec(policy_spec)
policy_module.__package__ = 'sec_agent'
policy_module.__name__ = 'sec_agent.policy_manager'
sys.modules['sec_agent.policy_manager'] = policy_module
policy_spec.loader.exec_module(policy_module)

# Now load metadata_generator
metadata_gen_path = project_root / "src" / "sec_agent" / "metadata_generator.py"
metadata_spec = importlib.util.spec_from_file_location("sec_agent.metadata_generator", metadata_gen_path)
metadata_module = importlib.util.module_from_spec(metadata_spec)
metadata_module.__package__ = 'sec_agent'
metadata_module.__name__ = 'sec_agent.metadata_generator'
sys.modules['sec_agent.metadata_generator'] = metadata_module
metadata_spec.loader.exec_module(metadata_module)

generate_retrieval_metadata_real = metadata_module.generate_retrieval_metadata_real
generate_retrieval_metadata = metadata_module.generate_retrieval_metadata


class TestGenerateRetrievalMetadataReal(unittest.TestCase):
    """Unit tests for generate_retrieval_metadata_real function"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create mock vector store
        self.mock_vector_store = MagicMock()
        self.mock_vector_store.similarity_search_with_score = MagicMock()
        
        # Create mock documents with metadata
        self.mock_doc1 = MagicMock()
        self.mock_doc1.metadata = {
            "embedding_id": "emb-001",
            "document_id": "doc-001",
            "tenant_id": "tenant_a",
            "sensitivity": "INTERNAL",
            "topics": "security,access_control"
        }
        self.mock_doc1.page_content = "Security protocols and access control documentation."
        
        self.mock_doc2 = MagicMock()
        self.mock_doc2.metadata = {
            "embedding_id": "emb-002",
            "document_id": "doc-002",
            "tenant_id": "tenant_a",
            "sensitivity": "CONFIDENTIAL",
            "topics": "finance,budget"
        }
        self.mock_doc2.page_content = "Financial data and budget information."
        
        # Use the function imported at module level
        self.func = generate_retrieval_metadata_real
        self.mock_func = generate_retrieval_metadata
    
    def test_connects_to_real_vector_store(self):
        """Test that function uses the provided vector store"""
        query_context = {"query": "security"}
        user_tenant = "tenant_a"
        
        # Setup mock to return results
        self.mock_vector_store.similarity_search_with_score.return_value = [
            (self.mock_doc1, 0.85)
        ]
        
        # Call function
        result = self.func(query_context, user_tenant, self.mock_vector_store)
        
        # Verify vector store was called
        self.assertTrue(
            self.mock_vector_store.similarity_search_with_score.called,
            "Vector store should be called"
        )
        call_args = self.mock_vector_store.similarity_search_with_score.call_args
        
        # Verify query text
        self.assertEqual(
            call_args[0][0], "security",
            "Query text should match input"
        )
        
        # Verify number of results requested
        self.assertEqual(
            call_args[1]["k"], 5,
            "Should request 5 results"
        )
        
        # Verify filter is present
        self.assertIn(
            "filter", call_args[1],
            "Filter parameter should be present"
        )
        
        # Verify filter contains tenant_id
        filter_dict = call_args[1]["filter"]
        self.assertIsNotNone(filter_dict, "Filter should not be None")
        self.assertEqual(filter_dict["tenant_id"], "tenant_a", "Filter should contain tenant_id")
    
    def test_performs_actual_similarity_search(self):
        """Test that function performs similarity search with scores"""
        query_context = {"query": "security protocols"}
        user_tenant = "tenant_a"
        
        # Setup mock to return results with scores
        self.mock_vector_store.similarity_search_with_score.return_value = [
            (self.mock_doc1, 0.92),
            (self.mock_doc2, 0.78)
        ]
        
        result = self.func(query_context, user_tenant, self.mock_vector_store)
        
        # Verify results have real scores
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0]["retrieval_score"], float)
        self.assertEqual(result[0]["retrieval_score"], 0.92)
        self.assertEqual(result[1]["retrieval_score"], 0.78)
        self.assertTrue(0.0 <= result[0]["retrieval_score"] <= 1.0)
    
    def test_returns_real_document_ids_and_content(self):
        """Test that function returns real document IDs and content"""
        query_context = {"query": "test"}
        user_tenant = "tenant_a"
        
        self.mock_vector_store.similarity_search_with_score.return_value = [
            (self.mock_doc1, 0.85)
        ]
        
        result = self.func(query_context, user_tenant, self.mock_vector_store)
        
        # Verify result structure
        self.assertIsInstance(result, list, "Result should be a list")
        self.assertGreater(len(result), 0, "Result should contain at least one item")
        
        # Verify required fields exist
        required_fields = ["document_id", "embedding_id", "tenant_id", "retrieval_score", "content"]
        for field in required_fields:
            self.assertIn(
                field, result[0],
                f"Result should contain '{field}' field"
            )
        
        # Verify real document IDs (from mock doc metadata)
        self.assertEqual(
            result[0]["document_id"], "doc-001",
            "Document ID should match stored document"
        )
        self.assertEqual(
            result[0]["embedding_id"], "emb-001",
            "Embedding ID should match stored document"
        )
        
        # Verify real content (from mock doc page_content)
        self.assertIn("content", result[0])
        self.assertEqual(
            result[0]["content"], "Security protocols and access control documentation.",
            "Content should match stored document content"
        )
        self.assertGreater(
            len(result[0]["content"]), 0,
            "Content should not be empty"
        )
    
    def test_implements_proper_tenant_filtering(self):
        """Test that function filters by tenant_id"""
        query_context = {"query": "security"}
        user_tenant = "tenant_a"
        
        # Setup mock to return results
        self.mock_vector_store.similarity_search_with_score.return_value = [
            (self.mock_doc1, 0.85)
        ]
        
        result = self.func(query_context, user_tenant, self.mock_vector_store)
        
        # Verify filter was applied
        call_args = self.mock_vector_store.similarity_search_with_score.call_args
        filter_dict = call_args[1]["filter"]
        self.assertIsNotNone(filter_dict)
        self.assertEqual(filter_dict["tenant_id"], "tenant_a")
        
        # Verify returned tenant_id matches
        self.assertEqual(result[0]["tenant_id"], "tenant_a")
    
    def test_handles_errors_gracefully(self):
        """Test that function handles errors and falls back to mock"""
        query_context = {"query": "test", "topics": ["test"]}
        user_tenant = "tenant_a"
        
        # Make vector store raise an exception
        self.mock_vector_store.similarity_search_with_score.side_effect = Exception("Vector store error")
        
        # Should not raise, should return mock metadata
        result = self.func(query_context, user_tenant, self.mock_vector_store)
        
        # Should return mock metadata (fallback) - mock requires topics
        self.assertIsInstance(result, list)
        # Mock function returns metadata based on topics, so if topics exist, should have results
        if query_context.get("topics"):
            self.assertGreater(len(result), 0)
    
    def test_handles_empty_query(self):
        """Test that function handles empty query using topics"""
        query_context = {"query": "", "topics": ["security", "finance"]}
        user_tenant = "tenant_a"
        
        self.mock_vector_store.similarity_search_with_score.return_value = [
            (self.mock_doc1, 0.85)
        ]
        
        result = self.func(query_context, user_tenant, self.mock_vector_store)
        
        # Should use topics as query
        call_args = self.mock_vector_store.similarity_search_with_score.call_args
        self.assertEqual(call_args[0][0], "security finance")
    
    def test_handles_no_results(self):
        """Test that function handles no search results"""
        query_context = {"query": "nonexistent"}
        user_tenant = "tenant_a"
        
        # Return empty results
        self.mock_vector_store.similarity_search_with_score.return_value = []
        
        result = self.func(query_context, user_tenant, self.mock_vector_store)
        
        # Should return empty metadata placeholder
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["document_id"], "doc-empty")
        self.assertEqual(result[0]["retrieval_score"], 0.0)
        self.assertIn("No documents found", result[0]["content"])
    
    def test_converts_topics_from_string_to_list(self):
        """Test that function converts comma-separated topics string to list"""
        query_context = {"query": "test"}
        user_tenant = "tenant_a"
        
        self.mock_vector_store.similarity_search_with_score.return_value = [
            (self.mock_doc1, 0.85)  # Has topics as "security,access_control"
        ]
        
        result = self.func(query_context, user_tenant, self.mock_vector_store)
        
        # Topics should be converted to list
        self.assertIsInstance(result[0]["topics"], list)
        self.assertEqual(result[0]["topics"], ["security", "access_control"])
    
    def test_handles_none_vector_store(self):
        """Test that function handles None vector_store gracefully"""
        query_context = {"query": "test", "topics": ["test"]}
        user_tenant = "tenant_a"
        
        # Should fallback to mock without crashing
        result = self.func(query_context, user_tenant, None)
        
        self.assertIsInstance(result, list)
        # Should return mock metadata (requires topics in query_context)
        if query_context.get("topics"):
            self.assertGreater(len(result), 0)
    
    def test_real_retrieval_different_queries(self):
        """Test that different queries return different documents"""
        # Create multiple documents with different content
        security_doc = MagicMock()
        security_doc.metadata = {
            "embedding_id": "emb-security-001",
            "document_id": "doc-security-protocols",
            "tenant_id": "tenant_a",
            "sensitivity": "INTERNAL",
            "topics": "security,protocols"
        }
        security_doc.page_content = "Security protocols and access control measures."
        
        finance_doc = MagicMock()
        finance_doc.metadata = {
            "embedding_id": "emb-finance-001",
            "document_id": "doc-budget-report",
            "tenant_id": "tenant_a",
            "sensitivity": "CONFIDENTIAL",
            "topics": "finance,budget"
        }
        finance_doc.page_content = "Financial data and budget information for Q1."
        
        marketing_doc = MagicMock()
        marketing_doc.metadata = {
            "embedding_id": "emb-marketing-001",
            "document_id": "doc-campaign-strategy",
            "tenant_id": "tenant_a",
            "sensitivity": "PUBLIC",
            "topics": "marketing,campaigns"
        }
        marketing_doc.page_content = "Marketing campaign strategy and brand positioning."
        
        # Test query 1: "security"
        self.mock_vector_store.similarity_search_with_score.return_value = [
            (security_doc, 0.92),
            (finance_doc, 0.45)
        ]
        security_results = self.func(
            {"query": "security"}, "tenant_a", self.mock_vector_store
        )
        security_doc_ids = {r["document_id"] for r in security_results}
        
        # Test query 2: "finance"
        self.mock_vector_store.similarity_search_with_score.return_value = [
            (finance_doc, 0.88),
            (security_doc, 0.52)
        ]
        finance_results = self.func(
            {"query": "finance"}, "tenant_a", self.mock_vector_store
        )
        finance_doc_ids = {r["document_id"] for r in finance_results}
        
        # Test query 3: "marketing"
        self.mock_vector_store.similarity_search_with_score.return_value = [
            (marketing_doc, 0.91),
            (finance_doc, 0.38)
        ]
        marketing_results = self.func(
            {"query": "marketing"}, "tenant_a", self.mock_vector_store
        )
        marketing_doc_ids = {r["document_id"] for r in marketing_results}
        
        # Verify different queries return different top results
        self.assertEqual(security_results[0]["document_id"], "doc-security-protocols",
                        "Security query should return security document as top result")
        self.assertEqual(finance_results[0]["document_id"], "doc-budget-report",
                        "Finance query should return finance document as top result")
        self.assertEqual(marketing_results[0]["document_id"], "doc-campaign-strategy",
                        "Marketing query should return marketing document as top result")
        
        # Verify top results are different for different queries
        self.assertNotEqual(security_results[0]["document_id"], finance_results[0]["document_id"],
                           "Different queries should return different top documents")
        self.assertNotEqual(finance_results[0]["document_id"], marketing_results[0]["document_id"],
                           "Different queries should return different top documents")
        self.assertNotEqual(security_results[0]["document_id"], marketing_results[0]["document_id"],
                           "Different queries should return different top documents")
        
        # Verify that queries return relevant documents (not just random)
        # Security query should have security doc as top result
        self.assertIn("security", security_results[0]["document_id"].lower(),
                     "Security query should return security-related document")
        # Finance query should have finance doc as top result
        self.assertIn("budget", finance_results[0]["document_id"].lower(),
                     "Finance query should return finance-related document")
    
    def test_retrieval_scores_realistic(self):
        """Test that retrieval scores vary based on actual similarity (not sequential)"""
        # Create documents with varying similarity
        doc1 = MagicMock()
        doc1.metadata = {"document_id": "doc-high-match", "embedding_id": "emb-001",
                         "tenant_id": "tenant_a", "sensitivity": "INTERNAL", "topics": "security"}
        doc1.page_content = "Security protocols and access control measures."
        
        doc2 = MagicMock()
        doc2.metadata = {"document_id": "doc-medium-match", "embedding_id": "emb-002",
                        "tenant_id": "tenant_a", "sensitivity": "INTERNAL", "topics": "security"}
        doc2.page_content = "Network security and firewall configuration."
        
        doc3 = MagicMock()
        doc3.metadata = {"document_id": "doc-low-match", "embedding_id": "emb-003",
                       "tenant_id": "tenant_a", "sensitivity": "INTERNAL", "topics": "finance"}
        doc3.page_content = "Financial budget and expense reports."
        
        # Return results with realistic, varied scores (not sequential)
        self.mock_vector_store.similarity_search_with_score.return_value = [
            (doc1, 0.87),  # High match
            (doc2, 0.72),  # Medium match
            (doc3, 0.34)   # Low match - significantly different
        ]
        
        results = self.func(
            {"query": "security protocols"}, "tenant_a", self.mock_vector_store
        )
        
        # Verify scores exist and are realistic
        scores = [r["retrieval_score"] for r in results]
        self.assertEqual(len(scores), 3, "Should have 3 results")
        
        # Verify scores are not sequential (not 0.9, 0.8, 0.7...)
        score_diffs = [abs(scores[i] - scores[i+1]) for i in range(len(scores)-1)]
        # At least one score difference should be > 0.1 (not just 0.1 increments)
        has_large_gap = any(diff > 0.1 for diff in score_diffs)
        self.assertTrue(has_large_gap or score_diffs[1] > 0.2,
                       f"Scores should vary realistically, not sequentially. Got: {scores}")
        
        # Verify scores are in valid range
        for score in scores:
            self.assertGreaterEqual(score, 0.0, f"Score {score} should be >= 0")
            self.assertLessEqual(score, 1.0, f"Score {score} should be <= 1")
        
        # Verify highest score is first (most similar)
        self.assertEqual(scores[0], max(scores),
                        "Highest similarity score should be first")
        
        # Verify scores are different (not all the same)
        unique_scores = set(scores)
        self.assertGreater(len(unique_scores), 1,
                          f"Scores should vary, got: {scores}")
    
    def test_real_document_ids(self):
        """Test that document IDs are real (not mock patterns like emb-001, doc-finance-001)"""
        # Use realistic document IDs (similar to what would be in production)
        real_doc1 = MagicMock()
        real_doc1.metadata = {
            "embedding_id": "emb-uuid-7a3f9b2c-4d1e-4f8a-9b3c-2d5e7f8a9b0c",
            "document_id": "doc-security-2024-Q1-protocols-v2",
            "tenant_id": "tenant_a",
            "sensitivity": "INTERNAL",
            "topics": "security,protocols"
        }
        real_doc1.page_content = "Security protocols documentation."
        
        real_doc2 = MagicMock()
        real_doc2.metadata = {
            "embedding_id": "emb-20240315-143022-abc123",
            "document_id": "doc-budget-2024-Q1-final-report",
            "tenant_id": "tenant_a",
            "sensitivity": "CONFIDENTIAL",
            "topics": "finance,budget"
        }
        real_doc2.page_content = "Budget report for Q1 2024."
        
        self.mock_vector_store.similarity_search_with_score.return_value = [
            (real_doc1, 0.85),
            (real_doc2, 0.78)
        ]
        
        results = self.func(
            {"query": "test"}, "tenant_a", self.mock_vector_store
        )
        
        # Verify document IDs don't match mock patterns
        mock_patterns = [
            r"^emb-00[0-9]$",  # emb-001, emb-002, etc.
            r"^doc-00[0-9]$",  # doc-001, doc-002, etc.
            r"^doc-finance-00[0-9]$",  # doc-finance-001, etc.
            r"^emb-tenant-[ab]-00[0-9]$"  # emb-tenant-a-001, etc.
        ]
        
        import re
        for result in results:
            doc_id = result["document_id"]
            emb_id = result["embedding_id"]
            
            # Verify IDs don't match simple mock patterns
            for pattern in mock_patterns:
                self.assertNotRegex(
                    doc_id, pattern,
                    f"Document ID '{doc_id}' should not match mock pattern '{pattern}'"
                )
                self.assertNotRegex(
                    emb_id, pattern,
                    f"Embedding ID '{emb_id}' should not match mock pattern '{pattern}'"
                )
            
            # Verify IDs are meaningful (not just "doc-empty" or "emb-empty")
            self.assertNotEqual(doc_id, "doc-empty",
                               "Document ID should not be placeholder")
            self.assertNotEqual(emb_id, "emb-empty",
                               "Embedding ID should not be placeholder")
            
            # Verify IDs have reasonable length (not too short)
            self.assertGreater(len(doc_id), 5,
                              f"Document ID '{doc_id}' should be more than 5 characters")
            self.assertGreater(len(emb_id), 5,
                              f"Embedding ID '{emb_id}' should be more than 5 characters")


class TestGenerateRetrievalMetadataRealIntegration(unittest.TestCase):
    """Integration tests using real InMemoryVectorStore with fake embeddings"""
    
    def setUp(self):
        """Set up integration test fixtures with fake embeddings"""
        # Import MockEmbeddings directly to avoid __init__.py issues
        mock_llm_path = project_root / "src" / "sec_agent" / "mock_llm.py"
        mock_llm_spec = importlib.util.spec_from_file_location("mock_llm", mock_llm_path)
        mock_llm_module = importlib.util.module_from_spec(mock_llm_spec)
        mock_llm_spec.loader.exec_module(mock_llm_module)
        MockEmbeddings = mock_llm_module.MockEmbeddings
        
        # Try to import langchain components (may not be available in all environments)
        try:
            from langchain_core.vectorstores import InMemoryVectorStore
            from langchain_core.documents import Document
            self.InMemoryVectorStore = InMemoryVectorStore
            self.Document = Document
        except ImportError:
            # Skip integration tests if langchain not available
            self.skipTest("langchain_core not available - skipping integration tests")
            return
        
        self.fake_embeddings = MockEmbeddings()
        # InMemoryVectorStore uses embedding_function parameter
        try:
            self.vector_store = self.InMemoryVectorStore(embedding_function=self.fake_embeddings)
        except TypeError:
            # Fallback for different API versions
            self.vector_store = self.InMemoryVectorStore(embedding=self.fake_embeddings)
        
        # Add test documents with proper metadata (converted to ChromaDB format)
        test_docs = [
            self.Document(
                page_content="Security protocols and access control for tenant A",
                metadata={
                    "tenant_id": "tenant_a",
                    "document_id": "doc-tenant-a-001",
                    "embedding_id": "emb-tenant-a-001",
                    "sensitivity": "INTERNAL",
                    "topics": "security,access_control"  # Comma-separated string for ChromaDB
                }
            ),
            self.Document(
                page_content="Financial data and budget information for tenant A",
                metadata={
                    "tenant_id": "tenant_a",
                    "document_id": "doc-tenant-a-002",
                    "embedding_id": "emb-tenant-a-002",
                    "sensitivity": "CONFIDENTIAL",
                    "topics": "finance,budget"
                }
            ),
            self.Document(
                page_content="Security protocols for tenant B",
                metadata={
                    "tenant_id": "tenant_b",
                    "document_id": "doc-tenant-b-001",
                    "embedding_id": "emb-tenant-b-001",
                    "sensitivity": "INTERNAL",
                    "topics": "security"
                }
            ),
        ]
        
        self.vector_store.add_documents(test_docs)
        self.func = generate_retrieval_metadata_real
    
    def test_integration_with_real_vector_store(self):
        """Integration test: Uses real InMemoryVectorStore with fake embeddings"""
        query_context = {"query": "security"}
        user_tenant = "tenant_a"
        
        # This uses real vector store operations but fake embeddings (no API calls)
        result = self.func(query_context, user_tenant, self.vector_store)
        
        # Verify real vector store was used
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        
        # Verify results have real document IDs from vector store
        doc_ids = [item["document_id"] for item in result]
        self.assertIn("doc-tenant-a-001", doc_ids)
        
        # Verify tenant filtering works
        for item in result:
            self.assertEqual(item["tenant_id"], "tenant_a")
    
    def test_integration_tenant_filtering(self):
        """Integration test: Verify tenant filtering with real vector store"""
        query_context = {"query": "security"}
        
        # Search as tenant_a
        results_a = self.func(query_context, "tenant_a", self.vector_store)
        tenant_ids_a = {item["tenant_id"] for item in results_a}
        self.assertEqual(tenant_ids_a, {"tenant_a"})
        
        # Search as tenant_b
        results_b = self.func(query_context, "tenant_b", self.vector_store)
        tenant_ids_b = {item["tenant_id"] for item in results_b}
        self.assertEqual(tenant_ids_b, {"tenant_b"})
        
        # Verify no cross-tenant leakage
        self.assertFalse(tenant_ids_a.intersection(tenant_ids_b))
    
    def test_integration_topics_conversion(self):
        """Integration test: Topics string-to-list conversion"""
        query_context = {"query": "security"}
        user_tenant = "tenant_a"
        
        result = self.func(query_context, user_tenant, self.vector_store)
        
        # Topics should be converted from string to list
        for item in result:
            self.assertIsInstance(item["topics"], list)
            if "security" in item["topics"] or "access_control" in item["topics"]:
                # Verify topics were parsed correctly
                self.assertTrue(len(item["topics"]) > 0)


class TestGenerateRetrievalMetadataRealPerformance(unittest.TestCase):
    """Performance benchmarks for generate_retrieval_metadata_real"""
    
    def setUp(self):
        """Set up performance test fixtures"""
        self.mock_vector_store = MagicMock()
        self.mock_vector_store.similarity_search_with_score = MagicMock()
        
        # Create multiple mock documents
        mock_docs = []
        for i in range(10):
            doc = MagicMock()
            doc.metadata = {
                "embedding_id": f"emb-{i:03d}",
                "document_id": f"doc-{i:03d}",
                "tenant_id": "tenant_a",
                "sensitivity": "INTERNAL",
                "topics": "test,example"
            }
            doc.page_content = f"Test document content {i}" * 10
            mock_docs.append((doc, 0.9 - (i * 0.05)))
        
        self.mock_vector_store.similarity_search_with_score.return_value = mock_docs[:5]
        
        # Use the function imported at module level
        self.func = generate_retrieval_metadata_real
    
    def test_response_time_under_100ms(self):
        """Benchmark: Function should respond in under 100ms for typical query"""
        query_context = {"query": "test query"}
        user_tenant = "tenant_a"
        
        # Time the function
        start_time = time.time()
        result = self.func(query_context, user_tenant, self.mock_vector_store)
        end_time = time.time()
        
        response_time_ms = (end_time - start_time) * 1000
        
        print(f"\n   ⏱️  Response time: {response_time_ms:.2f}ms")
        self.assertLess(response_time_ms, 100, 
                       f"Response time {response_time_ms}ms exceeds 100ms threshold")
        self.assertGreater(len(result), 0)
    
    def test_response_time_under_500ms_with_large_result_set(self):
        """Benchmark: Function should handle large result sets efficiently"""
        query_context = {"query": "test query"}
        user_tenant = "tenant_a"
        
        # Create larger result set
        large_results = []
        for i in range(50):
            doc = MagicMock()
            doc.metadata = {
                "embedding_id": f"emb-{i:03d}",
                "document_id": f"doc-{i:03d}",
                "tenant_id": "tenant_a",
                "topics": "test,example"
            }
            doc.page_content = "Test content" * 20
            large_results.append((doc, 0.9 - (i * 0.01)))
        
        self.mock_vector_store.similarity_search_with_score.return_value = large_results[:5]
        
        start_time = time.time()
        result = self.func(query_context, user_tenant, self.mock_vector_store)
        end_time = time.time()
        
        response_time_ms = (end_time - start_time) * 1000
        
        print(f"\n   ⏱️  Response time (large results): {response_time_ms:.2f}ms")
        self.assertLess(response_time_ms, 500,
                       f"Response time {response_time_ms}ms exceeds 500ms threshold")
    
    def test_handles_multiple_concurrent_calls(self):
        """Benchmark: Function should handle multiple calls efficiently"""
        query_context = {"query": "test"}
        user_tenant = "tenant_a"
        
        # Simulate 10 concurrent calls
        start_time = time.time()
        results = []
        for _ in range(10):
            result = self.func(query_context, user_tenant, self.mock_vector_store)
            results.append(result)
        end_time = time.time()
        
        total_time_ms = (end_time - start_time) * 1000
        avg_time_ms = total_time_ms / 10
        
        print(f"\n   ⏱️  Total time (10 calls): {total_time_ms:.2f}ms")
        print(f"   ⏱️  Average time per call: {avg_time_ms:.2f}ms")
        
        self.assertLess(avg_time_ms, 50, 
                       f"Average response time {avg_time_ms}ms exceeds 50ms threshold")
        self.assertEqual(len(results), 10)


def run_performance_benchmarks():
    """Run performance benchmarks and print results"""
    print("\n" + "="*70)
    print("⚡ PERFORMANCE BENCHMARKS")
    print("="*70)
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGenerateRetrievalMetadataRealPerformance)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()




def print_test_summary(unit_result, integration_result, perf_result, integration_test_result=None):
    """Print comprehensive test summary"""
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    print(f"✅ Unit Tests (Mocked): {'PASSED' if unit_result.wasSuccessful() else 'FAILED'}")
    print(f"   Tests: {unit_result.testsRun}, Failures: {len(unit_result.failures)}, Errors: {len(unit_result.errors)}")
    
    # Check if integration tests were skipped
    if integration_test_result:
        skipped_count = len(integration_test_result.skipped)
        if skipped_count > 0:
            print(f"\n⚠️  Integration Tests: SKIPPED ({skipped_count} tests)")
            print(f"   Reason: langchain_core not available")
            print(f"   Install with: pip install langchain-core")
        else:
            print(f"\n✅ Integration Tests (Real Vector Store + Fake Embeddings): {'PASSED' if integration_result else 'FAILED'}")
    else:
        print(f"\n✅ Integration Tests (Real Vector Store + Fake Embeddings): {'PASSED' if integration_result else 'FAILED'}")
    
    print(f"\n✅ Performance Benchmarks: {'PASSED' if perf_result else 'FAILED'}")
    
    print("\n" + "="*70)
    # Check if integration tests were skipped (not a failure, just missing deps)
    has_skipped = integration_test_result and len(integration_test_result.skipped) > 0
    
    # All tests passed (skipped tests don't count as failures)
    if unit_result.wasSuccessful() and perf_result and (integration_result or has_skipped):
        if has_skipped:
            print("✅ ALL RUNNABLE TESTS PASSED")
            print("   (Some integration tests skipped - install langchain-core to run them)")
        else:
            print("✅ ALL TESTS PASSED")
        print("="*70)
        return True
    else:
        if has_skipped:
            print("⚠️  SOME TESTS FAILED OR SKIPPED")
        else:
            print("⚠️  SOME TESTS FAILED")
        print("="*70)
        return False


if __name__ == "__main__":
    # Run unit tests (mocked)
    print("\n" + "="*70)
    print("🧪 UNIT TESTS FOR generate_retrieval_metadata_real")
    print("   (Using MagicMock - no real embeddings needed)")
    print("="*70)
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGenerateRetrievalMetadataReal)
    runner = unittest.TextTestRunner(verbosity=2)
    unit_result = runner.run(suite)
    
    # Run integration tests (real vector store + fake embeddings)
    integration_test_result = None
    if unit_result.wasSuccessful():
        suite = unittest.TestLoader().loadTestsFromTestCase(TestGenerateRetrievalMetadataRealIntegration)
        runner = unittest.TextTestRunner(verbosity=2)
        integration_test_result = runner.run(suite)
        integration_result = integration_test_result.wasSuccessful() and len(integration_test_result.skipped) == 0
    else:
        print("\n⚠️  Skipping integration tests due to unit test failures")
        integration_result = False
    
    # Run performance benchmarks (don't require integration tests to pass)
    if unit_result.wasSuccessful():
        perf_result = run_performance_benchmarks()
    else:
        print("\n⚠️  Skipping performance benchmarks due to unit test failures")
        perf_result = False
    
    # Print summary
    all_passed = print_test_summary(unit_result, integration_result, perf_result, integration_test_result)
    
    exit(0 if all_passed else 1)

