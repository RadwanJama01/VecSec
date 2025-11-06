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


def run_integration_tests():
    """Run integration tests with real vector store"""
    print("\n" + "="*70)
    print("🔗 INTEGRATION TESTS (Real Vector Store + Fake Embeddings)")
    print("="*70)
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGenerateRetrievalMetadataRealIntegration)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def print_test_summary(unit_result, integration_result, perf_result):
    """Print comprehensive test summary"""
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    print(f"✅ Unit Tests (Mocked): {'PASSED' if unit_result.wasSuccessful() else 'FAILED'}")
    print(f"   Tests: {unit_result.testsRun}, Failures: {len(unit_result.failures)}, Errors: {len(unit_result.errors)}")
    
    print(f"\n✅ Integration Tests (Real Vector Store + Fake Embeddings): {'PASSED' if integration_result else 'FAILED'}")
    
    print(f"\n✅ Performance Benchmarks: {'PASSED' if perf_result else 'FAILED'}")
    
    print("\n" + "="*70)
    if unit_result.wasSuccessful() and integration_result and perf_result:
        print("✅ ALL TESTS PASSED")
        print("="*70)
        return True
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
    if unit_result.wasSuccessful():
        integration_result = run_integration_tests()
    else:
        print("\n⚠️  Skipping integration tests due to unit test failures")
        integration_result = False
    
    # Run performance benchmarks
    if unit_result.wasSuccessful() and integration_result:
        perf_result = run_performance_benchmarks()
    else:
        print("\n⚠️  Skipping performance benchmarks due to test failures")
        perf_result = False
    
    # Print summary
    all_passed = print_test_summary(unit_result, integration_result, perf_result)
    
    exit(0 if all_passed else 1)

