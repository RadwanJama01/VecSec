"""
Unit and Integration Tests for RAG Orchestrator

Tests the RAG orchestrator with vector store retrieval:
- Vector store integration
- Error handling when vector store is unavailable
- API contract preservation
- Logging behavior
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Imports after path setup (needed for test package layout)
from langchain_core.documents import Document  # noqa: E402

from src.sec_agent.rag_orchestrator import RAGOrchestrator  # noqa: E402


class TestRAGOrchestrator(unittest.TestCase):
    """Unit tests for RAG orchestrator with vector store"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_llm = MagicMock()
        self.mock_llm.invoke.return_value = MagicMock(content="Test answer")

        self.mock_prompt_template = MagicMock()
        self.mock_prompt_template.format_messages.return_value = [
            {"role": "user", "content": "test"}
        ]

        self.mock_vector_store = MagicMock()
        self.mock_vector_store.similarity_search.return_value = [
            Document(page_content="Test document", metadata={})
        ]

        # Mock graph to avoid LangGraph initialization issues
        self.mock_graph = MagicMock()
        self.mock_graph.invoke.return_value = {"answer": "Test answer"}

    @patch("src.sec_agent.rag_orchestrator.generate_retrieval_metadata")
    def test_uses_vector_store_retrieval(self, mock_gen):
        """Test that vector store retrieval is used"""
        mock_gen.return_value = [{"document_id": "doc-real-001", "retrieval_score": 0.85}]

        orchestrator = RAGOrchestrator(
            vector_store=self.mock_vector_store,
            llm=self.mock_llm,
            prompt_template=self.mock_prompt_template,
        )
        orchestrator.graph = self.mock_graph

        _ = orchestrator.rag_with_rlsa(
            user_id="user1",
            tenant_id="tenant_a",
            clearance="INTERNAL",
            query="test query",
            role="analyst",
        )

        # Should call retrieval function with vector store
        mock_gen.assert_called_once()
        call_args = mock_gen.call_args
        self.assertEqual(len(call_args[0]), 3)  # query_context, tenant_id, vector_store
        self.assertEqual(call_args[0][2], self.mock_vector_store)

    @patch("src.sec_agent.rag_orchestrator.generate_retrieval_metadata")
    def test_handles_vector_store_error(self, mock_gen):
        """Test that handles errors when vector retrieval fails"""
        mock_gen.side_effect = Exception("Vector store error")

        orchestrator = RAGOrchestrator(
            vector_store=self.mock_vector_store,
            llm=self.mock_llm,
            prompt_template=self.mock_prompt_template,
        )
        orchestrator.graph = self.mock_graph

        _ = orchestrator.rag_with_rlsa(
            user_id="user1",
            tenant_id="tenant_a",
            clearance="INTERNAL",
            query="test query",
            role="analyst",
        )

        # Should call retrieval function and handle error gracefully
        mock_gen.assert_called_once()

    def test_handles_none_vector_store(self):
        """Test that handles None vector store gracefully"""
        orchestrator = RAGOrchestrator(
            vector_store=None,  # None vector store
            llm=self.mock_llm,
            prompt_template=self.mock_prompt_template,
        )
        orchestrator.graph = self.mock_graph

        result = orchestrator.rag_with_rlsa(
            user_id="user1",
            tenant_id="tenant_a",
            clearance="INTERNAL",
            query="test query",
            role="analyst",
        )

        # Should complete without error, returning empty metadata
        self.assertIsInstance(result, bool)

    @patch("src.sec_agent.rag_orchestrator.logger")
    @patch("src.sec_agent.rag_orchestrator.generate_retrieval_metadata")
    def test_logs_retrieval_info(self, mock_gen, mock_logger):
        """Test that logging occurs when using vector retrieval"""
        mock_gen.return_value = [{"document_id": "doc-real-001", "retrieval_score": 0.85}]

        orchestrator = RAGOrchestrator(
            vector_store=self.mock_vector_store,
            llm=self.mock_llm,
            prompt_template=self.mock_prompt_template,
        )
        orchestrator.graph = self.mock_graph

        orchestrator.rag_with_rlsa(
            user_id="user1",
            tenant_id="tenant_a",
            clearance="INTERNAL",
            query="test query",
            role="analyst",
        )

        # Check that logging occurred
        self.assertTrue(mock_logger.debug.called or mock_logger.info.called)

    def test_api_contract_unchanged(self):
        """Test that API contract (rag_with_rlsa signature) is unchanged"""
        orchestrator = RAGOrchestrator(
            vector_store=self.mock_vector_store,
            llm=self.mock_llm,
            prompt_template=self.mock_prompt_template,
        )
        orchestrator.graph = self.mock_graph

        # API should accept same parameters
        result = orchestrator.rag_with_rlsa(
            user_id="user1",
            tenant_id="tenant_a",
            clearance="INTERNAL",
            query="test query",
            role="analyst",
        )

        # Should return boolean (True = allowed, False = blocked)
        self.assertIsInstance(result, bool)


class TestRAGOrchestratorIntegration(unittest.TestCase):
    """Integration tests with real vector store"""

    def setUp(self):
        """Set up integration test fixtures"""
        try:
            from langchain_core.embeddings import FakeEmbeddings
            from langchain_core.vectorstores import InMemoryVectorStore

            self.InMemoryVectorStore = InMemoryVectorStore
            # FakeEmbeddings requires size parameter in newer versions
            self.fake_embeddings = FakeEmbeddings(size=384)  # Standard embedding size
            self.vector_store_available = True
        except ImportError:
            self.vector_store_available = False
            self.skipTest("InMemoryVectorStore not available")

    @unittest.skipIf(not hasattr(unittest.TestCase, "setUp"), "Skip if setUp failed")
    def test_integration_real_retrieval_with_inmemory_store(self):
        """Integration test: Real retrieval with InMemoryVectorStore"""
        if not self.vector_store_available:
            self.skipTest("Vector store not available")

        # Create real vector store with test documents
        # Handle different API versions (embedding_function vs embedding vs positional)
        try:
            vector_store = self.InMemoryVectorStore(embedding_function=self.fake_embeddings)
        except TypeError:
            try:
                # Try with 'embedding' parameter (newer API)
                vector_store = self.InMemoryVectorStore(embedding=self.fake_embeddings)
            except TypeError:
                # Try as positional argument (some versions)
                vector_store = self.InMemoryVectorStore(self.fake_embeddings)

        test_docs = [
            Document(
                page_content="Security protocols for access control",
                metadata={
                    "document_id": "doc-integration-001",
                    "embedding_id": "emb-integration-001",
                    "tenant_id": "test_tenant",
                    "sensitivity": "INTERNAL",
                    "topics": "security,access_control",
                },
            ),
            Document(
                page_content="Financial data and budget information",
                metadata={
                    "document_id": "doc-integration-002",
                    "embedding_id": "emb-integration-002",
                    "tenant_id": "test_tenant",
                    "sensitivity": "CONFIDENTIAL",
                    "topics": "finance,budget",
                },
            ),
        ]
        vector_store.add_documents(test_docs)

        # Create orchestrator with real vector store
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Test answer")
        mock_prompt = MagicMock()
        mock_prompt.format_messages.return_value = [{"role": "user", "content": "test"}]
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {"answer": "Test answer"}

        orchestrator = RAGOrchestrator(
            vector_store=vector_store, llm=mock_llm, prompt_template=mock_prompt
        )
        orchestrator.graph = mock_graph

        # Execute query
        result = orchestrator.rag_with_rlsa(
            user_id="user1",
            tenant_id="test_tenant",
            clearance="INTERNAL",
            query="security protocols",
            role="analyst",
        )

        # Should complete successfully
        self.assertIsInstance(result, bool)

    @unittest.skipIf(not hasattr(unittest.TestCase, "setUp"), "Skip if setUp failed")
    def test_integration_handles_none_vector_store(self):
        """Integration test: Handles None vector store gracefully"""
        if not self.vector_store_available:
            self.skipTest("Vector store not available")

        # Create orchestrator with None vector store
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Test answer")
        mock_prompt = MagicMock()
        mock_prompt.format_messages.return_value = [{"role": "user", "content": "test"}]
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {"answer": "Test answer"}

        orchestrator = RAGOrchestrator(
            vector_store=None,  # None vector store
            llm=mock_llm,
            prompt_template=mock_prompt,
        )
        orchestrator.graph = mock_graph

        # Execute query - should handle None vector store gracefully
        result = orchestrator.rag_with_rlsa(
            user_id="user1",
            tenant_id="test_tenant",
            clearance="INTERNAL",
            query="test query",
            role="analyst",
        )

        # Should complete successfully with empty metadata
        self.assertIsInstance(result, bool)


class ColoredTestResult(unittest.TextTestResult):
    """Custom test result with visual indicators"""

    def addSuccess(self, test):
        super().addSuccess(test)
        if self.showAll:
            self.stream.writeln("✅ PASSED")
        elif self.dots:
            self.stream.write("✅")
            self.stream.flush()

    def addError(self, test, err):
        super().addError(test, err)
        if self.showAll:
            self.stream.writeln("❌ ERROR")
        elif self.dots:
            self.stream.write("❌")
            self.stream.flush()

    def addFailure(self, test, err):
        super().addFailure(test, err)
        if self.showAll:
            self.stream.writeln("❌ FAILED")
        elif self.dots:
            self.stream.write("❌")
            self.stream.flush()

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        if self.showAll:
            self.stream.writeln(f"⏭️  SKIPPED: {reason}")
        elif self.dots:
            self.stream.write("⏭️")
            self.stream.flush()


class ColoredTestRunner(unittest.TextTestRunner):
    """Custom test runner with visual indicators"""

    resultclass = ColoredTestResult


if __name__ == "__main__":
    import sys

    # Use colored runner for better visual feedback
    runner = ColoredTestRunner(
        verbosity=2,  # Show test names and results
        stream=sys.stdout,
    )

    # Discover and run tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    result = runner.run(suite)

    # Print summary with visual indicators
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)

    total = result.testsRun
    passed = total - len(result.failures) - len(result.errors) - len(result.skipped)
    failed = len(result.failures) + len(result.errors)
    skipped = len(result.skipped)

    print(f"Total tests: {total}")
    print(f"✅ Passed:  {passed}")
    print(f"❌ Failed:  {failed}")
    print(f"⏭️  Skipped: {skipped}")

    if result.failures:
        print("\n❌ FAILURES:")
        for test, _traceback in result.failures:
            print(f"   • {test}")

    if result.errors:
        print("\n❌ ERRORS:")
        for test, _traceback in result.errors:
            print(f"   • {test}")

    if result.skipped:
        print("\n⏭️  SKIPPED:")
        for test, reason in result.skipped:
            print(f"   • {test}: {reason}")

    print("=" * 70)

    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
