"""
Unit and Integration Tests for RAG Orchestrator Migration

Tests the migration from mock to real vector retrieval with:
- Feature flag behavior (USE_REAL_VECTOR_RETRIEVAL)
- Backward compatibility (fallback to mock)
- Migration logging
- API contract preservation
- Error handling
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


class TestRAGOrchestratorMigration(unittest.TestCase):
    """Unit tests for migration feature flag and backward compatibility"""

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

    def test_feature_flag_defaults_to_true(self):
        """Test that feature flag defaults to True (real retrieval enabled)"""
        # Reset env var to ensure default
        with patch.dict(os.environ, {}, clear=False):
            if "USE_REAL_VECTOR_RETRIEVAL" in os.environ:
                del os.environ["USE_REAL_VECTOR_RETRIEVAL"]
            # Reload config to get default
            import importlib

            import src.sec_agent.config as config_module

            importlib.reload(config_module)
            # Default should be True
            self.assertTrue(config_module.USE_REAL_VECTOR_RETRIEVAL)

    def test_feature_flag_respects_env_var(self):
        """Test that feature flag respects USE_REAL_VECTOR_RETRIEVAL env var"""
        # Test with flag enabled
        with patch.dict(os.environ, {"USE_REAL_VECTOR_RETRIEVAL": "true"}):
            import importlib

            import src.sec_agent.config as config_module

            importlib.reload(config_module)
            self.assertTrue(config_module.USE_REAL_VECTOR_RETRIEVAL)

        # Test with flag disabled
        with patch.dict(os.environ, {"USE_REAL_VECTOR_RETRIEVAL": "false"}):
            import importlib

            import src.sec_agent.config as config_module

            importlib.reload(config_module)
            self.assertFalse(config_module.USE_REAL_VECTOR_RETRIEVAL)

    @patch("src.sec_agent.rag_orchestrator.generate_retrieval_metadata_real")
    @patch("src.sec_agent.rag_orchestrator.generate_retrieval_metadata")
    @patch("src.sec_agent.rag_orchestrator.USE_REAL_VECTOR_RETRIEVAL", True)
    def test_uses_real_retrieval_when_flag_enabled(self, mock_mock_gen, mock_real_gen):
        """Test that real retrieval is used when feature flag is enabled"""
        mock_real_gen.return_value = [{"document_id": "doc-real-001", "retrieval_score": 0.85}]

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

        # Should call real retrieval function
        mock_real_gen.assert_called_once()
        mock_mock_gen.assert_not_called()

    @patch("src.sec_agent.rag_orchestrator.generate_retrieval_metadata_real")
    @patch("src.sec_agent.rag_orchestrator.generate_retrieval_metadata")
    @patch("src.sec_agent.rag_orchestrator.USE_REAL_VECTOR_RETRIEVAL", False)
    def test_uses_mock_retrieval_when_flag_disabled(self, mock_mock_gen, mock_real_gen):
        """Test that mock retrieval is used when feature flag is disabled"""
        mock_mock_gen.return_value = [{"document_id": "doc-mock-001", "retrieval_score": 0.9}]

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

        # Should call mock retrieval function
        mock_mock_gen.assert_called_once()
        mock_real_gen.assert_not_called()

    @patch("src.sec_agent.rag_orchestrator.generate_retrieval_metadata_real")
    @patch("src.sec_agent.rag_orchestrator.generate_retrieval_metadata")
    @patch("src.sec_agent.rag_orchestrator.USE_REAL_VECTOR_RETRIEVAL", True)
    def test_falls_back_to_mock_on_real_retrieval_error(self, mock_mock_gen, mock_real_gen):
        """Test that falls back to mock when real retrieval fails"""
        mock_real_gen.side_effect = Exception("Vector store error")
        mock_mock_gen.return_value = [{"document_id": "doc-fallback-001", "retrieval_score": 0.8}]

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

        # Should try real first, then fallback to mock
        mock_real_gen.assert_called_once()
        mock_mock_gen.assert_called_once()

    @patch("src.sec_agent.rag_orchestrator.generate_retrieval_metadata_real")
    @patch("src.sec_agent.rag_orchestrator.generate_retrieval_metadata")
    @patch("src.sec_agent.rag_orchestrator.USE_REAL_VECTOR_RETRIEVAL", True)
    def test_falls_back_to_mock_when_vector_store_none(self, mock_mock_gen, mock_real_gen):
        """Test that falls back to mock when vector_store is None"""
        mock_mock_gen.return_value = [{"document_id": "doc-fallback-001", "retrieval_score": 0.8}]

        orchestrator = RAGOrchestrator(
            vector_store=None,  # None vector store
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

        # Should use mock when vector_store is None
        mock_mock_gen.assert_called_once()
        mock_real_gen.assert_not_called()

    @patch("src.sec_agent.rag_orchestrator.logger")
    @patch("src.sec_agent.rag_orchestrator.generate_retrieval_metadata_real")
    @patch("src.sec_agent.rag_orchestrator.USE_REAL_VECTOR_RETRIEVAL", True)
    def test_logs_migration_info_when_using_real_retrieval(self, mock_real_gen, mock_logger):
        """Test that migration logging occurs when using real retrieval"""
        mock_real_gen.return_value = [{"document_id": "doc-real-001", "retrieval_score": 0.85}]

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

        # Check that migration logging occurred
        log_calls = [str(call) for call in mock_logger.info.call_args_list]
        migration_logged = any("real vector store" in str(call).lower() for call in log_calls)
        self.assertTrue(
            migration_logged, "Migration logging should occur when using real retrieval"
        )

    @patch("src.sec_agent.rag_orchestrator.logger")
    @patch("src.sec_agent.rag_orchestrator.generate_retrieval_metadata")
    @patch("src.sec_agent.rag_orchestrator.USE_REAL_VECTOR_RETRIEVAL", False)
    def test_logs_migration_info_when_using_mock_retrieval(self, mock_mock_gen, mock_logger):
        """Test that migration logging occurs when using mock retrieval"""
        mock_mock_gen.return_value = [{"document_id": "doc-mock-001", "retrieval_score": 0.9}]

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

        # Check that migration logging occurred (debug level for mock)
        log_calls = [str(call) for call in mock_logger.debug.call_args_list]
        migration_logged = any("mock metadata" in str(call).lower() for call in log_calls)
        self.assertTrue(
            migration_logged, "Migration logging should occur when using mock retrieval"
        )

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

    @patch("src.sec_agent.rag_orchestrator.generate_retrieval_metadata_real")
    @patch("src.sec_agent.rag_orchestrator.USE_REAL_VECTOR_RETRIEVAL", True)
    def test_passes_vector_store_to_real_retrieval(self, mock_real_gen):
        """Test that vector_store is passed correctly to real retrieval function"""
        mock_real_gen.return_value = [{"document_id": "doc-real-001", "retrieval_score": 0.85}]

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

        # Check that vector_store was passed as third argument
        call_args = mock_real_gen.call_args
        self.assertIsNotNone(call_args)
        args, kwargs = call_args
        self.assertEqual(len(args), 3)  # query_context, tenant_id, vector_store
        self.assertEqual(args[2], self.mock_vector_store)


class TestRAGOrchestratorMigrationIntegration(unittest.TestCase):
    """Integration tests for migration with real vector store"""

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

        with patch("src.sec_agent.rag_orchestrator.USE_REAL_VECTOR_RETRIEVAL", True):
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
    def test_integration_backward_compatibility_mock_fallback(self):
        """Integration test: Backward compatibility with mock fallback"""
        if not self.vector_store_available:
            self.skipTest("Vector store not available")

        # Create orchestrator with feature flag disabled
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Test answer")
        mock_prompt = MagicMock()
        mock_prompt.format_messages.return_value = [{"role": "user", "content": "test"}]
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {"answer": "Test answer"}

        with patch("src.sec_agent.rag_orchestrator.USE_REAL_VECTOR_RETRIEVAL", False):
            orchestrator = RAGOrchestrator(
                vector_store=None,  # Even with None, should work with mock
                llm=mock_llm,
                prompt_template=mock_prompt,
            )
            orchestrator.graph = mock_graph

            # Execute query - should use mock metadata generator
            result = orchestrator.rag_with_rlsa(
                user_id="user1",
                tenant_id="test_tenant",
                clearance="INTERNAL",
                query="test query",
                role="analyst",
            )

            # Should complete successfully with mock
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
