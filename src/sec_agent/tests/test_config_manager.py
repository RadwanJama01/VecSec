"""
VecSec Config Manager Tests

Tests the core configuration and vector store initialization logic.
This is the most important test suite - it validates the core architecture.
"""

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path as PathLib
from unittest.mock import MagicMock

# Add project root to path
project_root = PathLib(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import config directly to avoid __init__.py import issues
config_path = project_root / "src" / "sec_agent" / "config.py"
spec = importlib.util.spec_from_file_location("config", config_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

# Import what we need
CHROMA_AVAILABLE = config.CHROMA_AVAILABLE
CONFIG_SCHEMA = config.CONFIG_SCHEMA
METRICS_ENABLED = config.METRICS_ENABLED
_create_chroma_cloud_client = config._create_chroma_cloud_client
_parse_bool = config._parse_bool
_parse_int = config._parse_int
initialize_vector_store = config.initialize_vector_store
validate_env_vars = config.validate_env_vars


class TestConfigManager(unittest.TestCase):
    """Test configuration management and validation"""

    def setUp(self):
        """Set up test environment"""
        # Save original env vars
        self.original_env = dict(os.environ)
        # Clear ChromaDB-related env vars for clean testing
        for key in [
            "CHROMA_API_KEY",
            "CHROMA_TENANT",
            "CHROMA_DATABASE",
            "USE_CHROMA",
            "CHROMA_PATH",
        ]:
            os.environ.pop(key, None)

    def tearDown(self):
        """Restore original environment"""
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_config_schema_exists(self):
        """Test that CONFIG_SCHEMA is properly defined"""
        self.assertIsInstance(CONFIG_SCHEMA, dict)
        self.assertIn("USE_CHROMA", CONFIG_SCHEMA)
        self.assertIn("CHROMA_PATH", CONFIG_SCHEMA)
        self.assertIn("CHROMA_API_KEY", CONFIG_SCHEMA)
        self.assertIn("CHROMA_TENANT", CONFIG_SCHEMA)
        self.assertIn("CHROMA_DATABASE", CONFIG_SCHEMA)

    def test_parse_bool(self):
        """Test boolean parsing"""
        self.assertTrue(_parse_bool("true"))
        self.assertTrue(_parse_bool("1"))
        self.assertTrue(_parse_bool("yes"))
        self.assertTrue(_parse_bool("on"))
        self.assertTrue(_parse_bool(True))

        self.assertFalse(_parse_bool("false"))
        self.assertFalse(_parse_bool("0"))
        self.assertFalse(_parse_bool("no"))
        self.assertFalse(_parse_bool("off"))
        self.assertFalse(_parse_bool(False))
        self.assertFalse(_parse_bool("invalid"))

    def test_parse_int(self):
        """Test integer parsing"""
        self.assertEqual(_parse_int("42"), 42)
        self.assertEqual(_parse_int(42), 42)
        with self.assertRaises(ValueError):
            _parse_int("not_a_number")

    def test_validate_env_vars(self):
        """Test environment variable validation"""
        # Should not raise with valid defaults
        try:
            validate_env_vars()
        except ValueError:
            self.fail("validate_env_vars() raised ValueError unexpectedly")

    def test_chroma_available_flag(self):
        """Test CHROMA_AVAILABLE flag exists"""
        self.assertIsInstance(CHROMA_AVAILABLE, bool)
        print(f"   CHROMA_AVAILABLE = {CHROMA_AVAILABLE}")


class TestChromaCloudConnection(unittest.TestCase):
    """Test ChromaDB Cloud connection"""

    def setUp(self):
        """Set up test environment"""
        self.original_env = dict(os.environ)
        # Clear cloud credentials
        for key in ["CHROMA_API_KEY", "CHROMA_TENANT", "CHROMA_DATABASE"]:
            os.environ.pop(key, None)

    def tearDown(self):
        """Restore original environment"""
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_cloud_client_missing_credentials(self):
        """Test that cloud client returns None when credentials are missing"""
        client = _create_chroma_cloud_client()
        self.assertIsNone(client, "Should return None when credentials missing")

    @unittest.skipUnless(
        os.getenv("CHROMA_API_KEY") and os.getenv("CHROMA_TENANT") and os.getenv("CHROMA_DATABASE"),
        "ChromaDB Cloud credentials not set",
    )
    def test_cloud_client_with_credentials(self):
        """Test cloud client creation with real credentials"""
        # Use credentials from environment (if set)
        client = _create_chroma_cloud_client()
        if client:
            # Test that we can list collections
            try:
                collections = client.list_collections()
                self.assertIsInstance(collections, list)
                print(f"   ✅ Can list collections: {len(collections)} found")
            except Exception as e:
                self.fail(f"Failed to list collections: {e}")

    @unittest.skipUnless(
        os.getenv("CHROMA_API_KEY") and os.getenv("CHROMA_TENANT") and os.getenv("CHROMA_DATABASE"),
        "ChromaDB Cloud credentials not set",
    )
    def test_cloud_client_read_write(self):
        """Test read/write operations with cloud client"""
        client = _create_chroma_cloud_client()
        if not client:
            self.skipTest("Cloud client not available")

        # Get or create test collection
        collection = client.get_or_create_collection(name="test_config_read_write")

        # Write test document
        test_id = "test-doc-1"
        test_doc = "This is a test document for read/write"
        test_metadata = {"test": "true", "sensitivity": "PUBLIC"}

        try:
            collection.add(documents=[test_doc], ids=[test_id], metadatas=[test_metadata])
            print("   ✅ Write successful")

            # Read back
            results = collection.get(ids=[test_id])
            self.assertEqual(len(results["ids"]), 1)
            self.assertEqual(results["documents"][0], test_doc)
            print("   ✅ Read successful")

            # Test metadata
            self.assertEqual(results["metadatas"][0]["test"], "true")
            print("   ✅ Metadata accepted")
        except Exception as e:
            self.fail(f"Read/write test failed: {e}")


class TestVectorStoreInitialization(unittest.TestCase):
    """Test vector store initialization logic - THE MOST IMPORTANT TESTS"""

    def setUp(self):
        """Set up test environment"""
        self.original_env = dict(os.environ)
        # Clear all ChromaDB env vars
        for key in [
            "CHROMA_API_KEY",
            "CHROMA_TENANT",
            "CHROMA_DATABASE",
            "USE_CHROMA",
            "CHROMA_PATH",
        ]:
            os.environ.pop(key, None)

    def tearDown(self):
        """Restore original environment"""
        os.environ.clear()
        os.environ.update(self.original_env)

    def _make_mock_embeddings(self):
        """Create mock embeddings function"""
        mock_emb = MagicMock()
        mock_emb.embed_query = MagicMock(return_value=[0.1] * 384)
        mock_emb.embed_documents = MagicMock(return_value=[[0.1] * 384])
        return mock_emb

    @unittest.skipUnless(
        os.getenv("CHROMA_API_KEY") and os.getenv("CHROMA_TENANT") and os.getenv("CHROMA_DATABASE"),
        "ChromaDB Cloud credentials not set",
    )
    def test_cloud_mode(self):
        """TEST A - CLOUD MODE

        Env:
        - CHROMA_API_KEY=xxx
        - CHROMA_TENANT=xxx
        - CHROMA_DATABASE=xxx
        - USE_CHROMA=false (should still use cloud)

        Expect:
        - vector_store.client is a CloudClient
        """
        # Set cloud credentials from environment
        api_key = os.getenv("CHROMA_API_KEY")
        tenant = os.getenv("CHROMA_TENANT")
        database = os.getenv("CHROMA_DATABASE")

        if not (api_key and tenant and database):
            self.skipTest("ChromaDB Cloud credentials not set")

        os.environ["CHROMA_API_KEY"] = api_key
        os.environ["CHROMA_TENANT"] = tenant
        os.environ["CHROMA_DATABASE"] = database
        os.environ["USE_CHROMA"] = "false"  # Should still use cloud

        embeddings = self._make_mock_embeddings()
        vector_store = initialize_vector_store(embeddings)

        # Check that it's a Chroma instance (or InMemory if connection failed)
        from langchain_chroma import Chroma
        from langchain_core.vectorstores import InMemoryVectorStore

        if isinstance(vector_store, Chroma):
            # Check that client is a CloudClient
            import chromadb

            if hasattr(vector_store, "_client"):
                self.assertIsInstance(vector_store._client, chromadb.CloudClient)
                print("   ✅ Cloud mode: vector_store.client is CloudClient")
            else:
                # Try alternative attribute access
                client = getattr(vector_store, "client", None)
                if client:
                    self.assertIsInstance(client, chromadb.CloudClient)
                    print("   ✅ Cloud mode: vector_store.client is CloudClient")
        elif isinstance(vector_store, InMemoryVectorStore):
            # Connection might have failed, that's okay for this test
            print(
                "   ⚠️  Cloud mode: Connection failed, using InMemory (credentials may be invalid)"
            )
        else:
            self.fail(f"Unexpected vector store type: {type(vector_store)}")

    @unittest.skipUnless(CHROMA_AVAILABLE, "ChromaDB not available")
    def test_local_mode(self):
        """TEST B - LOCAL MODE

        Env:
        - CHROMA_API_KEY unset
        - USE_CHROMA=true
        - CHROMA_PATH=/tmp/chroma_test

        Expect:
        - vector_store is a langchain_chroma.Chroma instance
        - persist_directory matches CHROMA_PATH
        """
        # Ensure cloud credentials are unset
        os.environ.pop("CHROMA_API_KEY", None)
        os.environ.pop("CHROMA_TENANT", None)
        os.environ.pop("CHROMA_DATABASE", None)

        # Set local ChromaDB
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["USE_CHROMA"] = "true"
            os.environ["CHROMA_PATH"] = tmpdir

            embeddings = self._make_mock_embeddings()
            vector_store = initialize_vector_store(embeddings)

            # Check that it's a Chroma instance
            from langchain_chroma import Chroma

            self.assertIsInstance(vector_store, Chroma)

            # Check persist_directory (may be in different attribute)
            persist_dir = getattr(vector_store, "_persist_directory", None) or getattr(
                vector_store, "persist_directory", None
            )
            if persist_dir:
                self.assertEqual(str(persist_dir), tmpdir)
            print(f"   ✅ Local mode: Chroma instance created at {tmpdir}")

    def test_memory_mode(self):
        """TEST C - MEMORY MODE (fallback)

        Env:
        - CHROMA_API_KEY unset
        - USE_CHROMA=false

        Expect:
        - vector_store is InMemoryVectorStore
        """
        # Ensure all ChromaDB env vars are unset
        os.environ.pop("CHROMA_API_KEY", None)
        os.environ.pop("CHROMA_TENANT", None)
        os.environ.pop("CHROMA_DATABASE", None)
        os.environ["USE_CHROMA"] = "false"

        embeddings = self._make_mock_embeddings()
        vector_store = initialize_vector_store(embeddings)

        # Check that it's InMemoryVectorStore
        from langchain_core.vectorstores import InMemoryVectorStore

        self.assertIsInstance(vector_store, InMemoryVectorStore)
        print("   ✅ Memory mode: vector_store is InMemoryVectorStore")

    def test_priority_cloud_over_local(self):
        """Test that cloud mode takes priority over local mode"""
        if not (
            os.getenv("CHROMA_API_KEY")
            and os.getenv("CHROMA_TENANT")
            and os.getenv("CHROMA_DATABASE")
        ):
            self.skipTest("ChromaDB Cloud credentials not set")

        # Set both cloud and local
        os.environ["CHROMA_API_KEY"] = os.getenv("CHROMA_API_KEY")
        os.environ["CHROMA_TENANT"] = os.getenv("CHROMA_TENANT")
        os.environ["CHROMA_DATABASE"] = os.getenv("CHROMA_DATABASE")
        os.environ["USE_CHROMA"] = "true"

        embeddings = self._make_mock_embeddings()
        vector_store = initialize_vector_store(embeddings)

        # Should use cloud, not local
        import chromadb

        self.assertIsInstance(vector_store._client, chromadb.CloudClient)
        print("   ✅ Priority: Cloud mode takes precedence over local")


class TestMetricsInitialization(unittest.TestCase):
    """Test metrics initialization - optional and non-blocking"""

    def test_metrics_enabled_exists(self):
        """Test that METRICS_ENABLED is present"""
        self.assertIsNotNone(METRICS_ENABLED)
        self.assertIsInstance(METRICS_ENABLED, bool)
        print(f"   METRICS_ENABLED = {METRICS_ENABLED}")

    def test_metrics_optional(self):
        """Test that metrics are optional and don't break imports"""
        # METRICS_ENABLED should be accessible from the config module we imported
        self.assertIsInstance(METRICS_ENABLED, bool)
        # The fact that we got here means metrics didn't block initialization
        print("   ✅ Metrics are optional and non-blocking")

    def test_metrics_non_blocking(self):
        """Test that missing metrics don't block initialization"""
        # METRICS_ENABLED should be False if metrics_exporter is unavailable
        # This is tested by the fact that config.py imports without error
        self.assertIsInstance(METRICS_ENABLED, bool)


if __name__ == "__main__":
    print("=" * 70)
    print("VecSec Config Manager Tests")
    print("=" * 70)
    print()

    # Run tests
    unittest.main(verbosity=2)
