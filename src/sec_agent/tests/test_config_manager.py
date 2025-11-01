"""
VecSec Config Functional Diagnostic
Tests real runtime behavior of config.py subsystems
"""

import os
import sys
import traceback
import importlib
from pathlib import Path

# Add repo root to Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

print("🚀 Starting VecSec Config Functional Diagnostics\n")


def reset_env():
    """Reset environment variables to safe defaults"""
    os.environ["USE_CHROMA"] = "false"
    os.environ.pop("CHROMA_PATH", None)
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ["METRICS_PORT"] = "8080"


# ------------------------------------------------------------
# 1️⃣ Vector Store Functional Test
# ------------------------------------------------------------
def test_vector_store():
    """Test vector store initialization and functionality"""
    print("🧠 Testing Vector Store behavior...")
    reset_env()
    
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_core.documents import Document

    try:
        from src.sec_agent import config
        
        # Create proper mock embeddings class
        class MockEmbeddings:
            def embed_documents(self, texts):
                return [[0.1] * 384 for _ in texts]
            
            def embed_query(self, text):
                return [0.1] * 384
        
        embeddings = MockEmbeddings()
        
        # Test vector store initialization
        store = config.initialize_vector_store(embeddings)
        assert isinstance(store, InMemoryVectorStore), f"Expected InMemoryVectorStore, got {type(store)}"
        print("✅ VectorStore initialized as InMemoryVectorStore")
        
        # Test sample document loading
        config.initialize_sample_documents(store)
        print("✅ Sample documents loaded")

        # Functional test: similarity_search
        results = store.similarity_search("LangChain", k=1)
        assert len(results) > 0, "Similarity search should return results"
        assert hasattr(results[0], 'page_content'), "Result should have page_content"
        print(f"🔍 Search result: {results[0].page_content[:60]}...")
        print("✅ VectorStore functional with InMemory fallback")
        
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ VectorStore functional test failed: {e}")
    print()


# ------------------------------------------------------------
# 2️⃣ Metrics Exporter Test
# ------------------------------------------------------------
def test_metrics_exporter():
    """Test metrics exporter initialization and state"""
    print("📊 Testing Metrics Exporter...")
    reset_env()
    
    try:
        # Force reload to get fresh state
        if 'src.sec_agent.config' in sys.modules:
            importlib.reload(sys.modules['src.sec_agent.config'])
        
        from src.sec_agent import config
        
        # Check metrics flag exists and is boolean
        assert hasattr(config, 'METRICS_ENABLED'), "METRICS_ENABLED should exist"
        assert isinstance(config.METRICS_ENABLED, bool), f"METRICS_ENABLED should be bool, got {type(config.METRICS_ENABLED)}"
        print(f"METRICS_ENABLED = {config.METRICS_ENABLED}")
        
        # Check Chroma availability flag
        assert hasattr(config, 'CHROMA_AVAILABLE'), "CHROMA_AVAILABLE should exist"
        assert isinstance(config.CHROMA_AVAILABLE, bool), f"CHROMA_AVAILABLE should be bool, got {type(config.CHROMA_AVAILABLE)}"
        print(f"CHROMA_AVAILABLE = {config.CHROMA_AVAILABLE}")
        
        print("✅ Metrics exporter flags read successfully")
        
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Metrics exporter test failed: {e}")
    print()


# ------------------------------------------------------------
# 3️⃣ Prompt Template Test
# ------------------------------------------------------------
def test_prompt_template():
    """Test prompt template creation and rendering"""
    print("📝 Testing Prompt Template...")
    
    try:
        from src.sec_agent import config
        
        # Test template creation
        template = config.create_rag_prompt_template()
        assert template is not None, "Template should not be None"
        print(f"✅ Template created: {type(template).__name__}")
        
        # Test template rendering
        rendered = template.format(context="RAG is a technique", question="What is RAG?")
        assert "Context:" in rendered, "Rendered template should contain 'Context:'"
        assert "Question:" in rendered, "Rendered template should contain 'Question:'"
        assert "RAG is a technique" in rendered, "Rendered template should contain context"
        assert "What is RAG?" in rendered, "Rendered template should contain question"
        
        print("✅ Prompt template renders correctly:")
        print("   " + "\n   ".join(rendered.split("\n")[:3]) + "...")
        
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Prompt template test failed: {e}")
    print()


# ------------------------------------------------------------
# 4️⃣ Chaos Env Validation Test
# ------------------------------------------------------------
def test_env_validation_behavior():
    """Test behavior with invalid environment variables"""
    print("🔥 Chaos Env Test...")
    reset_env()
    
    # Set invalid values
    os.environ["USE_CHROMA"] = "maybe"  # Invalid boolean
    os.environ["METRICS_PORT"] = "abc"  # Invalid int
    
    try:
        from src.sec_agent import config
        
        # Test validation function
        print("Testing validate_env_vars() with invalid values...")
        try:
            config.validate_env_vars()
            print("⚠️ Validation passed but should have failed with invalid values")
            print("   Expected: ValueError for invalid USE_CHROMA and METRICS_PORT")
        except ValueError as e:
            print("✅ Validation correctly raised ValueError:")
            print(f"   {str(e)[:100]}...")
            assert "Invalid boolean value" in str(e) or "Invalid integer value" in str(e), \
                "Error should mention invalid boolean or integer"
        except Exception as e:
            print(f"⚠️ Unexpected exception: {e}")
        
        # Test that invalid USE_CHROMA is parsed safely
        use_chroma = config._parse_bool(os.getenv("USE_CHROMA", "false"))
        print(f"USE_CHROMA='maybe' parsed as: {use_chroma} (should be False)")
        assert use_chroma == False, "Invalid boolean should default to False"
        print("✅ Invalid USE_CHROMA='maybe' correctly treated as False")
        
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Chaos env test failed: {e}")
    print()


# ------------------------------------------------------------
# 5️⃣ Path Configuration Test
# ------------------------------------------------------------
def test_chroma_path_configuration():
    """Test if ChromaDB path is configurable"""
    print("📁 Testing Chroma Path Configuration...")
    reset_env()
    
    try:
        from src.sec_agent import config
        from langchain_core.vectorstores import InMemoryVectorStore
        
        # Create mock embeddings
        class MockEmbeddings:
            def embed_documents(self, texts):
                return [[0.1] * 384 for _ in texts]
            def embed_query(self, text):
                return [0.1] * 384
        
        # Test with custom CHROMA_PATH
        test_path = "/tmp/test_chroma_path"
        os.environ["CHROMA_PATH"] = test_path
        os.environ["USE_CHROMA"] = "false"  # Use InMemory to avoid actual Chroma init
        
        embeddings = MockEmbeddings()
        store = config.initialize_vector_store(embeddings)
        
        # Check that CHROMA_PATH is read from env (even if not used due to USE_CHROMA=false)
        chroma_path_from_env = os.getenv("CHROMA_PATH", "./chroma_db")
        print(f"CHROMA_PATH env var = {chroma_path_from_env}")
        
        # Test that config schema has CHROMA_PATH
        assert "CHROMA_PATH" in config.CONFIG_SCHEMA, "CONFIG_SCHEMA should have CHROMA_PATH"
        assert config.CONFIG_SCHEMA["CHROMA_PATH"]["type"] == str, "CHROMA_PATH should be string type"
        print(f"✅ CONFIG_SCHEMA includes CHROMA_PATH with default: {config.CONFIG_SCHEMA['CHROMA_PATH']['default']}")
        
        # Test path validation
        print("Testing path validation...")
        config.validate_env_vars()  # Should not raise if path is valid or can be created
        print("✅ Path validation passed")
        
        # Test that default path is in schema
        default_path = config.CONFIG_SCHEMA["CHROMA_PATH"]["default"]
        assert default_path == "./chroma_db", f"Default path should be './chroma_db', got '{default_path}'"
        print("✅ Default path configured correctly in schema")
        
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Path configuration test failed: {e}")
    print()


# ------------------------------------------------------------
# 6️⃣ Sample Documents Test
# ------------------------------------------------------------
def test_sample_documents():
    """Test sample document loading"""
    print("📄 Testing Sample Documents...")
    reset_env()
    
    try:
        from langchain_core.vectorstores import InMemoryVectorStore
        from src.sec_agent import config
        
        class MockEmbeddings:
            def embed_documents(self, texts):
                return [[0.1] * 384 for _ in texts]
            def embed_query(self, text):
                return [0.1] * 384
        
        embeddings = MockEmbeddings()
        store = InMemoryVectorStore(embeddings)
        
        # Load sample docs
        config.initialize_sample_documents(store)
        print("✅ Sample documents loaded")
        
        # Try to verify docs were added (search for known content)
        results = store.similarity_search("RAG", k=5)
        print(f"🔍 Found {len(results)} documents matching 'RAG'")
        
        # Check if we get at least one result
        assert len(results) > 0, "Should find documents containing 'RAG'"
        
        # Verify content
        found_rag_content = any("RAG" in doc.page_content for doc in results)
        assert found_rag_content, "Should find document about RAG"
        print("✅ Sample documents contain expected content")
        
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Sample documents test failed: {e}")
    print()


# ------------------------------------------------------------
# Run All Tests
# ------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("VecSec Config Functional Diagnostics")
    print("=" * 60)
    print()
    
    test_vector_store()
    test_metrics_exporter()
    test_prompt_template()
    test_chroma_path_configuration()
    test_sample_documents()
    test_env_validation_behavior()
    
    print("=" * 60)
    print("🏁 Diagnostics complete.")
    print("=" * 60)
