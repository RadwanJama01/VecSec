"""
VecSec ChromaDB Integration Functional Diagnostic
Tests real runtime behavior of Chroma initialization and fallbacks
"""

import os
import sys
import traceback
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

print("🚀 Starting VecSec ChromaDB Integration Diagnostics\n")
print("=" * 60)


def reset_env():
    os.environ.pop("USE_CHROMA", None)
    os.environ.pop("CHROMA_PATH", None)


def _make_mock_embeddings(dim: int = 384):
    class MockEmbeddings:
        def embed_documents(self, texts):
            return [[0.1] * dim for _ in texts]

        def embed_query(self, text):
            return [0.1] * dim

    return MockEmbeddings()


# ------------------------------------------------------------
# 1️⃣ InMemory Fallback when USE_CHROMA=false
# ------------------------------------------------------------
def test_inmemory_fallback_when_disabled():
    print("\n🔧 Testing InMemory fallback when USE_CHROMA=false...")
    reset_env()
    os.environ["USE_CHROMA"] = "false"

    try:
        from langchain_core.vectorstores import InMemoryVectorStore

        from src.sec_agent import config

        store = config.initialize_vector_store(_make_mock_embeddings())
        assert isinstance(store, InMemoryVectorStore), "Expected InMemory when USE_CHROMA=false"
        print("✅ InMemoryVectorStore used when disabled")

    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ------------------------------------------------------------
# 2️⃣ Chroma path handling and fallback
# ------------------------------------------------------------
def test_chroma_path_and_fallback_behavior():
    print("\n📁 Testing CHROMA_PATH handling and fallback...")
    reset_env()
    os.environ["USE_CHROMA"] = "true"
    tmp_path = "/tmp/vecsec_chroma_test"
    os.environ["CHROMA_PATH"] = tmp_path

    try:
        from langchain_core.vectorstores import InMemoryVectorStore

        from src.sec_agent import config

        # Ensure directory does not exist beforehand
        path_obj = Path(tmp_path)
        if path_obj.exists():
            # Try to clean up from previous runs
            try:
                for child in path_obj.glob("**/*"):
                    if child.is_file():
                        child.unlink()
                for child in sorted(path_obj.glob("**/*"), reverse=True):
                    if child.is_dir():
                        child.rmdir()
                path_obj.rmdir()
            except Exception:
                pass

        store = config.initialize_vector_store(_make_mock_embeddings())

        # If Chroma is available, the path should now exist; otherwise, we should have fallen back
        if getattr(config, "CHROMA_AVAILABLE", False):
            assert path_obj.exists(), "CHROMA_PATH directory should be created when using Chroma"
            print(f"✅ CHROMA_PATH exists: {tmp_path}")
        else:
            assert isinstance(store, InMemoryVectorStore), (
                "Should fall back to InMemory when Chroma unavailable"
            )
            print("✅ Fell back to InMemory when Chroma unavailable")

    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ------------------------------------------------------------
# 3️⃣ Sample documents work with active store
# ------------------------------------------------------------
def test_sample_documents_similarity_search():
    print("\n📄 Testing sample documents with similarity_search...")
    reset_env()
    os.environ["USE_CHROMA"] = "false"  # use InMemory to avoid dependency

    try:
        from src.sec_agent import config

        store = config.initialize_vector_store(_make_mock_embeddings())
        config.initialize_sample_documents(store)
        results = store.similarity_search("RAG", k=1)
        assert len(results) > 0, "Expected at least one result from sample docs"
        print("✅ similarity_search returns results with sample docs")

    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ------------------------------------------------------------
# 4️⃣ Optional: CHROMA_CLOUD_SETUP behavior (skipped if chromadb missing)
# ------------------------------------------------------------
def test_chroma_cloud_setup_optional():
    print("\n☁️  Testing CHROMA_CLOUD_SETUP (optional)...")
    reset_env()
    try:
        import importlib

        module = importlib.import_module("src.CHROMA_CLOUD_SETUP")
        setup_fn = getattr(module, "setup_chroma_cloud", None)
        get_coll = getattr(module, "get_collection", None)
        assert callable(setup_fn), "setup_chroma_cloud should exist"
        assert callable(get_coll), "get_collection should exist"

        client = setup_fn()
        assert client is not None, "setup_chroma_cloud should return a client"
        print("✅ CHROMA_CLOUD_SETUP available and returned a client")

    except ModuleNotFoundError:
        print("ℹ️  CHROMA_CLOUD_SETUP module not found; skipping optional test")
    except ImportError:
        print("ℹ️  chromadb not installed; skipping optional cloud setup test")
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Optional cloud setup test failed: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("VecSec ChromaDB Integration Diagnostics")
    print("=" * 60)

    test_inmemory_fallback_when_disabled()
    test_chroma_path_and_fallback_behavior()
    test_sample_documents_similarity_search()
    test_chroma_cloud_setup_optional()

    print("\n" + "=" * 60)
    print("🏁 ChromaDB Integration Diagnostics Complete")
    print("=" * 60)
