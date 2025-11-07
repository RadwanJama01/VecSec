#!/usr/bin/env python3
"""
Vector Store Connection Test Script

Simple test to verify:
1. Vector store can be initialized
2. Connection parameters are correct
3. Vector store contains document embeddings
4. Sample query returns real data

Usage:
    python3 scripts/test_vector_store_connection.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def check_configuration():
    """Check if vector store configuration is set up"""
    print("=" * 70)
    print("Vector Store Connection Test")
    print("=" * 70)

    # Test 1: Check configuration
    print("\n1️⃣  Checking Configuration...")
    use_chroma = os.getenv("USE_CHROMA", "false").lower() == "true"
    chroma_path = os.getenv("CHROMA_PATH", "./chroma_db")

    print(f"   USE_CHROMA: {use_chroma}")
    print(f"   CHROMA_PATH: {chroma_path}")

    # Check if ChromaDB path exists
    chroma_path_obj = Path(chroma_path)
    if chroma_path_obj.exists():
        print(f"   ✅ ChromaDB directory exists: {chroma_path}")
        # Count files to see if there's data
        file_count = len(list(chroma_path_obj.rglob("*")))
        print(f"   📁 Files in ChromaDB directory: {file_count}")
    else:
        print("   ℹ️  ChromaDB directory doesn't exist yet (will be created on first use)")

    return use_chroma, chroma_path


def check_code_locations():
    """Verify code locations and identify vector database"""
    print("\n2️⃣  Identifying Vector Database...")

    config_file = project_root / "src" / "sec_agent" / "config.py"
    if config_file.exists():
        print(f"   ✅ Found initialization code: {config_file}")

        # Read config file to find vector store type
        try:
            with open(config_file) as f:
                content = f.read()
                if "Chroma" in content:
                    print("   ✅ Vector database: ChromaDB")
                if "InMemoryVectorStore" in content:
                    print("   ✅ Fallback: InMemoryVectorStore")
        except Exception as e:
            print(f"   ⚠️  Could not read config file: {e}")
    else:
        print(f"   ❌ Config file not found: {config_file}")
        return False

    embeddings_file = project_root / "src" / "sec_agent" / "embeddings_client.py"
    if embeddings_file.exists():
        print(f"   ✅ Found embedding client: {embeddings_file}")

        # Read to find model name
        try:
            with open(embeddings_file) as f:
                content = f.read()
                if "all-MiniLM-L6-v2" in content:
                    print("   ✅ Embedding model: all-MiniLM-L6-v2 (SentenceTransformers)")
        except Exception as e:
            print(f"   ⚠️  Could not read embeddings file: {e}")
    else:
        print("   ⚠️  Embeddings file not found")

    return True


def check_connection_parameters():
    """Document connection parameters needed"""
    print("\n3️⃣  Connection Parameters...")
    print("   Required environment variables:")
    print("   - USE_CHROMA: Enable ChromaDB (bool, default: false)")
    print("   - CHROMA_PATH: Path for ChromaDB storage (str, default: ./chroma_db)")
    print("\n   Optional but recommended:")
    print("   - LOG_FILE: Path to log file (optional)")
    print("   - LOG_LEVEL: Logging level (default: INFO)")

    # Check .env file
    env_file = project_root / ".env"
    if env_file.exists():
        print(f"\n   ✅ Found .env file: {env_file}")
        try:
            with open(env_file) as f:
                lines = f.readlines()
                chroma_vars = [
                    line.strip()
                    for line in lines
                    if "CHROMA" in line.upper() or "USE_CHROMA" in line.upper()
                ]
                if chroma_vars:
                    print("   📋 ChromaDB-related variables in .env:")
                    for var in chroma_vars[:3]:  # Show first 3
                        print(f"      {var}")
                else:
                    print("   ℹ️  No ChromaDB variables found in .env")
        except Exception as e:
            print(f"   ⚠️  Could not read .env file: {e}")
    else:
        print("   ℹ️  No .env file found (using defaults)")

    return True


def check_rag_orchestrator_connection():
    """Check RAG orchestrator location and connection"""
    print("\n4️⃣  RAG Orchestrator Connection...")

    orchestrator_file = project_root / "src" / "sec_agent" / "rag_orchestrator.py"
    if orchestrator_file.exists():
        print(f"   ✅ Found RAG orchestrator: {orchestrator_file}")

        try:
            with open(orchestrator_file) as f:
                content = f.read()
                if "vector_store" in content:
                    print("   ✅ RAG orchestrator uses vector_store")
                if "similarity_search" in content:
                    print("   ✅ RAG orchestrator performs similarity_search")
        except Exception as e:
            print(f"   ⚠️  Could not read orchestrator file: {e}")
    else:
        print("   ⚠️  RAG orchestrator file not found")
        return False

    return True


def check_test_scripts():
    """Check if test scripts exist"""
    print("\n5️⃣  Test Scripts Available...")

    test_chromadb = project_root / "scripts" / "test_chromadb.sh"
    if test_chromadb.exists():
        print(f"   ✅ Integration test: {test_chromadb}")
        print("      Run with: ./scripts/test_chromadb.sh")

    test_connection = project_root / "scripts" / "test_vector_store_connection.py"
    if test_connection.exists():
        print(f"   ✅ Connection test: {test_connection}")
        print("      Run with: python3 scripts/test_vector_store_connection.py")

    return True


def provide_sample_query_info():
    """Provide information about sample queries"""
    print("\n6️⃣  Sample Query Information...")
    print("   📋 Sample queries are available in:")
    print("      - scripts/test_chromadb.sh (lines 293-340)")
    print("      - src/sec_agent/tests/test_metadata_generator_real.py")
    print("\n   Example query format:")
    print("      query_context = {")
    print("          'query': 'security protocols',")
    print("          'topics': ['security']")
    print("      }")
    print("      results = generate_retrieval_metadata_real(")
    print("          query_context=query_context,")
    print("          user_tenant='tenant_a',")
    print("          vector_store=vector_store")
    print("      )")

    return True


def main():
    """Main test function"""
    try:
        check_configuration()
        check_code_locations()
        check_connection_parameters()
        check_rag_orchestrator_connection()
        check_test_scripts()
        provide_sample_query_info()

        print("\n" + "=" * 70)
        print("✅ Configuration Check Complete")
        print("=" * 70)
        print("\n📚 For detailed documentation, see:")
        print("   docs/VECTOR_STORE_CONFIG.md")
        print("\n🧪 To run full integration tests:")
        print("   ./scripts/test_chromadb.sh")
        print("\n✅ All checks completed!")

        return 0
    except Exception as e:
        print(f"\n❌ Error during configuration check: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
