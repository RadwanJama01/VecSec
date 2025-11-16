#!/usr/bin/env python3
"""
Isolated ChromaDB Cloud Connection Test
Just tests connection - doesn't touch anything else
"""

import os
import chromadb
from dotenv import load_dotenv

# Load credentials from .env file
load_dotenv()

# Get credentials from environment (loaded from .env)
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")

print("=" * 70)
print("ChromaDB Cloud Connection Test")
print("=" * 70)
print()

# Check credentials are loaded
if not CHROMA_API_KEY or not CHROMA_TENANT or not CHROMA_DATABASE:
    print("❌ Missing credentials in .env file!")
    print("   Required: CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE")
    exit(1)

print(f"📋 Using credentials from .env:")
print(f"   Tenant: {CHROMA_TENANT}")
print(f"   Database: {CHROMA_DATABASE}")
print()

# Test 1: Create Cloud Client
print("1️⃣  Creating ChromaDB Cloud Client...")
try:
    client = chromadb.CloudClient(
        api_key=CHROMA_API_KEY,
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE
    )
    print("   ✅ Cloud client created successfully")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    exit(1)

# Test 2: List collections
print("\n2️⃣  Testing connection...")
try:
    collections = client.list_collections()
    print(f"   ✅ Connected!")
    print(f"   📊 Found {len(collections)} collection(s)")
    for coll in collections:
        print(f"      - {coll.name}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    exit(1)

# Test 3: Get or create test collection
print("\n3️⃣  Testing collection access...")
try:
    collection = client.get_or_create_collection(name="test_collection")
    print(f"   ✅ Collection ready")
    print(f"   📊 Count: {collection.count()}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    exit(1)

# Test 4: Store some documents
print("\n4️⃣  Storing documents...")
try:
    # Sample documents to store
    documents = [
        "RAG (Retrieval-Augmented Generation) combines retrieval with generation.",
        "Vector stores enable semantic similarity search using embeddings.",
        "ChromaDB is a vector database for storing and querying embeddings.",
        "Security enforcement ensures proper access control in RAG systems.",
    ]

    # IDs for each document
    ids = ["doc-1", "doc-2", "doc-3", "doc-4"]

    # Metadata for each document
    metadatas = [
        {"topic": "rag", "sensitivity": "PUBLIC", "tenant_id": "default"},
        {"topic": "vectors", "sensitivity": "PUBLIC", "tenant_id": "default"},
        {"topic": "chromadb", "sensitivity": "INTERNAL", "tenant_id": "default"},
        {"topic": "security", "sensitivity": "INTERNAL", "tenant_id": "default"},
    ]

    # Add documents to collection
    collection.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas
    )

    print(f"   ✅ Stored {len(documents)} documents")
    print(f"   📊 Collection count: {collection.count()}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    exit(1)

# Test 5: Query the stored documents
print("\n5️⃣  Querying stored documents...")
try:
    results = collection.query(
        query_texts=["What is RAG?"],
        n_results=2
    )

    print(f"   ✅ Query successful")
    print(f"   📊 Found {len(results['ids'][0])} result(s)")
    if results['ids'][0]:
        print(f"   📄 Top result: {results['documents'][0][0][:60]}...")
        print(f"   🏷️  Metadata: {results['metadatas'][0][0]}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    exit(1)

print("\n" + "=" * 70)
print("✅ All tests passed! Documents stored successfully!")
print("=" * 70)

