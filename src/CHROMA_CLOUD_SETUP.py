#!/usr/bin/env python3
"""
ChromaDB Cloud Memory Setup for VecSec
Stores vector embeddings in cloud-hosted ChromaDB
"""

import os

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()


def setup_chroma_cloud():
    """Initialize ChromaDB with cloud memory"""

    # Option 1: Use ChromaDB Cloud (requires API key)
    chroma_api_key = os.getenv("CHROMA_API_KEY")
    chroma_host = os.getenv("CHROMA_HOST", "chromadb.net")

    if chroma_api_key:
        print("☁️  Using ChromaDB Cloud")
        client = chromadb.HttpClient(
            host=chroma_host,
            port=443,
            headers={
                "Authorization": f"Bearer {chroma_api_key}",
            },
        )
        return client

    # Option 2: Use local ChromaDB with persistent storage
    print("💾 Using Local ChromaDB with persistent storage")
    client = chromadb.PersistentClient(
        path="./chroma_db",  # Persistent storage directory
        settings=Settings(anonymized_telemetry=False),
    )
    return client


def get_collection(collection_name="vecsec_threat_patterns"):
    """Get or create a ChromaDB collection"""
    client = setup_chroma_cloud()

    try:
        collection = client.get_collection(name=collection_name)
        print(f"✅ Using existing collection: {collection_name}")
    except Exception:
        collection = client.create_collection(
            name=collection_name, metadata={"description": "VecSec threat detection patterns"}
        )
        print(f"✅ Created new collection: {collection_name}")

    return collection


if __name__ == "__main__":
    print("\n🔧 ChromaDB Cloud Memory Setup")
    print("=" * 50)

    client = setup_chroma_cloud()
    collection = get_collection()

    print("\n📊 Collection Info:")
    print(f"   Collection: {collection.name}")
    print(f"   Count: {collection.count()}")
    print("\n✅ ChromaDB ready for VecSec!")
    print("\n💡 To use cloud memory, set CHROMA_API_KEY in .env")
    print("   Or use local persistent storage (default)")
