# Vector Store Configuration Documentation

## Overview

VecSec uses **ChromaDB** as the primary vector database for persistent document embeddings storage. The system falls back to `InMemoryVectorStore` if ChromaDB is unavailable.

## Vector Database

**Type**: ChromaDB (via `langchain-chroma`)  
**Collection Name**: `vecsec_documents`  
**Storage**: Persistent disk-based (local) or in-memory (fallback)

## Connection Parameters

### Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `USE_CHROMA` | bool | `false` | Enable ChromaDB for persistent storage |
| `CHROMA_PATH` | str | `./chroma_db` | Directory path for ChromaDB persistence |

### Configuration Example

```bash
# Enable ChromaDB persistence
export USE_CHROMA=true
export CHROMA_PATH=./chroma_db
```

Or in `.env` file:
```
USE_CHROMA=true
CHROMA_PATH=./chroma_db
```

## Initialization Code Location

**File**: `src/sec_agent/config.py`  
**Function**: `initialize_vector_store(embeddings)`

```python
from src.sec_agent.config import initialize_vector_store
from src.sec_agent.mock_llm import MockEmbeddings

embeddings = MockEmbeddings()
vector_store = initialize_vector_store(embeddings)
```

### Initialization Logic

1. Checks `USE_CHROMA` environment variable
2. If enabled and ChromaDB available:
   - Creates/uses directory at `CHROMA_PATH`
   - Initializes ChromaDB with collection name `vecsec_documents`
   - Returns `Chroma` instance
3. If disabled or ChromaDB unavailable:
   - Falls back to `InMemoryVectorStore`
   - Returns in-memory vector store (non-persistent)

## Embedding Model

**Model**: `all-MiniLM-L6-v2` (SentenceTransformers)  
**Location**: `src/sec_agent/embeddings_client.py`  
**Dimensions**: 384  
**Type**: Local, offline (no API required)

The embedding model is used by `EmbeddingClient` class:
```python
from src.sec_agent.embeddings_client import EmbeddingClient

client = EmbeddingClient(model_name="all-MiniLM-L6-v2")
```

## RAG Orchestrator Connection

The RAG orchestrator connects to the vector store as follows:

**File**: `src/sec_agent/rag_orchestrator.py`  
**Connection**: Via constructor injection

```python
from src.sec_agent.rag_orchestrator import RAGOrchestrator
from src.sec_agent.config import initialize_vector_store, initialize_sample_documents

# Initialize vector store
vector_store = initialize_vector_store(embeddings)
initialize_sample_documents(vector_store)

# Create orchestrator with vector store
orchestrator = RAGOrchestrator(
    vector_store=vector_store,
    llm=llm,
    prompt_template=template,
    threat_embedder=threat_embedder,
    qwen_client=embedding_client
)
```

The orchestrator uses `vector_store.similarity_search()` for document retrieval.

## Verifying Document Embeddings

The vector store contains actual document embeddings loaded via:

**Function**: `initialize_sample_documents(vector_store)` in `config.py`

This function:
- Loads sample security documents
- Converts metadata to ChromaDB-compatible format (lists → comma-separated strings)
- Adds documents with embeddings to the vector store

## Testing Connectivity

### Quick Connection Test

Run the connection test script:
```bash
python3 scripts/test_vector_store_connection.py
```

### Full Integration Test

For comprehensive testing with real ChromaDB:
```bash
./scripts/test_chromadb.sh
```

### Sample Query

See `scripts/test_chromadb.sh` lines 293-340 for a sample query that demonstrates:
- Real vector store connection
- Actual similarity search
- Real document IDs and content
- Tenant filtering

## Dependencies

### Required Packages

```bash
pip install langchain-chroma sentence-transformers
```

### ChromaDB Installation

ChromaDB is included via `langchain-chroma`:
```bash
pip install langchain-chroma
```

## Troubleshooting

### ChromaDB Not Initializing

1. Check if `USE_CHROMA=true` is set
2. Verify `langchain-chroma` is installed: `pip install langchain-chroma`
3. Check directory permissions for `CHROMA_PATH`
4. System will automatically fall back to `InMemoryVectorStore` if ChromaDB fails

### Embeddings Not Working

1. Verify `sentence-transformers` is installed: `pip install sentence-transformers`
2. Check `EmbeddingClient.enabled` property
3. Model downloads automatically on first use

### Vector Store Empty

1. Ensure `initialize_sample_documents(vector_store)` is called after initialization
2. Check if documents were added successfully (no errors during initialization)
3. Verify collection exists: Check `CHROMA_PATH` directory contents

## Related Files

- `src/sec_agent/config.py` - Vector store initialization
- `src/sec_agent/embeddings_client.py` - Embedding model
- `src/sec_agent/rag_orchestrator.py` - RAG orchestrator (uses vector store)
- `scripts/test_chromadb.sh` - Integration test script
- `scripts/test_vector_store_connection.py` - Connection test script

