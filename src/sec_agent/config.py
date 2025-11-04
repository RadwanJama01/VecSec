"""
VecSec Security Agent - Configuration & Initialization Module

This module handles all configuration, environment setup, and initialization for the VecSec
security agent system. It provides a centralized configuration management layer with validation,
schema-based env var handling, and robust initialization for vector stores, metrics, and
documentation.

Key Responsibilities:
-------------------

1. **Environment Variable Management**
   - Loads and validates environment variables from .env file
   - Provides config schema with types, defaults, and required flags
   - Validates env var types and values (boolean, int, string)
   - Fails early with clear error messages for invalid config

2. **Vector Store Initialization**
   - Initializes ChromaDB for persistent storage (if configured)
   - Falls back to InMemoryVectorStore if ChromaDB unavailable
   - Configurable via CHROMA_PATH env var
   - Handles initialization errors gracefully

3. **Metrics Exporter Setup**
   - Initializes Prometheus metrics exporter
   - Handles port conflicts gracefully
   - Provides METRICS_ENABLED flag for conditional metrics tracking

4. **Document Loading**
   - Provides sample document initialization (fallback)
   - Supports dynamic document loading from files/database (extensible)
   - Loads documents into vector store for RAG operations

5. **RAG Prompt Template**
   - Creates standardized prompt template for RAG operations
   - Ensures consistent prompt format across the system

Configuration Variables:
----------------------

Environment variables (see CONFIG_SCHEMA for complete list):
- USE_CHROMA: Enable ChromaDB persistence (bool, default: false)
- CHROMA_PATH: Path for ChromaDB storage (str, default: ./chroma_db)
- METRICS_PORT: Port for metrics exporter (int, default: 8080)
- BASETEN_MODEL_ID: BaseTen model ID for embeddings (str, optional)
- BASETEN_API_KEY: BaseTen API key for embeddings (str, optional)

Usage:
------

    from src.sec_agent.config import (
        initialize_vector_store,
        initialize_sample_documents,
        create_rag_prompt_template,
        validate_env_vars,
        CONFIG_SCHEMA
    )
    
    # Validate configuration before use
    validate_env_vars()
    
    # Initialize components
    embeddings = MockEmbeddings()
    vector_store = initialize_vector_store(embeddings)
    initialize_sample_documents(vector_store)
    template = create_rag_prompt_template()


Author: VecSec Labs
License: For authorized security testing and research only.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Try to import ChromaDB for persistent storage
try:
    from langchain_chroma import Chroma
    CHROMA_AVAILABLE = True
except ImportError:
    print("⚠️  langchain-chroma not installed, using InMemory storage")
    CHROMA_AVAILABLE = False

# Import Document first (needed for fallback)
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# Try to import InMemoryVectorStore (available in langchain-core >= 0.2.0)
# For older versions (0.1.x), we need to use a fallback
try:
    from langchain_core.vectorstores import InMemoryVectorStore
except ImportError:
    # Fallback for older versions - try alternative locations
    try:
        from langchain.vectorstores import InMemoryVectorStore
    except ImportError:
        try:
            from langchain_core.vectorstores.in_memory import InMemoryVectorStore
        except ImportError:
            # Last resort - create a simple fallback that mimics the interface
            from typing import List
            class InMemoryVectorStore:
                """Fallback InMemoryVectorStore for older langchain-core versions"""
                def __init__(self, embeddings):
                    self.embeddings = embeddings
                    self.documents: List[Document] = []
                
                def add_documents(self, docs: List[Document]):
                    """Add documents to the store"""
                    self.documents.extend(docs)
                
                def similarity_search(self, query: str, k: int = 4) -> List[Document]:
                    """Simple similarity search - returns first k documents"""
                    # For old versions without proper similarity, just return first k
                    return self.documents[:k]

# ============================================================================
# Configuration Schema
# ============================================================================

CONFIG_SCHEMA: Dict[str, Dict[str, Any]] = {
    "USE_CHROMA": {
        "type": bool,
        "required": False,
        "default": False,
        "description": "Enable ChromaDB for persistent vector storage"
    },
    "CHROMA_PATH": {
        "type": str,
        "required": False,
        "default": "./chroma_db",
        "description": "Directory path for ChromaDB persistence"
    },
    "METRICS_PORT": {
        "type": int,
        "required": False,
        "default": 8080,
        "description": "Port for Prometheus metrics exporter"
    },
    "BASETEN_MODEL_ID": {
        "type": str,
        "required": False,
        "default": None,
        "description": "BaseTen model ID for embeddings"
    },
    "BASETEN_API_KEY": {
        "type": str,
        "required": False,
        "default": None,
        "description": "BaseTen API key for embeddings"
    }
}


# ============================================================================
# Environment Variable Validation
# ============================================================================

def _parse_bool(value: str) -> bool:
    """Parse boolean from string (handles 'true', 'false', '1', '0', etc.)"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes', 'on')
    return False


def _parse_int(value: str) -> int:
    """Parse integer from string"""
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            raise ValueError(f"Invalid integer value: '{value}'")
    raise ValueError(f"Cannot convert {type(value)} to int")


def validate_env_vars() -> None:
    """
    Validate all environment variables against CONFIG_SCHEMA.
    
    Raises:
        ValueError: If any env var has invalid type or value
        FileNotFoundError: If required path doesn't exist and can't be created
    """
    errors = []
    
    for var_name, schema in CONFIG_SCHEMA.items():
        env_value = os.getenv(var_name)
        
        # If not set, use default (unless required)
        if env_value is None:
            if schema.get("required", False):
                errors.append(f"Required environment variable '{var_name}' is not set")
            continue
        
        # Validate type
        expected_type = schema["type"]
        
        try:
            if expected_type == bool:
                # Validate boolean - must be 'true' or 'false'
                if env_value.lower() not in ('true', 'false', '1', '0', 'yes', 'no', 'on', 'off'):
                    errors.append(
                        f"Invalid boolean value for '{var_name}': '{env_value}'. "
                        f"Expected 'true' or 'false'"
                    )
            elif expected_type == int:
                # Validate integer
                try:
                    int(env_value)
                except ValueError:
                    errors.append(
                        f"Invalid integer value for '{var_name}': '{env_value}'. "
                        f"Expected a valid integer"
                    )
            elif expected_type == str:
                # String validation - check if empty when required
                if schema.get("required", False) and not env_value:
                    errors.append(f"Required string '{var_name}' cannot be empty")
        except Exception as e:
            errors.append(f"Validation error for '{var_name}': {e}")
    
    # Validate CHROMA_PATH if set
    chroma_path = os.getenv("CHROMA_PATH", CONFIG_SCHEMA["CHROMA_PATH"]["default"])
    if chroma_path:
        path = Path(chroma_path)
        # Check if path exists or can be created
        if not path.exists():
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(
                    f"CHROMA_PATH '{chroma_path}' does not exist and cannot be created: {e}"
                )
    
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(error_msg)


# ============================================================================
# Metrics Initialization (Refactored - CONFIG-003)
# ============================================================================

def _initialize_metrics() -> bool:
    """
    Initialize metrics exporter with centralized error handling.
    
    Returns:
        bool: True if metrics enabled, False otherwise
    """
    try:
        from src.metrics_exporter import metrics_exporter, start_metrics_exporter
        metrics_exporter.start_server()
        return True
    except ImportError:
        # Fallback to same directory import
        try:
            from metrics_exporter import metrics_exporter, start_metrics_exporter
            metrics_exporter.start_server()
            return True
        except ImportError:
            print("⚠️  metrics_exporter not available")
            return False
        except Exception as e:
            print(f"⚠️  Could not start metrics exporter: {e}")
            return False
    except Exception as e:
        print(f"⚠️  Could not start metrics exporter: {e}")
        return False


# Initialize metrics exporter
METRICS_ENABLED = _initialize_metrics()


# ============================================================================
# Vector Store Initialization
# ============================================================================

def initialize_vector_store(embeddings):
    """
    Initialize vector store (ChromaDB or InMemory) based on configuration.
    
    Uses CHROMA_PATH env var if set, otherwise defaults to './chroma_db'.
    Falls back to InMemoryVectorStore if ChromaDB is unavailable or fails.
    
    Args:
        embeddings: Embedding function compatible with LangChain vector stores
        
    Returns:
        Vector store instance (Chroma or InMemoryVectorStore)
    """
    # Check if ChromaDB should be used (CONFIG-002: Read from env var)
    use_chroma = _parse_bool(os.getenv("USE_CHROMA", str(CONFIG_SCHEMA["USE_CHROMA"]["default"])))
    
    if CHROMA_AVAILABLE and use_chroma:
        try:
            # Use configurable path (CONFIG-002 fix)
            persist_directory = os.getenv(
                "CHROMA_PATH",
                CONFIG_SCHEMA["CHROMA_PATH"]["default"]
            )
            
            # Ensure directory exists or can be created
            path = Path(persist_directory)
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
            
            vector_store = Chroma(
                collection_name="vecsec_documents",
                embedding_function=embeddings,
                persist_directory=persist_directory
            )
            print(f"✅ Using ChromaDB for persistent vector storage at: {persist_directory}")
            return vector_store
        except Exception as e:
            print(f"⚠️  ChromaDB initialization failed: {e}, using InMemory")
            return InMemoryVectorStore(embeddings)
    else:
        return InMemoryVectorStore(embeddings)


# ============================================================================
# Document Loading
# ============================================================================

def initialize_sample_documents(vector_store) -> None:
    """
    Add sample documents to the vector store (fallback/default).
    
    This is a static document loader for demo/testing purposes.
    For production, use load_documents_from_file() or load_documents_from_db().
    
    Args:
        vector_store: Vector store instance to add documents to
    """
    sample_docs = [
        Document(page_content="RAG (Retrieval-Augmented Generation) is a technique that combines information retrieval with text generation to produce more accurate and contextually relevant responses."),
        Document(page_content="LangChain is a framework for developing applications powered by language models. It provides tools for building RAG applications."),
        Document(page_content="Vector stores are databases that store and retrieve documents based on semantic similarity using embeddings."),
        Document(page_content="Embeddings are dense vector representations of text that capture semantic meaning and enable similarity search.")
    ]
    vector_store.add_documents(sample_docs)


def load_documents_from_file(file_path: str, tenant_id: Optional[str] = None) -> List[Document]:
    """
    Load documents from a file (JSON, YAML, or text).
    
    This function provides dynamic document loading as an alternative to
    static sample documents. Supports tenant-specific document loading.
    
    Args:
        file_path: Path to document file (supports .json, .yaml, .txt)
        tenant_id: Optional tenant ID for filtering documents
        
    Returns:
        List of Document objects
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is unsupported
        
    Note: This is a placeholder for CONFIG-004. Full implementation pending.
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Document file not found: {file_path}")
    
    # TODO: Implement JSON/YAML/TXT parsing
    # For now, return empty list as placeholder
    print(f"⚠️  load_documents_from_file() not yet implemented (CONFIG-004)")
    return []


def load_documents_from_db(query: str, tenant_id: Optional[str] = None) -> List[Document]:
    """
    Load documents from a database query.
    
    This function provides database-backed document loading as an alternative to
    static sample documents. Supports tenant-specific filtering.
    
    Args:
        query: Database query string or SQL
        tenant_id: Optional tenant ID for filtering documents
        
    Returns:
        List of Document objects
        
    Raises:
        NotImplementedError: Database loading not yet implemented
        
    Note: This is a placeholder for CONFIG-004. Full implementation pending.
    """
    # TODO: Implement database query execution
    # For now, raise NotImplementedError
    raise NotImplementedError(
        "load_documents_from_db() not yet implemented (CONFIG-004). "
        "Use initialize_sample_documents() as fallback."
    )


# ============================================================================
# RAG Prompt Template
# ============================================================================

def create_rag_prompt_template() -> ChatPromptTemplate:
    """
    Create RAG prompt template for consistent prompt formatting.
    
    Returns:
        ChatPromptTemplate instance with context and question placeholders
    """
    return ChatPromptTemplate.from_template("""
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:
""")


# ============================================================================
# Module Initialization
# ============================================================================

# Validate configuration on module import (optional - can be called explicitly)
# Uncomment below to enable auto-validation:
# try:
#     validate_env_vars()
# except ValueError as e:
#     print(f"⚠️  Configuration validation warning: {e}")
#     print("⚠️  Continuing with defaults - fix config for production!")
