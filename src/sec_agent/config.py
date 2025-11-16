"""
VecSec Security Agent - Configuration & Initialization Module

Simple, focused configuration module that:
1. Loads and validates environment variables
2. Selects vector store (Cloud ChromaDB > Local ChromaDB > InMemory)
3. Configures logging
4. Configures metrics (optional)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# Section 1: Load + Validate Environment Variables
# ============================================================================

CONFIG_SCHEMA: dict[str, dict[str, Any]] = {
    # Required variables (if any)
    # Add here if needed
    # Optional variables
    "USE_CHROMA": {
        "type": bool,
        "required": False,
        "default": False,
        "description": "Enable local ChromaDB for persistent vector storage",
    },
    "CHROMA_PATH": {
        "type": str,
        "required": False,
        "default": "./chroma_db",
        "description": "Directory path for local ChromaDB persistence",
    },
    "CHROMA_API_KEY": {
        "type": str,
        "required": False,
        "default": None,
        "description": "ChromaDB Cloud API key (enables cloud mode)",
    },
    "CHROMA_TENANT": {
        "type": str,
        "required": False,
        "default": None,
        "description": "ChromaDB Cloud tenant ID",
    },
    "CHROMA_DATABASE": {
        "type": str,
        "required": False,
        "default": None,
        "description": "ChromaDB Cloud database name",
    },
    "METRICS_PORT": {
        "type": int,
        "required": False,
        "default": 8080,
        "description": "Port for Prometheus metrics exporter",
    },
    "LOG_FILE": {
        "type": str,
        "required": False,
        "default": None,
        "description": "Path to log file",
    },
    "LOG_LEVEL": {
        "type": str,
        "required": False,
        "default": "INFO",
        "description": "Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    },
    "LOG_TO_CONSOLE": {
        "type": bool,
        "required": False,
        "default": True,
        "description": "Whether to log to console",
    },
}


def _parse_bool(value: str | bool) -> bool:
    """Parse boolean from string (handles 'true', 'false', '1', '0', etc.)"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes", "on")
    return False


def _parse_int(value: str | int) -> int:
    """Parse integer from string"""
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as err:
            raise ValueError(f"Invalid integer value: '{value}'") from err
    raise ValueError(f"Cannot convert {type(value)} to int")


def validate_env_vars() -> None:
    """
    Validate all environment variables against CONFIG_SCHEMA.

    Raises:
        ValueError: If any env var has invalid type or value
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
            if expected_type is bool:
                if env_value.lower() not in ("true", "false", "1", "0", "yes", "no", "on", "off"):
                    errors.append(
                        f"Invalid boolean value for '{var_name}': '{env_value}'. "
                        "Expected 'true' or 'false'"
                    )
            elif expected_type is int:
                try:
                    int(env_value)
                except ValueError:
                    errors.append(
                        f"Invalid integer value for '{var_name}': '{env_value}'. "
                        "Expected a valid integer"
                    )
            elif expected_type is str:
                if schema.get("required", False) and not env_value:
                    errors.append(f"Required string '{var_name}' cannot be empty")
        except Exception as e:
            errors.append(f"Validation error for '{var_name}': {e}")

    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(error_msg)


# ============================================================================
# Section 2: Vector Store Selector (The Key Logic)
# ============================================================================

# Try to import ChromaDB
try:
    import chromadb
    from langchain_chroma import Chroma

    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    chromadb = None  # type: ignore
    Chroma = None  # type: ignore

# Try to import InMemoryVectorStore
try:
    from langchain_core.vectorstores import InMemoryVectorStore
except ImportError:
    # Fallback - create simple in-memory store
    from langchain_core.documents import Document

    class InMemoryVectorStore:  # type: ignore
        """Simple in-memory vector store fallback"""

        def __init__(self, embedding_function):
            self.embedding_function = embedding_function
            self.documents: list[Document] = []

        def add_documents(self, docs: list[Document]):
            """Add documents to the store"""
            self.documents.extend(docs)

        def similarity_search(self, query: str, k: int = 4) -> list[Document]:
            """Simple similarity search - returns first k documents"""
            return self.documents[:k]

        def add_texts(self, texts: list[str], metadatas: list[dict] | None = None):
            """Add texts to the store"""
            metadatas_list = metadatas or [{}] * len(texts)
            docs = [
                Document(page_content=text, metadata=meta or {})
                for text, meta in zip(texts, metadatas_list, strict=True)
            ]
            self.documents.extend(docs)


def _create_chroma_cloud_client():
    """
    Create ChromaDB Cloud client using the working pattern from test_chroma_cloud.py

    Returns:
        ChromaDB CloudClient instance or None if credentials missing/failed
    """
    if not CHROMA_AVAILABLE:
        return None

    # Get credentials from .env (same pattern as test_chroma_cloud.py)
    chroma_api_key = os.getenv("CHROMA_API_KEY")
    chroma_tenant = os.getenv("CHROMA_TENANT")
    chroma_database = os.getenv("CHROMA_DATABASE")

    # Check if all cloud credentials are present
    if not chroma_api_key or not chroma_tenant or not chroma_database:
        return None

    try:
        # Use the exact same pattern as test_chroma_cloud.py
        client = chromadb.CloudClient(
            api_key=chroma_api_key, tenant=chroma_tenant, database=chroma_database
        )
        print(f"☁️  Using ChromaDB Cloud (tenant: {chroma_tenant}, database: {chroma_database})")
        return client
    except Exception as e:
        print(f"⚠️  ChromaDB Cloud connection failed: {e}")
        return None


def initialize_vector_store(embeddings):
    """
    Initialize vector store based on configuration.

    Priority:
    1. ChromaDB Cloud (if CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE are set)
    2. Local ChromaDB (if USE_CHROMA=true)
    3. InMemoryVectorStore (fallback)

    Args:
        embeddings: Embedding function compatible with LangChain vector stores

    Returns:
        Vector store instance (Chroma or InMemoryVectorStore)
    """
    # Priority 1: Try ChromaDB Cloud first
    if CHROMA_AVAILABLE:
        cloud_client = _create_chroma_cloud_client()
        if cloud_client:
            try:
                # Wrap cloud client in LangChain Chroma with embedding function
                vector_store = Chroma(
                    client=cloud_client,
                    collection_name="vecsec_documents",
                    embedding_function=embeddings,
                )
                print("✅ Using ChromaDB Cloud for vector storage")
                return vector_store
            except Exception as e:
                print(f"⚠️  ChromaDB Cloud initialization failed: {e}, trying local...")
                # Fall through to local ChromaDB

    # Priority 2: Try local ChromaDB if USE_CHROMA is enabled
    use_chroma = _parse_bool(os.getenv("USE_CHROMA", str(CONFIG_SCHEMA["USE_CHROMA"]["default"])))

    if CHROMA_AVAILABLE and use_chroma:
        try:
            persist_directory = os.getenv("CHROMA_PATH", CONFIG_SCHEMA["CHROMA_PATH"]["default"])

            # Ensure directory exists or can be created
            path = Path(persist_directory)
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)

            vector_store = Chroma(
                collection_name="vecsec_documents",
                embedding_function=embeddings,
                persist_directory=persist_directory,
            )
            print(f"✅ Using ChromaDB for persistent vector storage at: {persist_directory}")
            return vector_store
        except Exception as e:
            print(f"⚠️  ChromaDB initialization failed: {e}, using InMemory")
            return InMemoryVectorStore(embeddings)

    # Priority 3: Fallback to InMemoryVectorStore
    print("💾 Using InMemoryVectorStore (no persistence)")
    return InMemoryVectorStore(embeddings)


# ============================================================================
# Section 3: Logging Configuration
# ============================================================================


def setup_logging(
    log_file: str | None = None, log_level: str = "INFO", log_to_console: bool = True
) -> None:
    """
    Configure logging for the VecSec application.

    Args:
        log_file: Path to log file (default: None, logs to console only)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_console: Whether to also log to console (default: True)
    """
    # Convert string level to logging constant
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    level = level_map.get(log_level.upper(), logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Create handlers
    handlers: list[logging.Handler] = []

    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)

    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        if log_path.parent and not log_path.parent.exists():
            log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=level,
        handlers=handlers,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# Auto-configure logging if env vars are set
if not logging.getLogger().handlers:
    _log_file = os.getenv("LOG_FILE")
    _log_level = os.getenv("LOG_LEVEL", "INFO")
    _log_to_console = _parse_bool(os.getenv("LOG_TO_CONSOLE", "true"))

    if _log_file or _log_to_console:
        setup_logging(log_file=_log_file, log_level=_log_level, log_to_console=_log_to_console)


# ============================================================================
# Section 4: Metrics Configuration (Optional)
# ============================================================================


def _initialize_metrics() -> bool:
    """
    Initialize metrics exporter with centralized error handling.

    Returns:
        bool: True if metrics enabled, False otherwise
    """
    try:
        from src.metrics_exporter import (
            metrics_exporter as metrics_exporter_instance,  # type: ignore[no-redef]
        )

        metrics_exporter_instance.start_server()
        return True
    except ImportError:
        # Fallback to same directory import
        try:
            from metrics_exporter import (
                metrics_exporter as metrics_exporter_instance,  # type: ignore[no-redef]
            )

            metrics_exporter_instance.start_server()
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
# Compatibility Stubs (for existing code - should be moved elsewhere)
# ============================================================================


def initialize_sample_documents(vector_store) -> None:
    """
    Stub function for compatibility.

    Note: This should be moved to a separate document loading module.
    """
    # Minimal implementation - just a placeholder
    print("⚠️  initialize_sample_documents() is a stub - implement document loading elsewhere")
    pass


def create_rag_prompt_template():
    """
    Stub function for compatibility.

    Note: This should be moved to a separate RAG/prompt module.
    """
    from langchain_core.prompts import ChatPromptTemplate

    # Minimal implementation - return basic template
    return ChatPromptTemplate.from_template("""
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:
""")
