"""
VecSec Security Agent - RLS-Enforced RAG System with Threat Detection

BACKWARD COMPATIBILITY WRAPPER
===============================
This file maintains backward compatibility with existing imports.
New code should import from src.sec_agent package directly.

Example:
    from src.sec_agent import RAGOrchestrator, EmbeddingClient
    # or
    from src.sec_agent.cli import main
"""

import sys
from pathlib import Path

# Add project root to Python path for imports
# This allows running: python3 src/Sec_Agent.py
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import components for backward compatibility
# Imports after path setup - needed for running script directly
from src.sec_agent.config import (  # noqa: E402
    create_rag_prompt_template,
    initialize_sample_documents,
    initialize_vector_store,
)
from src.sec_agent.embeddings_client import EmbeddingClient  # noqa: E402
from src.sec_agent.mock_llm import MockEmbeddings, MockLLM  # noqa: E402
from src.sec_agent.rag_orchestrator import RAGOrchestrator  # noqa: E402
from src.sec_agent.threat_detector import ContextualThreatEmbedding  # noqa: E402

# Initialize singleton instances for backward compatibility
llm = MockLLM()
embeddings = MockEmbeddings()
qwen_client = (
    EmbeddingClient()
)  # Note: variable name kept as 'qwen_client' for backward compatibility
# Backward compatibility alias for QwenEmbeddingClient
QwenEmbeddingClient = EmbeddingClient
threat_embedder = ContextualThreatEmbedding(qwen_client)

# Initialize vector store
vector_store = initialize_vector_store(embeddings)
initialize_sample_documents(vector_store)

# Initialize prompt template
prompt = create_rag_prompt_template()

# Create orchestrator instance
orchestrator = RAGOrchestrator(
    vector_store=vector_store,
    llm=llm,
    prompt_template=prompt,
    threat_embedder=threat_embedder,
    qwen_client=qwen_client,
)


def rag_with_rlsa(user_id, tenant_id, clearance, query, role="analyst"):
    """Backward-compatible wrapper for rag_with_rlsa"""
    return orchestrator.rag_with_rlsa(user_id, tenant_id, clearance, query, role)


def main():
    """Backward-compatible main function"""
    from src.sec_agent.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
