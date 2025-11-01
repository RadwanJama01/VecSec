"""
VecSec Security Agent - RLS-Enforced RAG System with Threat Detection

BACKWARD COMPATIBILITY WRAPPER
===============================
This file maintains backward compatibility with existing imports.
New code should import from src.sec_agent package directly.

Example:
    from src.sec_agent import RAGOrchestrator, QwenEmbeddingClient
    # or
    from src.sec_agent.cli import main
"""

# Import components for backward compatibility
from src.sec_agent.config import METRICS_ENABLED, initialize_vector_store, initialize_sample_documents, create_rag_prompt_template
from src.sec_agent.mock_llm import MockLLM, MockEmbeddings
from src.sec_agent.embeddings_client import QwenEmbeddingClient
from src.sec_agent.threat_detector import ContextualThreatEmbedding
from src.sec_agent.policy_manager import TENANT_POLICIES, ROLE_POLICIES
from src.sec_agent.query_parser import extract_query_context
from src.sec_agent.metadata_generator import generate_retrieval_metadata
from src.sec_agent.rls_enforcer import rlsa_guard_comprehensive
from src.sec_agent.rag_orchestrator import RAGOrchestrator

# Initialize singleton instances for backward compatibility
llm = MockLLM()
embeddings = MockEmbeddings()
qwen_client = QwenEmbeddingClient()
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
    qwen_client=qwen_client
)


def rag_with_rlsa(user_id, tenant_id, clearance, query, role="analyst"):
    """Backward-compatible wrapper for rag_with_rlsa"""
    return orchestrator.rag_with_rlsa(user_id, tenant_id, clearance, query, role)


def main():
    """Backward-compatible main function"""
    from .sec_agent.cli import main as cli_main
    cli_main()


if __name__ == "__main__":
    main()
