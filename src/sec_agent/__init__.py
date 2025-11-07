"""
VecSec Security Agent Package

Modular security agent with RLS enforcement and threat detection.
"""

from .cli import main
from .config import (
    CHROMA_AVAILABLE,
    METRICS_ENABLED,
    create_rag_prompt_template,
    initialize_sample_documents,
    initialize_vector_store,
)
from .embeddings_client import EmbeddingClient
from .metadata_generator import generate_retrieval_metadata
from .mock_llm import MockEmbeddings, MockLLM
from .policy_manager import (
    ROLE_POLICIES,
    TENANT_POLICIES,
    can_access_clearance,
    can_access_tenant,
    can_access_topic,
    can_bypass_restriction,
    compare_clearance_levels,
    get_role_policy,
    get_tenant_policy,
    has_operation_permission,
    validate_role_policy,
    validate_tenant_policy,
)
from .query_parser import extract_query_context
from .rag_orchestrator import RAGOrchestrator, State
from .rls_enforcer import rlsa_guard_comprehensive
from .threat_detector import ContextualThreatEmbedding

__all__ = [
    # Classes
    "EmbeddingClient",
    "ContextualThreatEmbedding",
    "MockLLM",
    "MockEmbeddings",
    "RAGOrchestrator",
    "State",
    # Functions
    "extract_query_context",
    "generate_retrieval_metadata",
    "rlsa_guard_comprehensive",
    "get_tenant_policy",
    "get_role_policy",
    "validate_tenant_policy",
    "validate_role_policy",
    "can_access_clearance",
    "can_access_topic",
    "can_access_tenant",
    "can_bypass_restriction",
    "has_operation_permission",
    "compare_clearance_levels",
    "main",
    "initialize_vector_store",
    "initialize_sample_documents",
    "create_rag_prompt_template",
    # Constants
    "TENANT_POLICIES",
    "ROLE_POLICIES",
    "METRICS_ENABLED",
    "CHROMA_AVAILABLE",
]
