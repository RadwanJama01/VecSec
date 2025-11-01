"""
Metadata Generator - Generate retrieval metadata for documents
"""

from typing import Dict, Any, List
from .policy_manager import TENANT_POLICIES


def generate_retrieval_metadata(query_context: Dict[str, Any], user_tenant: str) -> List[Dict[str, Any]]:
    """Generate mock retrieval metadata based on query context"""
    metadata = []
    
    # Simulate documents that might be retrieved
    target_tenant = query_context.get("target_tenant") or user_tenant  # Default to user tenant if None
    topics = query_context.get("topics", [])
    
    # Generate mock embeddings with different tenants and sensitivities
    for i, topic in enumerate(topics):
        metadata.append({
            "embedding_id": f"emb-{i+1:03d}",
            "tenant_id": target_tenant,
            "sensitivity": TENANT_POLICIES.get(target_tenant, {}).get("sensitivity", "INTERNAL"),
            "topics": [topic],
            "document_id": f"doc-{topic}-{i+1:03d}",
            "retrieval_score": 0.9 - (i * 0.1)
        })
    
    # Add some cross-tenant documents to test isolation (only if explicitly targeting different tenant)
    if target_tenant != user_tenant and query_context.get("target_tenant"):
        metadata.append({
            "embedding_id": f"emb-cross-001",
            "tenant_id": user_tenant,
            "sensitivity": TENANT_POLICIES.get(user_tenant, {}).get("sensitivity", "INTERNAL"),
            "topics": topics[:1] if topics else ["general"],
            "document_id": f"doc-cross-001",
            "retrieval_score": 0.7
        })
    
    return metadata

