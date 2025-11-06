"""
Metadata Generator - Generate retrieval metadata for documents
"""

import logging
from typing import Dict, Any, List
from .policy_manager import TENANT_POLICIES

# Set up module-level logger
logger = logging.getLogger(__name__)


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


def generate_retrieval_metadata_real(
    query_context: Dict[str, Any],
    user_tenant: str,
    vector_store,
) -> List[Dict[str, Any]]:
    """
    Generate REAL retrieval metadata using actual vector search
    
    Args:
        query_context: Contains 'query' and other context
        user_tenant: Tenant ID for filtering
        vector_store: The initialized vector store instance
    
    Returns:
        List of metadata dicts matching the mock format
    """
    
    # Extract query text from context
    query_text = query_context.get("query", "")
    if not query_text:
        query_text = " ".join(query_context.get("topics", ["general"]))
    
    logger.debug(
        f"Starting vector search for tenant '{user_tenant}'",
        extra={"query": query_text[:100], "tenant_id": user_tenant}
    )

    # Search vector store for relevant documents
    try:
        # Check if vector_store is None or invalid
        if vector_store is None:
            raise AttributeError("Vector store is None")
        
        # Perform vector search with filtering
        filter_dict = {"tenant_id": user_tenant} if user_tenant else None
        
        # Try with filter first (works for ChromaDB)
        try:
            results = vector_store.similarity_search_with_score(
                query_text, 
                k=5,
                filter=filter_dict,
            )
        except (TypeError, AttributeError) as filter_error:
            # InMemoryVectorStore doesn't support filter dict - search without filter and filter manually
            # But only if vector_store is actually valid
            if vector_store is None:
                raise  # Re-raise the original error
            all_results = vector_store.similarity_search_with_score(query_text, k=10)
            if filter_dict and "tenant_id" in filter_dict:
                # Manually filter results by tenant_id
                results = [
                    (doc, score) for doc, score in all_results 
                    if doc.metadata.get("tenant_id") == filter_dict["tenant_id"]
                ][:5]
            else:
                results = all_results[:5]
        
        logger.info(
            f"Vector search completed: found {len(results)} results",
            extra={
                "tenant_id": user_tenant,
                "result_count": len(results),
                "has_filter": filter_dict is not None
            }
        )

        # Initialize metadata list
        metadata = []
        # Process search results into metadata format
        for i, (doc, score) in enumerate(results):
            # ChromaDB stores topics as comma-separated string, convert back to list
            topics_raw = doc.metadata.get("topics", "")
            if isinstance(topics_raw, str):
                # Convert comma-separated string to list
                topics = [t.strip() for t in topics_raw.split(",")] if topics_raw else []
            elif isinstance(topics_raw, list):
                # Already a list (shouldn't happen if stored correctly, but handle it)
                topics = topics_raw
            else:
                # Fallback to query context topics
                topics = query_context.get("topics", [])
            
            metadata.append({
                "embedding_id": doc.metadata.get("embedding_id", f"emb-{i+1:03d}"),
                "tenant_id": doc.metadata.get("tenant_id", user_tenant),
                "sensitivity": doc.metadata.get(
                    "sensitivity", 
                    TENANT_POLICIES.get(user_tenant, {}).get("sensitivity", "INTERNAL")
                ),
                "topics": topics,  # Now properly converted from string to list
                "document_id": doc.metadata.get("document_id", f"doc-real-{i+1:03d}"),
                "retrieval_score": float(score),  # REAL similarity score!
                "content": doc.page_content[:200] if doc.page_content else ""
            })

        # Handle no results case
        if not metadata:
            logger.warning(
                f"No documents found for query with tenant filter '{user_tenant}'",
                extra={
                    "tenant_id": user_tenant,
                    "query_preview": query_text[:100],
                    "fallback": "empty_metadata"
                }
            )
            # No results found
            metadata = [{
                "embedding_id": "emb-empty",
                "tenant_id": user_tenant,
                "sensitivity": "INTERNAL",
                "topics": query_context.get("topics", []),
                "document_id": "doc-empty",
                "retrieval_score": 0.0,
                "content": "No documents found"
            }]

        logger.info(
            f"Retrieval metadata generated successfully: {len(metadata)} items",
            extra={"tenant_id": user_tenant, "metadata_count": len(metadata)}
        )
        return metadata
    except Exception as e:
        logger.error(
            f"Vector search failed, falling back to mock metadata",
            exc_info=True,
            extra={
                "tenant_id": user_tenant,
                "query_preview": query_text[:100],
                "error_type": type(e).__name__,
                "fallback": "mock_metadata"
            }
        )
        return generate_retrieval_metadata(query_context, user_tenant)
    