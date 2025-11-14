"""
Metadata Generator - Generate retrieval metadata for documents
"""

import logging
from typing import Any

from .policy_manager import TENANT_POLICIES

# Set up module-level logger
logger = logging.getLogger(__name__)


def generate_retrieval_metadata(
    query_context: dict[str, Any],
    user_tenant: str,
    vector_store,
) -> list[dict[str, Any]]:
    """
    Generate retrieval metadata using actual vector search

    Args:
        query_context: Contains 'query' and other context
        user_tenant: Tenant ID for filtering
        vector_store: The initialized vector store instance

    Returns:
        List of metadata dicts with document information and retrieval scores
    """

    # Extract query text from context
    query_text = query_context.get("query", "")
    if not query_text:
        query_text = " ".join(query_context.get("topics", ["general"]))

    logger.debug(
        f"Starting vector search for tenant '{user_tenant}'",
        extra={"query": query_text[:100], "tenant_id": user_tenant},
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
        except (TypeError, AttributeError):
            # InMemoryVectorStore doesn't support filter dict - search without filter and filter manually
            # But only if vector_store is actually valid
            if vector_store is None:
                raise  # Re-raise the original error
            all_results = vector_store.similarity_search_with_score(query_text, k=10)
            if filter_dict and "tenant_id" in filter_dict:
                # Manually filter results by tenant_id
                results = [
                    (doc, score)
                    for doc, score in all_results
                    if doc.metadata.get("tenant_id") == filter_dict["tenant_id"]
                ][:5]
            else:
                results = all_results[:5]

        logger.info(
            f"Vector search completed: found {len(results)} results",
            extra={
                "tenant_id": user_tenant,
                "result_count": len(results),
                "has_filter": filter_dict is not None,
            },
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

            metadata.append(
                {
                    "embedding_id": doc.metadata.get("embedding_id", f"emb-{i + 1:03d}"),
                    "tenant_id": doc.metadata.get("tenant_id", user_tenant),
                    "sensitivity": doc.metadata.get(
                        "sensitivity",
                        TENANT_POLICIES.get(user_tenant, {}).get("sensitivity", "INTERNAL"),
                    ),
                    "topics": topics,  # Now properly converted from string to list
                    "document_id": doc.metadata.get("document_id", f"doc-real-{i + 1:03d}"),
                    "retrieval_score": float(score),  # REAL similarity score!
                    "content": doc.page_content[:200] if doc.page_content else "",
                }
            )

        # Handle no results case
        if not metadata:
            logger.warning(
                f"No documents found for query with tenant filter '{user_tenant}'",
                extra={
                    "tenant_id": user_tenant,
                    "query_preview": query_text[:100],
                    "fallback": "empty_metadata",
                },
            )
            # No results found
            metadata = [
                {
                    "embedding_id": "emb-empty",
                    "tenant_id": user_tenant,
                    "sensitivity": "INTERNAL",
                    "topics": query_context.get("topics", []),
                    "document_id": "doc-empty",
                    "retrieval_score": 0.0,
                    "content": "No documents found",
                }
            ]

        logger.info(
            f"Retrieval metadata generated successfully: {len(metadata)} items",
            extra={"tenant_id": user_tenant, "metadata_count": len(metadata)},
        )
        return metadata
    except Exception as e:
        logger.error(
            "Vector search failed",
            exc_info=True,
            extra={
                "tenant_id": user_tenant,
                "query_preview": query_text[:100],
                "error_type": type(e).__name__,
            },
        )
        # Return empty metadata on error
        return [
            {
                "embedding_id": "emb-error",
                "tenant_id": user_tenant,
                "sensitivity": "INTERNAL",
                "topics": query_context.get("topics", []),
                "document_id": "doc-error",
                "retrieval_score": 0.0,
                "content": "Vector search error occurred",
            }
        ]
