"""
RAG Orchestrator - RAG graph and orchestration logic
"""

import json
import logging
import time
import uuid
from datetime import UTC, datetime
from typing import TypedDict

from langchain_core.documents import Document
from langgraph.graph import START, StateGraph

from .config import METRICS_ENABLED, USE_REAL_VECTOR_RETRIEVAL
from .metadata_generator import generate_retrieval_metadata, generate_retrieval_metadata_real
from .query_parser import extract_query_context
from .rls_enforcer import rlsa_guard_comprehensive

# Set up module-level logger
logger = logging.getLogger(__name__)

# Import metrics_exporter if available
try:
    from src.metrics_exporter import (
        metrics_exporter as metrics_exporter_instance,  # type: ignore[no-redef]
    )
except ImportError:
    try:
        from metrics_exporter import (
            metrics_exporter as metrics_exporter_instance,  # type: ignore[no-redef]
        )
    except ImportError:
        metrics_exporter_instance = None


# Define State
class State(TypedDict):
    question: str
    context: list[Document]
    answer: str


class RAGOrchestrator:
    """RAG Orchestrator with RLS enforcement"""

    def __init__(self, vector_store, llm, prompt_template, threat_embedder=None, qwen_client=None):
        self.vector_store = vector_store
        self.llm = llm
        self.prompt_template = prompt_template
        self.threat_embedder = threat_embedder
        self.qwen_client = qwen_client

        # Build graph
        graph_builder = StateGraph(State).add_sequence([self.retrieve, self.generate])
        graph_builder.add_edge(START, "retrieve")
        self.graph = graph_builder.compile()

    def retrieve(self, state: State):
        """Retrieve documents from vector store"""
        retrieved_docs = self.vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}

    def generate(self, state: State):
        """Generate answer using LLM"""
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = self.prompt_template.format_messages(
            question=state["question"], context=docs_content
        )
        response = self.llm.invoke(messages)
        return {"answer": response.content}

    def rag_with_rlsa(self, user_id, tenant_id, clearance, query, role="analyst"):
        """Enhanced RLSA-wrapped RAG call"""
        start_time = time.time()

        # Step 1: Extract comprehensive context
        user_context = {
            "user_id": user_id,
            "tenant_id": tenant_id,
            "clearance": clearance,
            "role": role,
        }

        query_context = extract_query_context(query)

        # Migration: Use real vector store if feature flag enabled, otherwise use mock
        if USE_REAL_VECTOR_RETRIEVAL and self.vector_store is not None:
            logger.info(
                "Using real vector store for retrieval metadata",
                extra={
                    "tenant_id": tenant_id,
                    "migration": "real_vector_retrieval",
                    "query_preview": query[:100] if query else "",
                },
            )
            try:
                retrieval_metadata = generate_retrieval_metadata_real(
                    query_context, tenant_id, self.vector_store
                )
                logger.debug(
                    f"Real vector retrieval successful: {len(retrieval_metadata)} results",
                    extra={
                        "tenant_id": tenant_id,
                        "result_count": len(retrieval_metadata),
                        "migration": "real_vector_retrieval",
                    },
                )
            except Exception as e:
                # Fallback to mock on error (backward compatibility)
                # Only log full traceback at DEBUG level to reduce noise in tests
                logger.warning(
                    f"Real vector retrieval failed, falling back to mock: {e}",
                    exc_info=logger.isEnabledFor(
                        logging.DEBUG
                    ),  # Full traceback only in DEBUG mode
                    extra={
                        "tenant_id": tenant_id,
                        "migration": "real_vector_retrieval_fallback",
                        "error_type": type(e).__name__,
                    },
                )
                retrieval_metadata = generate_retrieval_metadata(query_context, tenant_id)
        else:
            # Backward compatibility: Use mock metadata generator
            if not USE_REAL_VECTOR_RETRIEVAL:
                logger.debug(
                    "Feature flag disabled, using mock metadata generator",
                    extra={
                        "tenant_id": tenant_id,
                        "migration": "mock_metadata",
                        "feature_flag": "USE_REAL_VECTOR_RETRIEVAL=false",
                    },
                )
            elif self.vector_store is None:
                logger.warning(
                    "Vector store is None, falling back to mock metadata generator",
                    extra={
                        "tenant_id": tenant_id,
                        "migration": "mock_metadata_fallback",
                        "reason": "vector_store_none",
                    },
                )
            retrieval_metadata = generate_retrieval_metadata(query_context, tenant_id)

        # Step 2: Comprehensive RLSA Enforcement
        decision = rlsa_guard_comprehensive(
            user_context, query_context, retrieval_metadata, threat_embedder=self.threat_embedder
        )

        # Track metrics
        if METRICS_ENABLED and metrics_exporter_instance:
            duration = time.time() - start_time
            # Check if decision is allowed: either True (boolean) or dict with "allowed": True
            is_allowed = decision is True or (
                isinstance(decision, dict) and decision.get("allowed") is True
            )
            is_blocked = not is_allowed
            has_threat = bool(query_context.get("detected_threats"))

            # Track request and performance
            metrics_exporter_instance.track_request(
                "blocked" if is_blocked else "allowed", duration
            )

            if is_blocked:
                metrics_exporter_instance.track_file_processed("blocked")
                if has_threat:
                    metrics_exporter_instance.track_detection_result(
                        True, True, True
                    )  # Accurate block
                else:
                    metrics_exporter_instance.track_detection_result(
                        False, True, False
                    )  # False positive
            else:
                metrics_exporter_instance.track_file_processed("approved")

        # Check if decision is blocked
        # Decision can be:
        # - True (boolean) -> allowed
        # - False (boolean) -> blocked
        # - dict with "allowed": True -> allowed
        # - dict with "allowed": False or "status": "DENIED" or "violations" -> blocked
        if decision is True:
            # Explicitly allowed (boolean True)
            is_blocked = False
        elif isinstance(decision, dict):
            # Check dict-based decision
            allowed = decision.get("allowed")
            status = decision.get("status")
            has_violations = "violations" in decision and decision.get("violations")

            # Blocked if explicitly denied or has violations
            is_blocked = (
                allowed is False
                or status == "DENIED"
                or (has_violations and len(decision.get("violations", [])) > 0)
            )
        else:
            # Default to blocked for any other type (including False)
            is_blocked = True

        if is_blocked:
            # Add success field to denial response
            decision["success"] = False
            decision["user_context"] = user_context
            decision["query_context"] = query_context
            decision["retrieval_metadata"] = retrieval_metadata
            print(json.dumps(decision, indent=2))

            # LEARN FROM BLOCKED ATTACK: Add to threat embedding patterns
            if self.threat_embedder and query_context.get("detected_threats"):
                # Extract attack metadata if available
                attack_metadata = {
                    "attack_type": query_context.get("detected_threats", [""])[0]
                    if query_context.get("detected_threats")
                    else "unknown",
                    "config": {"severity": "HIGH"},
                    "attack_intent": "User query blocked by security system",
                }

                # Learn the pattern
                self.threat_embedder.learn_threat_pattern(
                    query=query,
                    user_context=user_context,
                    attack_metadata=attack_metadata,
                    was_blocked=True,
                )

                # Track learning metrics
                if METRICS_ENABLED and metrics_exporter_instance:
                    metrics_exporter_instance.track_learning_event(
                        {
                            "type": "pattern_learned",
                            "attack_type": attack_metadata["attack_type"],
                            "timestamp": datetime.now(UTC).isoformat(),
                        }
                    )

            # Flush any pending batch items
            if self.qwen_client:
                self.qwen_client.flush_batch()

            return False

        # Step 3: Proceed with RAG (only if all checks pass)
        result = self.graph.invoke({"question": query})

        # Create comprehensive success response
        success_response = {
            "success": True,
            "status": "ALLOWED",
            "action": "PROCESS",
            "user_context": user_context,
            "query_context": query_context,
            "retrieval_metadata": retrieval_metadata,
            "answer": result["answer"],
            "policy_context": {
                "rules_applied": ["TenantIsolationPolicy", "TopicScopeRule", "SensitivityRule"],
                "violations_found": 0,
                "compliance_status": "FULL_COMPLIANCE",
            },
            "timestamp": datetime.now(UTC).isoformat(),
            "incident_id": str(uuid.uuid4()),
        }

        print(json.dumps(success_response, indent=2))

        # Flush any pending batch items
        if self.qwen_client:
            self.qwen_client.flush_batch()

        return True
