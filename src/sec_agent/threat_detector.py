"""
Contextual Threat Embedding - Threat pattern learning and detection
"""

from datetime import UTC, datetime
from typing import Any

import numpy as np

from .embeddings_client import EmbeddingClient


class ContextualThreatEmbedding:
    """Maintains context while creating embeddings for threats"""

    def __init__(self, qwen_client: EmbeddingClient):
        self.qwen = qwen_client
        self.learned_patterns: list[dict[str, Any]] = []
        self.max_patterns = 200  # Limit stored patterns

    def create_contextual_prompt(
        self,
        query: str,
        role: str,
        clearance: str,
        tenant_id: str,
        attack_type: str | None = None,
        attack_metadata: dict[Any, Any] | None = None,
    ) -> str:
        """Create structured prompt for embeddings"""

        contextual_prompt = f"""ROLE: {role}
CLEARANCE: {clearance}
TENANT: {tenant_id}
QUERY: {query}"""

        if attack_type:
            contextual_prompt += f"\nATTACK_TYPE: {attack_type}"

        if attack_metadata:
            severity = attack_metadata.get("config", {}).get("severity", "UNKNOWN")
            intent = attack_metadata.get("attack_intent", "unknown")
            contextual_prompt += f"\nSEVERITY: {severity}\nINTENT: {intent}"

        return contextual_prompt.strip()

    def learn_threat_pattern(
        self,
        query: str,
        user_context: dict,
        attack_metadata: dict[Any, Any] | None = None,
        was_blocked: bool = False,
    ):
        """Learn a threat pattern with full context"""

        prompt = self.create_contextual_prompt(
            query=query,
            role=user_context.get("role", "analyst"),
            clearance=user_context.get("clearance", "INTERNAL"),
            tenant_id=user_context.get("tenant_id", "tenantA"),
            attack_type=attack_metadata.get("attack_type") if attack_metadata else None,
            attack_metadata=attack_metadata,
        )

        embedding = self.qwen.get_embedding(prompt)

        pattern = {
            "embedding": embedding,
            "query": query,
            "user_context": user_context,
            "attack_metadata": attack_metadata,
            "was_blocked": was_blocked,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        self.learned_patterns.append(pattern)

        # Limit stored patterns (keep only most recent)
        if len(self.learned_patterns) > self.max_patterns:
            self.learned_patterns = self.learned_patterns[-self.max_patterns :]

        # Notify client about patterns learned
        self.qwen.set_patterns_learned(len(self.learned_patterns))

        return pattern

    def check_semantic_threat(
        self, query: str, user_context: dict, similarity_threshold: float = 0.85
    ) -> tuple:
        """Check if query is semantically similar to known threats"""

        if not self.learned_patterns:
            return False, {}

        # Create contextual prompt
        prompt = self.create_contextual_prompt(
            query=query,
            role=user_context.get("role", "analyst"),
            clearance=user_context.get("clearance", "INTERNAL"),
            tenant_id=user_context.get("tenant_id", "tenantA"),
        )

        # Get embedding for query
        query_embedding = self.qwen.get_embedding(prompt)

        # Check similarity against learned patterns
        for pattern in self.learned_patterns:
            if pattern.get("was_blocked", True):  # Only check blocked patterns
                try:
                    similarity = np.dot(query_embedding, pattern["embedding"]) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(pattern["embedding"])
                    )

                    if similarity > similarity_threshold:
                        return True, {
                            "threat_detected": True,
                            "similarity_score": float(similarity),
                            "matched_pattern": pattern["query"][:100],
                            "detection_method": "semantic_similarity",
                        }
                except Exception:
                    continue

        return False, {"detection_method": "semantic_similarity"}
