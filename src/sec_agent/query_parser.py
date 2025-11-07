"""
Query Parser - Extract context from user queries
"""

import re
from typing import Any


def extract_query_context(query: str) -> dict[str, Any]:
    """Extract semantic context from user query"""
    query_lower = query.lower()

    # Extract target tenant
    target_tenant = None
    tenant_pattern = r"tenant([a-z])"
    tenant_match = re.search(tenant_pattern, query_lower)
    if tenant_match:
        target_tenant = f"tenant{tenant_match.group(1).upper()}"

    # Extract intent
    intent = "data_retrieval"  # default
    if any(word in query_lower for word in ["summarize", "summary", "overview"]):
        intent = "summarization"
    elif any(word in query_lower for word in ["analyze", "analysis", "trend"]):
        intent = "analysis"
    elif any(word in query_lower for word in ["compare", "comparison"]):
        intent = "comparison"

    # Extract topics with word boundary matching
    topics = []
    topic_keywords = {
        "finance": ["revenue", "financial", "money", "cost", "budget", "profit", "finance"],
        "marketing": ["marketing", "campaign", "advertisement", "promotion"],
        "hr": ["employee", "staff", "personnel", "hiring", "hr"],
        "retrieval": ["retrieval", "search", "find", "query"],
        "rag": ["rag", "generation", "llm", "ai"],
    }

    # Detect malicious patterns
    malicious_patterns = {
        "prompt_injection": [
            "ignore",
            "override",
            "disregard",
            "bypass",
            "previous instructions",
            "safety constraints",
            "hidden secrets",
            "confidential system data",
        ],
        "data_exfiltration": [
            "export",
            "send",
            "retrieve",
            "print",
            "api keys",
            "credentials",
            "base64",
            "encode",
            "evil.example.com",
            "vector database",
        ],
        "social_engineering": [
            "admin asked",
            "system auditor",
            "disaster recovery",
            "superuser",
            "authorized",
            "disable rls",
            "manager authorized",
            "pretend i'm",
            "auditor",
            "must reveal",
            "for compliance",
        ],
        "obfuscation": [
            "base64:",
            "eval(atob",
            "encoded script",
            "bypass all policy",
            "dump internal",
            "state variables",
        ],
        "jailbreak": [
            "developer mode",
            "unrestricted",
            "private embeddings",
            "root role",
            "forget constraints",
            "switch to root",
            "forget all",
            "previous constraints",
            "display restricted",
            "reveal everything",
            "unrestricted access",
        ],
        "poisoning": [
            "update the system",
            "inject",
            "training data",
            "corrupted example",
            "poison the vector",
            "malicious data",
            "secret_key",
            "password123",
            "ignore security rules",
            "all users have",
            "tenantA can access",
        ],
        "privilege_escalation": [
            "need to access",
            "level documents",
            "clearance",
            "but i need",
            "should be able to",
            "trying to access",
            "requesting",
            "admin access",
            "superuser privileges",
            "bypass clearance",
            "cross-tenant access",
        ],
    }

    detected_threats = []
    for threat_type, patterns in malicious_patterns.items():
        for pattern in patterns:
            if pattern in query_lower:
                detected_threats.append(threat_type)
                break

    for topic, keywords in topic_keywords.items():
        for keyword in keywords:
            # Use word boundary to avoid false matches
            if re.search(r"\b" + re.escape(keyword) + r"\b", query_lower):
                topics.append(topic)
                break

    # Remove duplicates while preserving order
    topics = list(dict.fromkeys(topics))

    return {
        "query": query,
        "intent": intent,
        "target_tenant": target_tenant,
        "topics": topics,
        "detected_threats": detected_threats,
        "phase": "retrieval",
    }
