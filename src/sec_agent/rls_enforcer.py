"""
RLS Enforcer - Comprehensive Row-Level Security enforcement
"""

import uuid
from datetime import UTC, datetime
from typing import Any

from .config import METRICS_ENABLED
from .policy_manager import ROLE_POLICIES, TENANT_POLICIES

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


def rlsa_guard_comprehensive(
    user_context: dict[str, Any],
    query_context: dict[str, Any],
    retrieval_metadata: list[dict[str, Any]],
    threat_embedder=None,
) -> dict[str, Any]:
    """Comprehensive RLS enforcement with full context"""
    user_tenant = user_context["tenant_id"]
    user_clearance = user_context["clearance"]
    user_role = user_context["role"]
    target_tenant = query_context.get("target_tenant", user_tenant)
    topics = query_context.get("topics", [])

    violations = []
    detected_threats = query_context.get("detected_threats", [])

    # Get role policy
    role_policy = ROLE_POLICIES.get(user_role, ROLE_POLICIES["guest"])

    # Initialize policy context early
    policy_context = {
        "user_tenant": user_tenant,
        "target_tenant": target_tenant,
        "user_clearance": user_clearance,
        "user_role": user_role,
        "role_policy": role_policy,
        "query_topics": topics,
        "detected_threats": detected_threats,
        "rules_applied": [],
    }

    # SEMANTIC THREAT DETECTION (new layer)
    # Check if query is semantically similar to known attack patterns
    query = query_context.get("query", "")
    if threat_embedder:
        is_semantic_threat, semantic_result = threat_embedder.check_semantic_threat(
            query, user_context
        )

        if is_semantic_threat:
            violations.append(
                {
                    "type": "semantic_threat",
                    "rule": "SemanticSimilarityDetection",
                    "severity": "HIGH",
                    "message": f"Query matches known attack pattern (similarity: {semantic_result.get('similarity_score', 0):.2f})",
                    "similarity_score": semantic_result.get("similarity_score"),
                    "matched_pattern": semantic_result.get("matched_pattern"),
                    "detection_method": "semantic_similarity",
                }
            )
            policy_context["rules_applied"].append("SemanticSimilarityDetection")

    # 0. Malicious Threat Detection (highest priority)
    if detected_threats:
        # Define threat categories
        always_blocked_threats = [
            "prompt_injection",
            "obfuscation",
            "jailbreak",
            "privilege_escalation",
        ]
        role_dependent_threats = ["data_exfiltration", "social_engineering"]

        # Check for always blocked threats
        always_blocked_detected = [
            threat for threat in detected_threats if threat in always_blocked_threats
        ]
        role_dependent_detected = [
            threat for threat in detected_threats if threat in role_dependent_threats
        ]

        # Always block these threats regardless of role
        if always_blocked_detected:
            violations.append(
                {
                    "type": "malicious_threat",
                    "rule": "ThreatDetectionPolicy",
                    "severity": "CRITICAL",
                    "message": f"Always-blocked malicious patterns detected: {', '.join(always_blocked_detected)}",
                    "detected_threats": always_blocked_detected,
                    "threat_count": len(always_blocked_detected),
                    "threat_category": "always_blocked",
                }
            )
            policy_context["rules_applied"].append("ThreatDetectionPolicy")

        # Check role-dependent threats
        if role_dependent_detected:
            # Check if user role has privilege to bypass these threats
            privileged_roles = ["admin", "superuser"]
            can_bypass = user_role in privileged_roles

            if not can_bypass:
                violations.append(
                    {
                        "type": "malicious_threat",
                        "rule": "ThreatDetectionPolicy",
                        "severity": "HIGH",
                        "message": f"Role-dependent malicious patterns detected: {', '.join(role_dependent_detected)} (role {user_role} not authorized to bypass)",
                        "detected_threats": role_dependent_detected,
                        "threat_count": len(role_dependent_detected),
                        "threat_category": "role_dependent",
                        "user_role": user_role,
                        "can_bypass": can_bypass,
                        "privileged_roles": privileged_roles,
                    }
                )
                policy_context["rules_applied"].append("ThreatDetectionPolicy")
            else:
                # Log the bypass for audit purposes
                policy_context["rules_applied"].append("ThreatDetectionPolicy_Bypassed")
                policy_context["bypassed_threats"] = {
                    "threats": role_dependent_detected,
                    "reason": f"Role {user_role} has privilege to bypass",
                    "timestamp": datetime.now(UTC).isoformat(),
                }

    # 0.5. Role-based Access Control and Clearance Level Check
    clearance_levels = {"PUBLIC": 1, "INTERNAL": 2, "CONFIDENTIAL": 3, "SECRET": 4}
    user_level = clearance_levels.get(user_clearance, 1)
    role_max_level = clearance_levels.get(role_policy["max_clearance"], 1)

    # Check if user's clearance exceeds role's max clearance
    if user_level > role_max_level:
        violations.append(
            {
                "type": "role_clearance_violation",
                "rule": "RoleBasedAccessControl",
                "severity": "HIGH",
                "message": f"User clearance {user_clearance} exceeds role {user_role} max clearance {role_policy['max_clearance']}",
                "user_clearance": user_clearance,
                "role_max_clearance": role_policy["max_clearance"],
                "user_role": user_role,
            }
        )
        policy_context["rules_applied"].append("RoleBasedAccessControl")

    # Check if query is requesting content above user's clearance level
    # Only trigger on explicit requests for higher clearance content
    query_lower = query.lower()
    requested_clearance = None
    requested_level = user_level  # Default to user level

    # Detect EXPLICIT requests for specific clearance levels (not just mentions)
    explicit_keywords = {
        "SECRET": [
            "show me secret",
            "retrieve secret",
            "get secret",
            "access secret",
            "i need secret",
            "give me secret",
            "return secret",
            "need to see secret",
            "show me classified",
            "i need classified",
            "get classified",
        ],
        "CONFIDENTIAL": [
            "show me confidential",
            "retrieve confidential",
            "get confidential",
            "i need confidential",
            "give me confidential",
        ],
        "INTERNAL": [
            "show me internal",
            "retrieve internal",
            "get internal",
            "i need internal",
            "company financial data",
        ],
    }

    for clearance_type, keywords in explicit_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            requested_clearance = clearance_type
            requested_level = clearance_levels.get(clearance_type, user_level)
            break

    # Only block if explicitly requesting content above user's clearance
    if requested_clearance and requested_level > user_level:
        violations.append(
            {
                "type": "insufficient_clearance_violation",
                "rule": "ClearanceLevelEnforcement",
                "severity": "HIGH",
                "message": f"Query explicitly requests {requested_clearance} content but user only has {user_clearance} clearance",
                "requested_clearance": requested_clearance,
                "user_clearance": user_clearance,
                "user_role": user_role,
                "clearance_level_difference": clearance_levels.get(requested_clearance, 0)
                - user_level,
            }
        )
        policy_context["rules_applied"].append("ClearanceLevelEnforcement")

    # 1. Tenant Isolation Check (with role bypass)
    if target_tenant and target_tenant != user_tenant and not role_policy["cross_tenant_access"]:
        violations.append(
            {
                "type": "cross_tenant_violation",
                "rule": "TenantIsolationPolicy",
                "severity": "CRITICAL",
                "message": f"User from {user_tenant} attempting to access {target_tenant} data (role {user_role} not authorized)",
                "required_tenant": user_tenant,
                "violating_tenant": target_tenant,
                "user_role": user_role,
                "role_cross_tenant_access": role_policy["cross_tenant_access"],
            }
        )
        policy_context["rules_applied"].append("TenantIsolationPolicy")

    # 2. Topic Scope Check (case-insensitive, with role bypass)
    if "topic_scope" not in role_policy["bypass_restrictions"]:
        allowed_topics = TENANT_POLICIES.get(user_tenant, {}).get("topics", [])
        allowed_topics_lower = [topic.lower() for topic in allowed_topics]
        forbidden_topics = [topic for topic in topics if topic.lower() not in allowed_topics_lower]

        if forbidden_topics:
            violations.append(
                {
                    "type": "topic_violation",
                    "rule": "TopicScopeRule",
                    "severity": "HIGH",
                    "message": f"Query references forbidden topics: {forbidden_topics} (role {user_role} not authorized to bypass)",
                    "allowed_topics": allowed_topics,
                    "forbidden_topics": forbidden_topics,
                    "user_role": user_role,
                    "role_bypass_topic_scope": "topic_scope" in role_policy["bypass_restrictions"],
                }
            )
            policy_context["rules_applied"].append("TopicScopeRule")

    # 3. Sensitivity vs Clearance Check (with role bypass)
    if "clearance_level" not in role_policy["bypass_restrictions"]:
        for metadata in retrieval_metadata:
            doc_sensitivity = metadata.get("sensitivity", "INTERNAL")
            doc_level = clearance_levels.get(doc_sensitivity, 2)

            if doc_level > user_level:
                violations.append(
                    {
                        "type": "clearance_violation",
                        "rule": "SensitivityRule",
                        "severity": "HIGH",
                        "message": f"Document sensitivity {doc_sensitivity} exceeds user clearance {user_clearance} (role {user_role} not authorized to bypass)",
                        "document_id": metadata.get("document_id"),
                        "required_clearance": doc_sensitivity,
                        "user_clearance": user_clearance,
                        "user_role": user_role,
                        "role_bypass_clearance": "clearance_level"
                        in role_policy["bypass_restrictions"],
                    }
                )
                policy_context["rules_applied"].append("SensitivityRule")

    # 4. Cross-tenant document access check
    for metadata in retrieval_metadata:
        doc_tenant = metadata.get("tenant_id")
        if doc_tenant != user_tenant:
            violations.append(
                {
                    "type": "document_tenant_violation",
                    "rule": "DocumentTenantIsolation",
                    "severity": "CRITICAL",
                    "message": f"Attempting to access document from {doc_tenant}",
                    "document_id": metadata.get("document_id"),
                    "document_tenant": doc_tenant,
                    "user_tenant": user_tenant,
                }
            )
            policy_context["rules_applied"].append("DocumentTenantIsolation")

    # Return result
    if violations:
        # Track metrics for blocked requests
        if METRICS_ENABLED and metrics_exporter_instance:
            for violation in violations:
                if violation.get("type") == "malicious_threat":
                    attack_type = (
                        violation.get("detected_threats", ["unknown"])[0]
                        if violation.get("detected_threats")
                        else "unknown"
                    )
                    if metrics_exporter_instance:
                        metrics_exporter_instance.track_attack_blocked(attack_type)
                if metrics_exporter_instance:
                    metrics_exporter_instance.track_threat_detected(
                        violation.get("type", "unknown")
                    )
                if "insufficient_clearance_violation" in violation.get("type", ""):
                    if metrics_exporter_instance:
                        metrics_exporter_instance.track_rlsa_violation("clearance_level")

        return {
            "status": "DENIED",
            "action": "BLOCK",
            "reason": "multiple_policy_violations",
            "violations": violations,
            "policy_context": policy_context,
            "detection_layers": {
                "malicious_threat_detected": any(
                    v["type"] == "malicious_threat" for v in violations
                ),
                "always_blocked_threats_detected": any(
                    v.get("threat_category") == "always_blocked" for v in violations
                ),
                "role_dependent_threats_detected": any(
                    v.get("threat_category") == "role_dependent" for v in violations
                ),
                "prompt_injection_detected": any(
                    "prompt_injection" in v.get("detected_threats", []) for v in violations
                ),
                "data_exfiltration_detected": any(
                    "data_exfiltration" in v.get("detected_threats", []) for v in violations
                ),
                "social_engineering_detected": any(
                    "social_engineering" in v.get("detected_threats", []) for v in violations
                ),
                "obfuscation_detected": any(
                    "obfuscation" in v.get("detected_threats", []) for v in violations
                ),
                "jailbreak_detected": any(
                    "jailbreak" in v.get("detected_threats", []) for v in violations
                ),
                "poisoning_detected": any(
                    "poisoning" in v.get("detected_threats", []) for v in violations
                ),
                "malware_detected": False,
                "semantic_violation": True,
                "cross_tenant_detected": any(
                    v["type"] == "cross_tenant_violation" for v in violations
                ),
                "clearance_violation_detected": any(
                    v["type"] == "clearance_violation" for v in violations
                ),
                "role_violation_detected": any(
                    v["type"] == "role_clearance_violation" for v in violations
                ),
            },
            "tenant_context": {"requester_tenant": user_tenant, "target_tenant": target_tenant},
            "timestamp": datetime.now(UTC).isoformat(),
            "incident_id": str(uuid.uuid4()),
            "recommendation": "Review access permissions or adjust query scope to comply with tenant isolation and clearance policies.",
        }

    return {"allowed": True, "reason": "All checks passed"}  # All checks passed
