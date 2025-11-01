"""
Policy Management - Tenant and Role Policies
"""

# --- Mock RLSA Policy Table
TENANT_POLICIES = {
    "tenantA": {"clearance": "INTERNAL", "topics": ["retrieval", "RAG", "LangChain", "marketing"], "sensitivity": "INTERNAL"},
    "tenantB": {"clearance": "CONFIDENTIAL", "topics": ["finance", "policy", "marketing"], "sensitivity": "CONFIDENTIAL"},
}

# Role-based access policies
ROLE_POLICIES = {
    "admin": {
        "allowed_operations": ["read", "write", "delete", "configure"],
        "max_clearance": "SECRET",
        "cross_tenant_access": True,
        "bypass_restrictions": ["topic_scope", "clearance_level"]
    },
    "superuser": {
        "allowed_operations": ["read", "write", "configure"],
        "max_clearance": "CONFIDENTIAL", 
        "cross_tenant_access": False,
        "bypass_restrictions": ["topic_scope"]
    },
    "analyst": {
        "allowed_operations": ["read"],
        "max_clearance": "INTERNAL",
        "cross_tenant_access": False,
        "bypass_restrictions": []
    },
    "guest": {
        "allowed_operations": ["read"],
        "max_clearance": "PUBLIC",
        "cross_tenant_access": False,
        "bypass_restrictions": []
    }
}


def get_tenant_policy(tenant_id: str) -> dict:
    """Get policy for a specific tenant"""
    return TENANT_POLICIES.get(tenant_id, {})


def get_role_policy(role: str) -> dict:
    """Get policy for a specific role"""
    return ROLE_POLICIES.get(role, ROLE_POLICIES["guest"])

