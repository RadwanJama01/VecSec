"""
Policy Management - Tenant and Role Policies
Refactored with immutability, validation, and configurability support
"""

import os
import json
from pathlib import Path
from types import MappingProxyType
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from copy import deepcopy


# ============================================================================
# Clearance Hierarchy (ordered from low to high)
# ============================================================================

CLEARANCE_LEVELS = ["PUBLIC", "INTERNAL", "CONFIDENTIAL", "SECRET"]
CLEARANCE_HIERARCHY = {level: idx for idx, level in enumerate(CLEARANCE_LEVELS)}


# ============================================================================
# Data Classes for Immutable Policies
# ============================================================================

@dataclass(frozen=True)
class TenantPolicy:
    """Immutable tenant policy"""
    clearance: str
    topics: List[str]
    sensitivity: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "clearance": self.clearance,
            "topics": self.topics.copy(),
            "sensitivity": self.sensitivity
        }


@dataclass(frozen=True)
class RolePolicy:
    """Immutable role policy"""
    allowed_operations: List[str]
    max_clearance: str
    cross_tenant_access: bool
    bypass_restrictions: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "allowed_operations": self.allowed_operations.copy(),
            "max_clearance": self.max_clearance,
            "cross_tenant_access": self.cross_tenant_access,
            "bypass_restrictions": self.bypass_restrictions.copy()
        }


# ============================================================================
# Default Policies (hardcoded fallback)
# ============================================================================

_DEFAULT_TENANT_POLICIES = {
    "tenantA": {
        "clearance": "INTERNAL",
        "topics": ["retrieval", "RAG", "LangChain", "marketing"],
        "sensitivity": "INTERNAL"
    },
    "tenantB": {
        "clearance": "CONFIDENTIAL",
        "topics": ["finance", "policy", "marketing"],
        "sensitivity": "CONFIDENTIAL"
    }
}

_DEFAULT_ROLE_POLICIES = {
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


# ============================================================================
# Policy Loading (with JSON support)
# ============================================================================

def _load_policies_from_json(policy_file: Optional[str] = None) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """
    Load policies from JSON file or use defaults
    
    Args:
        policy_file: Path to JSON policy file (optional)
        
    Returns:
        Tuple of (tenant_policies_dict, role_policies_dict)
    """
    if policy_file and Path(policy_file).exists():
        try:
            with open(policy_file, 'r') as f:
                data = json.load(f)
                tenant_policies = data.get("tenant_policies", _DEFAULT_TENANT_POLICIES)
                role_policies = data.get("role_policies", _DEFAULT_ROLE_POLICIES)
                print(f"✅ Loaded policies from {policy_file}")
                return tenant_policies, role_policies
        except Exception as e:
            print(f"⚠️  Failed to load policies from {policy_file}: {e}, using defaults")
    
    # Use defaults
    return _DEFAULT_TENANT_POLICIES.copy(), _DEFAULT_ROLE_POLICIES.copy()


def _create_immutable_policies(
    tenant_policies_dict: Dict[str, Dict],
    role_policies_dict: Dict[str, Dict]
) -> Tuple[Dict[str, TenantPolicy], Dict[str, RolePolicy]]:
    """
    Convert policy dicts to immutable dataclass instances
    
    Returns:
        Tuple of (tenant_policies, role_policies) with frozen dataclasses
    """
    tenant_policies = {}
    for tenant_id, policy_data in tenant_policies_dict.items():
        tenant_policies[tenant_id] = TenantPolicy(
            clearance=policy_data["clearance"],
            topics=policy_data.get("topics", []),
            sensitivity=policy_data.get("sensitivity", "INTERNAL")
        )
    
    role_policies = {}
    for role, policy_data in role_policies_dict.items():
        role_policies[role] = RolePolicy(
            allowed_operations=policy_data.get("allowed_operations", []),
            max_clearance=policy_data["max_clearance"],
            cross_tenant_access=policy_data.get("cross_tenant_access", False),
            bypass_restrictions=policy_data.get("bypass_restrictions", [])
        )
    
    return tenant_policies, role_policies


# Load policies (from JSON if available, otherwise defaults)
_policy_file = os.getenv("POLICY_FILE", None)
_tenant_policies_dict, _role_policies_dict = _load_policies_from_json(_policy_file)
_tenant_policies_immutable, _role_policies_immutable = _create_immutable_policies(_tenant_policies_dict, _role_policies_dict)

# Convert to read-only dict views for backward compatibility
# These are dict-like objects that prevent modification (immutable)
TENANT_POLICIES = MappingProxyType({
    k: v.to_dict() for k, v in _tenant_policies_immutable.items()
})
ROLE_POLICIES = MappingProxyType({
    k: v.to_dict() for k, v in _role_policies_immutable.items()
})


# ============================================================================
# Policy Access Functions (with validation)
# ============================================================================

def get_tenant_policy(tenant_id: str) -> Dict[str, Any]:
    """
    Get policy for a specific tenant
    
    Args:
        tenant_id: Tenant identifier
        
    Returns:
        Policy dictionary (read-only copy)
        
    Raises:
        ValueError: If tenant_id is None, empty, or not found
    """
    if tenant_id is None:
        raise ValueError("tenant_id cannot be None")
    if not tenant_id or not isinstance(tenant_id, str):
        raise ValueError(f"tenant_id must be a non-empty string, got: {tenant_id}")
    
    if tenant_id not in TENANT_POLICIES:
        raise ValueError(f"Tenant '{tenant_id}' not found in policies. Available tenants: {list(TENANT_POLICIES.keys())}")
    
    # Return a copy to prevent modification
    policy = TENANT_POLICIES[tenant_id]
    return deepcopy(policy)


def get_role_policy(role: str) -> Dict[str, Any]:
    """
    Get policy for a specific role
    
    Args:
        role: Role identifier
        
    Returns:
        Policy dictionary (read-only copy)
        
    Raises:
        ValueError: If role is None or empty
    """
    if role is None:
        raise ValueError("role cannot be None")
    if not role or not isinstance(role, str):
        raise ValueError(f"role must be a non-empty string, got: {role}")
    
    if role not in ROLE_POLICIES:
        raise ValueError(f"Role '{role}' not found in policies. Available roles: {list(ROLE_POLICIES.keys())}")
    
    # Return a copy to prevent modification
    policy = ROLE_POLICIES[role]
    return deepcopy(policy)


# ============================================================================
# Validation Functions
# ============================================================================

def validate_tenant_policy(tenant_id: str) -> bool:
    """
    Validate that tenant policy exists
    
    Args:
        tenant_id: Tenant identifier
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    get_tenant_policy(tenant_id)  # Raises ValueError if invalid
    return True


def validate_role_policy(role: str) -> bool:
    """
    Validate that role policy exists
    
    Args:
        role: Role identifier
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    get_role_policy(role)  # Raises ValueError if invalid
    return True


def can_access_clearance(user_clearance: str, required_clearance: str) -> bool:
    """
    Check if user clearance level is sufficient for required clearance
    
    Args:
        user_clearance: User's clearance level
        required_clearance: Required clearance level
        
    Returns:
        True if user can access, False otherwise
        
    Raises:
        ValueError: If clearance levels are invalid
    """
    if user_clearance not in CLEARANCE_HIERARCHY:
        raise ValueError(f"Invalid user clearance: {user_clearance}. Valid levels: {CLEARANCE_LEVELS}")
    if required_clearance not in CLEARANCE_HIERARCHY:
        raise ValueError(f"Invalid required clearance: {required_clearance}. Valid levels: {CLEARANCE_LEVELS}")
    
    user_level = CLEARANCE_HIERARCHY[user_clearance]
    required_level = CLEARANCE_HIERARCHY[required_clearance]
    
    return user_level >= required_level


def can_access_topic(tenant_id: str, topic: str) -> bool:
    """
    Check if tenant has access to a specific topic
    
    Args:
        tenant_id: Tenant identifier
        topic: Topic to check
        
    Returns:
        True if tenant can access topic, False otherwise
        
    Raises:
        ValueError: If tenant_id is invalid
    """
    policy = get_tenant_policy(tenant_id)  # Raises ValueError if invalid
    return topic in policy.get("topics", [])


def can_access_tenant(user_tenant: str, target_tenant: str, role: str) -> bool:
    """
    Check if user can access target tenant (cross-tenant access)
    
    Args:
        user_tenant: User's tenant
        target_tenant: Target tenant to access
        role: User's role
        
    Returns:
        True if user can access target tenant, False otherwise
        
    Raises:
        ValueError: If tenant_id or role is invalid
    """
    if user_tenant == target_tenant:
        return True  # Same tenant always allowed
    
    role_policy = get_role_policy(role)  # Raises ValueError if invalid
    return role_policy.get("cross_tenant_access", False)


def can_bypass_restriction(role: str, restriction: str) -> bool:
    """
    Check if role can bypass a specific restriction
    
    Args:
        role: User's role
        restriction: Restriction to bypass (e.g., "topic_scope", "clearance_level")
        
    Returns:
        True if role can bypass restriction, False otherwise
        
    Raises:
        ValueError: If role is invalid
    """
    role_policy = get_role_policy(role)  # Raises ValueError if invalid
    return restriction in role_policy.get("bypass_restrictions", [])


def has_operation_permission(role: str, operation: str) -> bool:
    """
    Check if role has permission for a specific operation
    
    Args:
        role: User's role
        operation: Operation to check (e.g., "read", "write", "delete")
        
    Returns:
        True if role has permission, False otherwise
        
    Raises:
        ValueError: If role is invalid
    """
    role_policy = get_role_policy(role)  # Raises ValueError if invalid
    return operation in role_policy.get("allowed_operations", [])


# ============================================================================
# Policy Comparison Functions
# ============================================================================

def compare_clearance_levels(clearance1: str, clearance2: str) -> int:
    """
    Compare two clearance levels
    
    Args:
        clearance1: First clearance level
        clearance2: Second clearance level
        
    Returns:
        -1 if clearance1 < clearance2
         0 if clearance1 == clearance2
         1 if clearance1 > clearance2
        
    Raises:
        ValueError: If clearance levels are invalid
    """
    if clearance1 not in CLEARANCE_HIERARCHY:
        raise ValueError(f"Invalid clearance: {clearance1}. Valid levels: {CLEARANCE_LEVELS}")
    if clearance2 not in CLEARANCE_HIERARCHY:
        raise ValueError(f"Invalid clearance: {clearance2}. Valid levels: {CLEARANCE_LEVELS}")
    
    level1 = CLEARANCE_HIERARCHY[clearance1]
    level2 = CLEARANCE_HIERARCHY[clearance2]
    
    if level1 < level2:
        return -1
    elif level1 > level2:
        return 1
    else:
        return 0


# Note: TENANT_POLICIES and ROLE_POLICIES are now MappingProxyType objects
# They behave like dicts (can use [], .get(), .keys(), etc.) but are read-only
# Attempts to modify them will raise TypeError
# 
# Example usage:
#   policy = TENANT_POLICIES["tenantA"]  # Works (dict-like access)
#   policy = TENANT_POLICIES.get("tenantA", {})  # Works (.get() method)
#   TENANT_POLICIES["new"] = {}  # Raises TypeError (read-only)
