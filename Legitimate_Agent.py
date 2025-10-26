#!/usr/bin/env python3
"""
Legitimate Agent - Sends data with proper permissions for legitimate operations

This agent is designed to perform legitimate operations with appropriate
role-based permissions, serving as a contrast to the malicious Evil_Agent.py
"""

import json
import argparse
import uuid
from datetime import datetime
from typing import Dict, Any, List
import random

# --- Legitimate Operation Templates ------------------------------------------

LEGITIMATE_OPERATIONS = {
    "data_retrieval": {
        "admin": [
            "Retrieve user access logs for security audit",
            "Export system configuration for backup purposes", 
            "Generate compliance report for tenantA data",
            "Access cross-tenant analytics for system monitoring"
        ],
        "superuser": [
            "Analyze system performance metrics",
            "Generate user activity reports",
            "Access configuration data for maintenance",
            "Review security logs for incident investigation"
        ],
        "analyst": [
            "Query marketing campaign performance data",
            "Retrieve RAG system usage statistics",
            "Access internal documentation for analysis",
            "Generate reports on LangChain implementation"
        ],
        "guest": [
            "View public documentation",
            "Access general system information",
            "Query basic usage statistics",
            "Retrieve public knowledge base articles"
        ]
    },
    "data_analysis": {
        "admin": [
            "Analyze security incident patterns across all tenants",
            "Generate comprehensive system health report",
            "Perform cross-tenant data correlation analysis",
            "Create executive summary of system performance"
        ],
        "superuser": [
            "Analyze user behavior patterns",
            "Generate system performance trends",
            "Create maintenance schedule recommendations",
            "Analyze resource utilization metrics"
        ],
        "analyst": [
            "Analyze marketing campaign effectiveness",
            "Generate RAG system performance insights",
            "Create user engagement reports",
            "Analyze content retrieval patterns"
        ],
        "guest": [
            "View basic system statistics",
            "Access public performance metrics",
            "Query general usage information",
            "View public analytics dashboard"
        ]
    },
    "system_maintenance": {
        "admin": [
            "Update system configuration for security hardening",
            "Modify access policies for compliance requirements",
            "Configure cross-tenant data sharing rules",
            "Update system-wide security parameters"
        ],
        "superuser": [
            "Update user permissions and roles",
            "Modify system performance settings",
            "Configure monitoring and alerting rules",
            "Update maintenance schedules"
        ],
        "analyst": [
            "Update content categorization rules",
            "Modify search and retrieval parameters",
            "Configure reporting templates",
            "Update data processing workflows"
        ],
        "guest": [
            "Update personal preferences",
            "Modify notification settings",
            "Configure display options",
            "Update basic user settings"
        ]
    }
}

# --- Role-Based Clearance Mapping -------------------------------------------

ROLE_CLEARANCE_MAPPING = {
    "admin": "SECRET",
    "superuser": "CONFIDENTIAL", 
    "analyst": "INTERNAL",
    "guest": "PUBLIC"
}

# --- Legitimate Agent Functions ---------------------------------------------

def generate_legitimate_operation(
    user_id: str,
    tenant_id: str,
    role: str,
    operation_type: str = None,
    seed: int = None
) -> Dict[str, Any]:
    """
    Generate a legitimate operation with proper role-based permissions
    """
    
    if seed:
        random.seed(seed)
    
    # Validate role
    if role not in ROLE_CLEARANCE_MAPPING:
        role = "guest"  # Default to least privileged
    
    # Get appropriate clearance for role
    clearance = ROLE_CLEARANCE_MAPPING[role]
    
    # Select operation type if not specified
    if not operation_type or operation_type not in LEGITIMATE_OPERATIONS:
        operation_type = random.choice(list(LEGITIMATE_OPERATIONS.keys()))
    
    # Get appropriate operation for role
    role_operations = LEGITIMATE_OPERATIONS[operation_type].get(role, [])
    if not role_operations:
        # Fallback to guest operations if role not found
        role_operations = LEGITIMATE_OPERATIONS[operation_type]["guest"]
    
    query = random.choice(role_operations)
    
    operation_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()
    
    example = {
        "user_id": user_id,
        "tenant_id": tenant_id,
        "clearance": clearance,
        "role": role,
        "query": query
    }
    
    metadata = {
        "operation_id": operation_id,
        "operation_type": operation_type,
        "operation_intent": f"Legitimate {operation_type} operation for {role} role",
        "random_seed": seed,
        "config": {
            "severity": "LEGITIMATE",
            "authorization_level": "APPROPRIATE",
            "risk_level": "LOW"
        },
        "timestamp": timestamp,
        "ethical_notice": "This is a legitimate operation with appropriate permissions.",
        "generation_method": "legitimate_template",
        "role_clearance_mapping": {
            "role": role,
            "clearance": clearance,
            "appropriate": True
        }
    }
    
    return {
        "example": example,
        "metadata": metadata
    }

def generate_legitimate_batch(
    user_id: str,
    tenant_id: str,
    role: str,
    operation_types: List[str] = None,
    count_per_type: int = 1,
    seed: int = None
) -> List[Dict[str, Any]]:
    """
    Generate multiple legitimate operations
    """
    
    if seed:
        random.seed(seed)
    
    results = []
    selected_types = operation_types or list(LEGITIMATE_OPERATIONS.keys())
    
    for operation_type in selected_types:
        for _ in range(count_per_type):
            results.append(generate_legitimate_operation(
                user_id, tenant_id, role, operation_type, seed
            ))
    
    return results

def pretty_print_operation(operation: Dict[str, Any]) -> None:
    """Pretty print a single legitimate operation"""
    
    example = operation["example"]
    metadata = operation["metadata"]
    
    print(f"🔒 LEGITIMATE OPERATION")
    print(f"👤 User: {example['user_id']}")
    print(f"🏢 Tenant: {example['tenant_id']}")
    print(f"🔐 Role: {example['role']}")
    print(f"📊 Clearance: {example['clearance']}")
    print(f"📝 Query: {example['query']}")
    print(f"🆔 Operation ID: {metadata['operation_id']}")
    print(f"⚡ Type: {metadata['operation_type']}")
    print(f"✅ Authorization: {metadata['config']['authorization_level']}")
    print(f"📅 Generated: {metadata['timestamp']}")
    print("-" * 60)

def pretty_print_batch(batch: List[Dict[str, Any]]) -> None:
    """Pretty print multiple legitimate operations"""
    
    print(f"🔒 LEGITIMATE OPERATIONS BATCH ({len(batch)} operations)")
    print("=" * 80)
    
    for i, operation in enumerate(batch, 1):
        print(f"\n📋 Operation {i}:")
        pretty_print_operation(operation)
    
    print("\n✅ All operations generated with appropriate role-based permissions")

# --- CLI --------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Legitimate Agent - Generate operations with proper permissions"
    )
    parser.add_argument("--user-id", default="legitimate_user", help="User ID")
    parser.add_argument("--tenant-id", default="tenantA", help="Tenant ID")
    parser.add_argument("--role", default="analyst", 
                       choices=["admin", "superuser", "analyst", "guest"],
                       help="User role")
    parser.add_argument("--operation-type", default=None,
                       choices=["data_retrieval", "data_analysis", "system_maintenance"],
                       help="Specific operation type to generate")
    parser.add_argument("--batch", type=int, default=0, 
                       help="Generate multiple operations per type")
    parser.add_argument("--seed", type=int, default=None, 
                       help="Optional random seed")
    
    args = parser.parse_args()
    
    if args.batch > 0:
        batch = generate_legitimate_batch(
            args.user_id, args.tenant_id, args.role,
            [args.operation_type] if args.operation_type else None,
            args.batch, args.seed
        )
        pretty_print_batch(batch)
    else:
        operation = generate_legitimate_operation(
            args.user_id, args.tenant_id, args.role, 
            args.operation_type, args.seed
        )
        print(json.dumps(operation, indent=2))
        print("\n  This is a legitimate operation with appropriate permissions.")
