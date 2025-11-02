"""
Batch Attack Generator
Orchestrates multiple attack generations across several attack categories
"""

from typing import List, Dict, Any

from .attack_generator import generate_attack
from .attack_catalog import ATTACK_TYPES


def generate_batch(
    user_id: str,
    tenant_id: str,
    clearance: str,
    attack_types: List[str] = None,
    count_per_type: int = 1,
    use_llm: bool = False,
    llm_provider: str = "google",
    role: str = "analyst"
) -> List[Dict[str, Any]]:
    """
    Generate multiple adversarial prompts across several attack categories.
    """

    results = []
    selected_types = attack_types or list(ATTACK_TYPES.keys())

    for attack_type in selected_types:
        for _ in range(count_per_type):
            results.append(generate_attack(
                user_id, tenant_id, clearance, attack_type, 
                use_llm=use_llm, llm_provider=llm_provider, role=role
            ))

    return results

