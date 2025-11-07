"""
Evil Agent - Malicious Prompt Generator (Adversarial Attack Agent)
-------------------------------------------------------------------
Generates adversarial prompts (e.g., prompt injection, jailbreak, social engineering)
for ethical security testing of LLM pipelines and RLS-enforced systems.

This module is organized into separate components following the Single Responsibility Principle:

- llm_client: LLM API client for Google, OpenAI, Flash
- attack_catalog: Static attack templates
- attack_generator: Core attack generation logic
- privilege_escalation: Privilege escalation attack generation
- batch_generator: Batch attack generation
- formatter: Output formatting
- attack_tester: Integration testing
- cli: Command-line interface

Author: VecSec Labs
License: For authorized security testing and research only.
"""

from .attack_catalog import ATTACK_TYPES, get_attack_type, get_attack_types, list_attack_types
from .attack_generator import generate_attack, generate_llm_attack
from .attack_tester import test_attack_against_agent
from .batch_generator import generate_batch
from .cli import main
from .formatter import pretty_print_batch
from .llm_client import LLMClient, get_llm_client
from .privilege_escalation import generate_privilege_escalation_attack

__all__ = [
    # Classes
    "LLMClient",
    # Functions
    "get_llm_client",
    "generate_attack",
    "generate_llm_attack",
    "generate_privilege_escalation_attack",
    "generate_batch",
    "pretty_print_batch",
    "test_attack_against_agent",
    "get_attack_types",
    "get_attack_type",
    "list_attack_types",
    "main",
    # Constants
    "ATTACK_TYPES",
]
