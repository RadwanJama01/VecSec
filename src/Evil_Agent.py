"""
Evil Agent - Backward Compatibility Wrapper
============================================
This file maintains backward compatibility with existing imports.
New code should import from src.evil_agent package directly.

Example:
    from src.evil_agent import generate_attack, generate_privilege_escalation_attack
    # or
    from src.evil_agent.cli import main
"""

import sys
from pathlib import Path

# Add project root to path if needed
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import components for backward compatibility
try:
    from src.evil_agent import (
        LLMClient,
        ATTACK_TYPES,
        generate_attack,
        generate_llm_attack,
        generate_privilege_escalation_attack,
        generate_batch,
        pretty_print_batch,
        test_attack_against_agent,
        get_llm_client
    )
except ImportError:
    # Fallback for different import styles
    import importlib.util
    spec = importlib.util.spec_from_file_location("evil_agent", Path(__file__).parent / "evil_agent" / "__init__.py")
    evil_agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evil_agent)
    
    LLMClient = evil_agent.LLMClient
    ATTACK_TYPES = evil_agent.ATTACK_TYPES
    generate_attack = evil_agent.generate_attack
    generate_llm_attack = evil_agent.generate_llm_attack
    generate_privilege_escalation_attack = evil_agent.generate_privilege_escalation_attack
    generate_batch = evil_agent.generate_batch
    pretty_print_batch = evil_agent.pretty_print_batch
    test_attack_against_agent = evil_agent.test_attack_against_agent
    get_llm_client = evil_agent.get_llm_client

# Initialize singleton LLM client for backward compatibility
llm_client = get_llm_client()


def main():
    """Backward-compatible main function"""
    try:
        from src.evil_agent.cli import main as cli_main
    except ImportError:
        import importlib.util
        spec = importlib.util.spec_from_file_location("cli", Path(__file__).parent / "evil_agent" / "cli.py")
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        cli_main = cli.main
    
    cli_main()


if __name__ == "__main__":
    main()
