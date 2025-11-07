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
        ATTACK_TYPES,
        LLMClient,
        generate_attack,
        generate_batch,
        generate_llm_attack,
        generate_privilege_escalation_attack,
        get_llm_client,
        pretty_print_batch,
        test_attack_against_agent,
    )
except ImportError:
    # Fallback for different import styles
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "evil_agent", Path(__file__).parent / "evil_agent" / "__init__.py"
    )
    if spec is not None and spec.loader is not None:
        evil_agent = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evil_agent)

        LLMClient = evil_agent.LLMClient  # type: ignore[assignment]
        ATTACK_TYPES = evil_agent.ATTACK_TYPES  # type: ignore[assignment]
        generate_attack = evil_agent.generate_attack  # type: ignore[assignment]
        generate_llm_attack = evil_agent.generate_llm_attack  # type: ignore[assignment]
        generate_privilege_escalation_attack = evil_agent.generate_privilege_escalation_attack  # type: ignore[assignment]
        generate_batch = evil_agent.generate_batch  # type: ignore[assignment]
        pretty_print_batch = evil_agent.pretty_print_batch  # type: ignore[assignment]
        test_attack_against_agent = evil_agent.test_attack_against_agent  # type: ignore[assignment]
        get_llm_client = evil_agent.get_llm_client  # type: ignore[assignment]
    else:
        raise ImportError("Could not load evil_agent module") from None

# Initialize singleton LLM client for backward compatibility
llm_client = get_llm_client()


def main():
    """Backward-compatible main function"""
    try:
        from src.evil_agent.cli import main as cli_main
    except ImportError:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "cli", Path(__file__).parent / "evil_agent" / "cli.py"
        )
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        cli_main = cli.main

    cli_main()


if __name__ == "__main__":
    main()
