"""
CLI Interface for Evil Agent
Command-line interface for generating adversarial attacks
"""

import json
import argparse

from .attack_generator import generate_attack
from .privilege_escalation import generate_privilege_escalation_attack
from .batch_generator import generate_batch
from .formatter import pretty_print_batch
from .attack_tester import test_attack_against_agent


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Malicious Prompt Generator (Adversarial Attack Agent)"
    )
    parser.add_argument("--user-id", default="dave", help="User ID")
    parser.add_argument("--tenant-id", default="tenantA", help="Tenant ID")
    parser.add_argument("--clearance", default="INTERNAL", help="Clearance level")
    parser.add_argument("--role", default="analyst", help="User role")
    parser.add_argument("--attack-type", default=None, help="Specific attack type to generate")
    parser.add_argument("--batch", type=int, default=0, help="Generate multiple examples per attack type")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed")
    parser.add_argument("--use-llm", action="store_true", help="Use LLM APIs to generate sophisticated attacks")
    parser.add_argument("--llm-provider", default="google", choices=["google", "openai", "flash"], help="LLM provider to use")
    parser.add_argument("--test-agent", action="store_true", help="Test generated attacks against Sec_Agent.py")
    parser.add_argument("--privilege-escalation", action="store_true", help="Generate privilege escalation attacks with mismatched roles")

    args = parser.parse_args()

    if args.privilege_escalation:
        # Generate privilege escalation attack
        attack = generate_privilege_escalation_attack(
            args.user_id, args.tenant_id, args.attack_type, args.seed, 
            use_llm=args.use_llm, llm_provider=args.llm_provider
        )
        print(json.dumps(attack, indent=2))
        print("\n  These privilege escalation prompts are for authorized red-team testing only.")
    elif args.batch > 0:
        batch = generate_batch(
            args.user_id, args.tenant_id, args.clearance, 
            count_per_type=args.batch, use_llm=args.use_llm, llm_provider=args.llm_provider, role=args.role
        )
        pretty_print_batch(batch)
    else:
        attack = generate_attack(
            args.user_id, args.tenant_id, args.clearance, 
            args.attack_type, args.seed, use_llm=args.use_llm, llm_provider=args.llm_provider, role=args.role
        )
        print(json.dumps(attack, indent=2))
        print("\n  These prompts are for authorized red-team testing only.")
    
    # Test against Sec_Agent.py if requested
    if args.test_agent:
        print("\n🧪 Testing generated attack against Sec_Agent.py...")
        attack_data = attack if 'attack' in locals() else batch[0] if args.batch > 0 else None
        if attack_data:
            test_attack_against_agent(attack_data["example"]["query"], args.tenant_id, args.clearance)


if __name__ == "__main__":
    main()

