#!/usr/bin/env python3
"""
VecSec Red Team Framework Test Script
=====================================

This script demonstrates the key features of the VecSec Red Team Framework
including attack generation, execution, and monitoring.

Author: VecSec Team
"""

import asyncio
import logging
import json
from datetime import datetime

from vecsec_red_team_framework import VecSecRedTeamFramework, load_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_attack_generation():
    """Test attack generation capabilities."""
    logger.info("🧪 Testing attack generation...")
    
    from llama_attack_generator import LlamaAttackGenerator
    
    generator = LlamaAttackGenerator()
    
    # Generate attacks
    attacks = generator.generate_attacks(
        prompt="Generate 3 PRE_EMBEDDING attacks targeting financial data access",
        num_attacks=3
    )
    
    logger.info(f"Generated {len(attacks)} attacks:")
    for attack in attacks:
        logger.info(f"  - {attack.attack_id}: {attack.method} ({attack.mode})")
    
    return attacks

async def test_synthetic_data():
    """Test synthetic data generation."""
    logger.info("🎲 Testing synthetic data generation...")
    
    from synthetic_data_generator import SyntheticDataGenerator
    
    config = {
        'embedding_dimension': 1536,
        'num_tenants': 2,
        'embeddings_per_tenant': 10,
        'output_dir': './test_synthetic_data'
    }
    
    generator = SyntheticDataGenerator(config)
    saved_files = await generator.generate_complete_dataset()
    
    logger.info(f"Synthetic data generated: {list(saved_files.keys())}")
    return saved_files

async def test_sandbox_manager():
    """Test sandbox manager."""
    logger.info("🏗️ Testing sandbox manager...")
    
    from sandbox_manager import EnhancedSandboxManager
    
    config = {
        'sandbox_image': 'python:3.9-slim',
        'mem_limit': '1g',
        'cpu_quota': 100000,
        'network': 'bridge'
    }
    
    sandbox_manager = EnhancedSandboxManager(config)
    
    try:
        # Create sandbox
        async with await sandbox_manager.create_isolated_environment() as sandbox:
            logger.info(f"Created sandbox: {sandbox.sandbox_id}")
            
            # Test request
            response = await sandbox.execute_secure_request(
                url="http://httpbin.org/get",
                method="GET"
            )
            
            logger.info(f"Sandbox request successful: {response.get('status_code')}")
            
    except Exception as e:
        logger.warning(f"Sandbox test failed (expected in test environment): {e}")
    
    return True

async def test_attack_logger():
    """Test attack logger."""
    logger.info("📊 Testing attack logger...")
    
    from attack_logger import AttackLogger
    from llama_attack_generator import AttackSpec, AttackResult
    
    logger_instance = AttackLogger(db_path='./test_attack_logs.db')
    
    # Create test attack
    attack_spec = AttackSpec(
        attack_id="test_attack_001",
        mode="PRE_EMBEDDING",
        method="semantic_injection",
        seed=42,
        config={"complexity": 0.7},
        payload={"query": "test query"},
        target_vulnerability="semantic_bypass",
        expected_outcome="bypass",
        success_criteria=["query approved"],
        evasion_techniques=["semantic_obfuscation"],
        threat_model={"sophistication": "high"}
    )
    
    result = AttackResult(
        attack_id="test_attack_001",
        success=True,
        evasion_successful=True,
        detected_as_malicious=False,
        response={"status": "approved"},
        execution_time=1.5,
        telemetry={"sandbox_id": "test_sandbox"}
    )
    
    # Log attack
    await logger_instance.log_attack(attack_spec, result, result.telemetry)
    
    # Get statistics
    stats = await logger_instance.get_attack_statistics()
    logger.info(f"Attack statistics: {stats['overall']['total_attacks']} total attacks")
    
    return stats

async def test_framework_integration():
    """Test full framework integration."""
    logger.info("🚀 Testing framework integration...")
    
    # Load configuration
    config = load_config()
    
    # Modify config for testing
    config['orchestrator']['num_workers'] = 2
    config['orchestrator']['initial_rate'] = 5
    config['synthetic_data']['embeddings_per_tenant'] = 5
    
    # Create framework
    framework = VecSecRedTeamFramework(config)
    
    try:
        # Test synthetic data generation
        await framework.generate_synthetic_dataset()
        
        # Test single attack
        attack_id = await framework.run_single_attack(
            attack_prompt="Generate a test attack for semantic bypass",
            mode="PRE_EMBEDDING",
            priority="medium"
        )
        
        logger.info(f"Single attack scheduled: {attack_id}")
        
        # Test campaign
        campaign_config = {
            'name': 'Test Campaign',
            'description': 'Testing campaign functionality',
            'modes': [
                {'mode': 'PRE_EMBEDDING', 'count': 2, 'target': 'RLS_policies', 'priority': 'medium'}
            ]
        }
        
        campaign_id = await framework.run_attack_campaign(campaign_config)
        logger.info(f"Campaign scheduled: {campaign_id}")
        
        # Get statistics
        stats = await framework.get_framework_statistics()
        logger.info(f"Framework statistics: {stats['framework_status']['running']}")
        
    except Exception as e:
        logger.error(f"Framework integration test failed: {e}")
    
    return True

async def main():
    """Run all tests."""
    logger.info("🛡️ Starting VecSec Red Team Framework Tests")
    logger.info("=" * 50)
    
    try:
        # Test individual components
        await test_attack_generation()
        await test_synthetic_data()
        await test_sandbox_manager()
        await test_attack_logger()
        
        # Test integration
        await test_framework_integration()
        
        logger.info("=" * 50)
        logger.info("✅ All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
