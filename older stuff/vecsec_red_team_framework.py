"""
VecSec Red Team Framework Integration
=====================================

This module integrates all components of the VecSec Red Team Framework
into a unified system for continuous adversarial testing against the
VecSec VectorSecurityAgent.

Author: VecSec Team
"""

import asyncio
import logging
import json
import os
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

# Import all framework components
from llama_attack_generator import LlamaAttackGenerator, AttackSpec
from adversarial_test_orchestrator import AdversarialTestOrchestrator, CampaignConfig
from sandbox_manager import EnhancedSandboxManager
from attack_logger import AttackLogger
from synthetic_data_generator import SyntheticDataGenerator
from analytics_dashboard import AnalyticsDashboard

logger = logging.getLogger(__name__)

class VecSecRedTeamFramework:
    """
    Main integration class for the VecSec Red Team Framework.
    
    This class orchestrates all components to provide:
    - Continuous adversarial testing
    - Vulnerability discovery
    - Performance monitoring
    - Analytics and reporting
    - Safe testing environment
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.running = False
        
        # Initialize components
        self._initialize_components()
        
        logger.info("VecSec Red Team Framework initialized")
    
    def _initialize_components(self):
        """Initialize all framework components."""
        try:
            # Initialize attack generator
            self.attack_generator = LlamaAttackGenerator(
                model_path=self.config.get('llama_model_path', 'meta-llama/Llama-3.1-8B-Instruct'),
                device=self.config.get('device', 'auto'),
                max_length=self.config.get('max_length', 4096)
            )
            
            # Initialize sandbox manager
            sandbox_config = self.config.get('sandbox', {})
            self.sandbox_manager = EnhancedSandboxManager(sandbox_config)
            
            # Initialize attack logger
            logger_config = self.config.get('logger', {})
            self.attack_logger = AttackLogger(
                db_path=logger_config.get('db_path', './attack_logs.db'),
                log_dir=logger_config.get('log_dir', './logs'),
                config=logger_config
            )
            
            # Initialize orchestrator
            orchestrator_config = self.config.get('orchestrator', {})
            self.orchestrator = AdversarialTestOrchestrator(
                attack_generator=self.attack_generator,
                target_endpoint=self.config.get('target_endpoint', 'http://localhost:8080'),
                sandbox_manager=self.sandbox_manager,
                attack_logger=self.attack_logger,
                config=orchestrator_config
            )
            
            # Initialize synthetic data generator
            synthetic_config = self.config.get('synthetic_data', {})
            self.synthetic_generator = SyntheticDataGenerator(synthetic_config)
            
            # Initialize analytics dashboard
            dashboard_config = self.config.get('dashboard', {})
            self.dashboard = AnalyticsDashboard(
                attack_logger=self.attack_logger,
                config=dashboard_config
            )
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    async def start_continuous_testing(self):
        """Start continuous adversarial testing."""
        if self.running:
            logger.warning("Continuous testing already running")
            return
        
        self.running = True
        logger.info("Starting continuous adversarial testing")
        
        try:
            # Start orchestrator
            orchestrator_task = asyncio.create_task(self.orchestrator.start())
            
            # Start analytics dashboard
            dashboard_task = asyncio.create_task(self._run_dashboard())
            
            # Start synthetic data generation
            synthetic_task = asyncio.create_task(self._run_synthetic_data_generation())
            
            # Wait for tasks
            await asyncio.gather(
                orchestrator_task,
                dashboard_task,
                synthetic_task,
                return_exceptions=True
            )
            
        except Exception as e:
            logger.error(f"Continuous testing failed: {e}")
        finally:
            self.running = False
    
    async def stop_continuous_testing(self):
        """Stop continuous adversarial testing."""
        if not self.running:
            return
        
        logger.info("Stopping continuous adversarial testing")
        self.running = False
        
        # Stop orchestrator
        await self.orchestrator.stop()
        
        # Cleanup sandboxes
        await self.sandbox_manager.cleanup_all_sandboxes()
        
        logger.info("Continuous testing stopped")
    
    async def run_attack_campaign(self, campaign_config: Dict) -> str:
        """
        Run a specific attack campaign.
        
        Args:
            campaign_config: Campaign configuration
            
        Returns:
            Campaign ID
        """
        try:
            # Create campaign config
            campaign = CampaignConfig(
                campaign_id=campaign_config.get('campaign_id', f"campaign_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                name=campaign_config.get('name', 'Custom Campaign'),
                description=campaign_config.get('description', ''),
                modes=campaign_config.get('modes', []),
                start_time=campaign_config.get('start_time'),
                end_time=campaign_config.get('end_time'),
                max_attacks=campaign_config.get('max_attacks'),
                rate_limit=campaign_config.get('rate_limit')
            )
            
            # Schedule campaign
            campaign_id = await self.orchestrator.schedule_attack_campaign(
                campaign, 
                start_immediately=True
            )
            
            logger.info(f"Campaign {campaign_id} scheduled and started")
            return campaign_id
            
        except Exception as e:
            logger.error(f"Failed to run campaign: {e}")
            raise
    
    async def run_single_attack(self, 
                               attack_prompt: str,
                               mode: str = "PRE_EMBEDDING",
                               priority: str = "medium") -> str:
        """
        Run a single attack based on prompt.
        
        Args:
            attack_prompt: Prompt for attack generation
            mode: Attack mode (PRE_EMBEDDING, POST_EMBEDDING, HYBRID)
            priority: Attack priority
            
        Returns:
            Attack ID
        """
        try:
            # Generate attack
            attacks = self.attack_generator.generate_attacks(
                prompt=attack_prompt,
                num_attacks=1
            )
            
            if not attacks:
                raise ValueError("No attacks generated")
            
            attack = attacks[0]
            attack.mode = mode
            attack.priority = priority
            
            # Schedule attack
            attack_id = await self.orchestrator.schedule_single_attack(
                attack, priority
            )
            
            logger.info(f"Single attack {attack_id} scheduled")
            return attack_id
            
        except Exception as e:
            logger.error(f"Failed to run single attack: {e}")
            raise
    
    async def generate_synthetic_dataset(self) -> Dict[str, str]:
        """Generate synthetic dataset for testing."""
        try:
            logger.info("Generating synthetic dataset")
            saved_files = await self.synthetic_generator.generate_complete_dataset()
            
            logger.info(f"Synthetic dataset generated: {saved_files}")
            return saved_files
            
        except Exception as e:
            logger.error(f"Failed to generate synthetic dataset: {e}")
            raise
    
    async def get_framework_statistics(self) -> Dict:
        """Get comprehensive framework statistics."""
        try:
            # Get orchestrator stats
            orchestrator_stats = self.orchestrator.get_statistics()
            
            # Get attack logger stats
            attack_stats = await self.attack_logger.get_attack_statistics()
            
            # Get sandbox stats
            sandbox_stats = await self.sandbox_manager.get_sandbox_stats()
            
            # Get vulnerabilities
            vulnerabilities = await self.attack_logger.get_top_vulnerabilities(10)
            
            return {
                'orchestrator': orchestrator_stats,
                'attacks': attack_stats,
                'sandboxes': sandbox_stats,
                'vulnerabilities': vulnerabilities,
                'framework_status': {
                    'running': self.running,
                    'components_initialized': True,
                    'last_updated': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get framework statistics: {e}")
            return {'error': str(e)}
    
    async def _run_dashboard(self):
        """Run analytics dashboard."""
        try:
            logger.info("Starting analytics dashboard")
            # Note: In a real implementation, this would run the Flask app
            # For now, we'll just log that it's running
            while self.running:
                await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Dashboard error: {e}")
    
    async def _run_synthetic_data_generation(self):
        """Run periodic synthetic data generation."""
        try:
            while self.running:
                # Generate synthetic data every hour
                await asyncio.sleep(3600)
                
                if self.running:
                    logger.info("Generating periodic synthetic data")
                    await self.generate_synthetic_dataset()
                    
        except Exception as e:
            logger.error(f"Synthetic data generation error: {e}")
    
    def create_default_campaigns(self) -> List[Dict]:
        """Create default attack campaigns for testing."""
        campaigns = [
            {
                'name': 'RLS Policy Testing',
                'description': 'Test Row-Level Security policy enforcement',
                'modes': [
                    {'mode': 'PRE_EMBEDDING', 'count': 20, 'target': 'RLS_policies', 'priority': 'high'},
                    {'mode': 'POST_EMBEDDING', 'count': 10, 'target': 'embedding_security', 'priority': 'medium'}
                ]
            },
            {
                'name': 'Topic Boundary Evasion',
                'description': 'Test topic-based access control boundaries',
                'modes': [
                    {'mode': 'PRE_EMBEDDING', 'count': 15, 'target': 'topic_boundaries', 'priority': 'high'},
                    {'mode': 'HYBRID', 'count': 5, 'target': 'multi_layer_defense', 'priority': 'medium'}
                ]
            },
            {
                'name': 'Embedding Security Assessment',
                'description': 'Comprehensive embedding security testing',
                'modes': [
                    {'mode': 'POST_EMBEDDING', 'count': 25, 'target': 'embedding_security', 'priority': 'high'},
                    {'mode': 'HYBRID', 'count': 10, 'target': 'vector_manipulation', 'priority': 'high'}
                ]
            },
            {
                'name': 'Malware Detection Bypass',
                'description': 'Test malware detection system robustness',
                'modes': [
                    {'mode': 'PRE_EMBEDDING', 'count': 30, 'target': 'malware_detection', 'priority': 'critical'},
                    {'mode': 'HYBRID', 'count': 15, 'target': 'multi_stage_evasion', 'priority': 'critical'}
                ]
            }
        ]
        
        return campaigns
    
    async def run_default_campaigns(self):
        """Run all default campaigns."""
        campaigns = self.create_default_campaigns()
        
        for campaign_config in campaigns:
            try:
                campaign_id = await self.run_attack_campaign(campaign_config)
                logger.info(f"Started default campaign: {campaign_config['name']} ({campaign_id})")
            except Exception as e:
                logger.error(f"Failed to start campaign {campaign_config['name']}: {e}")

# Configuration loader
def load_config(config_path: str = './red_team_config.json') -> Dict:
    """Load configuration from file."""
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            # Return default configuration
            return get_default_config()
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return get_default_config()

def get_default_config() -> Dict:
    """Get default configuration."""
    return {
        'llama_model_path': 'meta-llama/Llama-3.1-8B-Instruct',
        'device': 'auto',
        'max_length': 4096,
        'target_endpoint': 'http://localhost:8080',
        'sandbox': {
            'sandbox_image': 'vecsec-attack-sandbox:latest',
            'mem_limit': '4g',
            'cpu_quota': 200000,
            'pids_limit': 100,
            'network': 'sandbox-network',
            'max_execution_time': 300,
            'log_dir': './logs',
            'save_sandbox_logs': True,
            'max_sandbox_age_hours': 2,
            'cleanup_interval': 300
        },
        'logger': {
            'db_path': './attack_logs.db',
            'log_dir': './logs',
            'vulnerability_threshold': 3
        },
        'orchestrator': {
            'initial_rate': 10,
            'max_rate': 50,
            'min_rate': 1,
            'num_workers': 5
        },
        'synthetic_data': {
            'embedding_dimension': 1536,
            'num_tenants': 5,
            'embeddings_per_tenant': 1000,
            'output_dir': './synthetic_data'
        },
        'dashboard': {
            'refresh_interval': 30,
            'alert_thresholds': {
                'critical_vulnerabilities': 1,
                'high_evasion_rate': 0.3,
                'low_detection_rate': 0.5
            }
        }
    }

# Main execution function
async def main():
    """Main execution function."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    config = load_config()
    
    # Create framework
    framework = VecSecRedTeamFramework(config)
    
    try:
        # Generate synthetic data
        logger.info("Generating synthetic dataset...")
        await framework.generate_synthetic_dataset()
        
        # Run default campaigns
        logger.info("Running default campaigns...")
        await framework.run_default_campaigns()
        
        # Start continuous testing
        logger.info("Starting continuous testing...")
        await framework.start_continuous_testing()
        
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Framework error: {e}")
    finally:
        # Cleanup
        await framework.stop_continuous_testing()
        logger.info("Framework shutdown complete")

# Example usage
if __name__ == "__main__":
    # Run the framework
    asyncio.run(main())
