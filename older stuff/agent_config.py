"""
VecSec AI Agent Configuration
==============================

This file contains all configuration settings for the AI agent.
Sensitive values should be set via environment variables.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any

class AgentConfig:
    """Configuration class for the VecSec AI Agent"""
    
    def __init__(self, env_file: Optional[str] = None):
        """Initialize configuration with optional .env file"""
        self.env_file = env_file or ".env"
        self._load_env_file()
        self._setup_config()
    
    def _load_env_file(self):
        """Load environment variables from .env file if it exists"""
        env_path = Path(self.env_file)
        if env_path.exists():
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
    
    def _setup_config(self):
        """Setup configuration from environment variables"""
        
        # Agent Configuration
        self.agent_name = os.getenv('AGENT_NAME', 'VecSecAgent')
        self.agent_version = os.getenv('AGENT_VERSION', '1.0.0')
        self.agent_mode = os.getenv('AGENT_MODE', 'production')
        
        # Proxy Configuration
        self.proxy_url = os.getenv('PROXY_URL', 'http://localhost:8080')
        self.proxy_timeout = int(os.getenv('PROXY_TIMEOUT', '30'))
        self.proxy_max_retries = int(os.getenv('PROXY_MAX_RETRIES', '3'))
        
        # Malware Detection
        self.malware_model_path = os.getenv('MALWARE_MODEL_PATH', 'malware_bert_model.pth')
        self.malware_detection_enabled = os.getenv('MALWARE_DETECTION_ENABLED', 'true').lower() == 'true'
        self.malware_ml_enabled = os.getenv('MALWARE_ML_ENABLED', 'true').lower() == 'true'
        
        # Agent Memory and Learning
        self.agent_memory_file = os.getenv('AGENT_MEMORY_FILE', 'agent_memory.json')
        self.agent_max_memory_size = int(os.getenv('AGENT_MAX_MEMORY_SIZE', '10000'))
        self.agent_learning_enabled = os.getenv('AGENT_LEARNING_ENABLED', 'true').lower() == 'true'
        
        # Security Thresholds
        self.auto_block_threshold = float(os.getenv('AUTO_BLOCK_THRESHOLD', '0.8'))
        self.escalation_threshold = float(os.getenv('ESCALATION_THRESHOLD', '0.9'))
        self.investigation_threshold = float(os.getenv('INVESTIGATION_THRESHOLD', '0.3'))
        
        # Logging Configuration
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        self.log_file = os.getenv('LOG_FILE', 'agent.log')
        self.log_max_size = os.getenv('LOG_MAX_SIZE', '10MB')
        self.log_backup_count = int(os.getenv('LOG_BACKUP_COUNT', '5'))
        
        # API Keys and Secrets
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        self.azure_openai_api_key = os.getenv('AZURE_OPENAI_API_KEY')
        self.azure_openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        
        # Database Configuration
        self.database_url = os.getenv('DATABASE_URL', 'sqlite:///agent.db')
        self.redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        
        # External Services
        self.siem_endpoint = os.getenv('SIEM_ENDPOINT')
        self.soar_endpoint = os.getenv('SOAR_ENDPOINT')
        self.threat_intel_api = os.getenv('THREAT_INTEL_API')
        
        # Security Settings
        self.encryption_key = os.getenv('ENCRYPTION_KEY', 'default_encryption_key_change_me')
        self.jwt_secret = os.getenv('JWT_SECRET', 'default_jwt_secret_change_me')
        self.session_secret = os.getenv('SESSION_SECRET', 'default_session_secret_change_me')
        
        # Performance Settings
        self.max_concurrent_analyses = int(os.getenv('MAX_CONCURRENT_ANALYSES', '10'))
        self.analysis_timeout = int(os.getenv('ANALYSIS_TIMEOUT', '60'))
        self.cache_ttl = int(os.getenv('CACHE_TTL', '3600'))
        
        # Development Settings
        self.debug = os.getenv('DEBUG', 'false').lower() == 'true'
        self.testing = os.getenv('TESTING', 'false').lower() == 'true'
        self.mock_external_services = os.getenv('MOCK_EXTERNAL_SERVICES', 'false').lower() == 'true'
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary (excluding secrets)"""
        config = {
            'agent_name': self.agent_name,
            'agent_version': self.agent_version,
            'agent_mode': self.agent_mode,
            'proxy_url': self.proxy_url,
            'proxy_timeout': self.proxy_timeout,
            'proxy_max_retries': self.proxy_max_retries,
            'malware_model_path': self.malware_model_path,
            'malware_detection_enabled': self.malware_detection_enabled,
            'malware_ml_enabled': self.malware_ml_enabled,
            'agent_memory_file': self.agent_memory_file,
            'agent_max_memory_size': self.agent_max_memory_size,
            'agent_learning_enabled': self.agent_learning_enabled,
            'auto_block_threshold': self.auto_block_threshold,
            'escalation_threshold': self.escalation_threshold,
            'investigation_threshold': self.investigation_threshold,
            'log_level': self.log_level,
            'log_file': self.log_file,
            'log_max_size': self.log_max_size,
            'log_backup_count': self.log_backup_count,
            'database_url': self.database_url,
            'redis_url': self.redis_url,
            'max_concurrent_analyses': self.max_concurrent_analyses,
            'analysis_timeout': self.analysis_timeout,
            'cache_ttl': self.cache_ttl,
            'debug': self.debug,
            'testing': self.testing,
            'mock_external_services': self.mock_external_services
        }
        
        # Add external services if configured
        if self.siem_endpoint:
            config['siem_endpoint'] = self.siem_endpoint
        if self.soar_endpoint:
            config['soar_endpoint'] = self.soar_endpoint
        if self.threat_intel_api:
            config['threat_intel_api'] = self.threat_intel_api
        
        return config
    
    def get_secrets_dict(self) -> Dict[str, Any]:
        """Get secrets configuration (for debugging only)"""
        return {
            'openai_api_key': '***' if self.openai_api_key else None,
            'anthropic_api_key': '***' if self.anthropic_api_key else None,
            'azure_openai_api_key': '***' if self.azure_openai_api_key else None,
            'azure_openai_endpoint': '***' if self.azure_openai_endpoint else None,
            'encryption_key': '***' if self.encryption_key else None,
            'jwt_secret': '***' if self.jwt_secret else None,
            'session_secret': '***' if self.session_secret else None
        }
    
    def validate_config(self) -> bool:
        """Validate configuration settings"""
        errors = []
        
        # Validate thresholds
        if not 0 <= self.auto_block_threshold <= 1:
            errors.append("AUTO_BLOCK_THRESHOLD must be between 0 and 1")
        
        if not 0 <= self.escalation_threshold <= 1:
            errors.append("ESCALATION_THRESHOLD must be between 0 and 1")
        
        if not 0 <= self.investigation_threshold <= 1:
            errors.append("INVESTIGATION_THRESHOLD must be between 0 and 1")
        
        if self.auto_block_threshold >= self.escalation_threshold:
            errors.append("AUTO_BLOCK_THRESHOLD must be less than ESCALATION_THRESHOLD")
        
        # Validate URLs
        if not self.proxy_url.startswith(('http://', 'https://')):
            errors.append("PROXY_URL must start with http:// or https://")
        
        # Validate numeric values
        if self.proxy_timeout <= 0:
            errors.append("PROXY_TIMEOUT must be positive")
        
        if self.agent_max_memory_size <= 0:
            errors.append("AGENT_MAX_MEMORY_SIZE must be positive")
        
        if self.max_concurrent_analyses <= 0:
            errors.append("MAX_CONCURRENT_ANALYSES must be positive")
        
        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True

# Global configuration instance
config = AgentConfig()

# Example usage:
if __name__ == "__main__":
    print("VecSec AI Agent Configuration")
    print("=" * 40)
    
    # Show configuration
    config_dict = config.get_config_dict()
    for key, value in config_dict.items():
        print(f"{key}: {value}")
    
    # Show secrets status
    print("\nSecrets Configuration:")
    secrets = config.get_secrets_dict()
    for key, value in secrets.items():
        print(f"{key}: {value}")
    
    # Validate configuration
    print(f"\nConfiguration Valid: {config.validate_config()}")
