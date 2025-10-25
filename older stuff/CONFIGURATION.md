# VecSec AI Agent Configuration Guide

## Overview

The VecSec AI Agent uses a comprehensive configuration system that separates sensitive data from code. All configuration is managed through environment variables and a centralized configuration class.

## 🔧 Configuration Files

### 1. `agent_config.py`
- **Purpose**: Centralized configuration management
- **Features**: Environment variable loading, validation, secrets management
- **Usage**: Imported by the AI agent and other components

### 2. `env_template.txt`
- **Purpose**: Template for environment variables
- **Usage**: Copy to `.env` and customize values
- **Security**: Contains example values, not real secrets

## 🚀 Quick Setup

### 1. Create Environment File
```bash
# Copy the template
cp env_template.txt .env

# Edit the configuration
nano .env
```

### 2. Configure Basic Settings
```bash
# Agent Configuration
AGENT_NAME=VecSecAgent
AGENT_VERSION=1.0.0
AGENT_MODE=production

# Proxy Configuration
PROXY_URL=http://localhost:8080
PROXY_TIMEOUT=30

# Security Thresholds
AUTO_BLOCK_THRESHOLD=0.8
ESCALATION_THRESHOLD=0.9
```

### 3. Set Security Secrets
```bash
# Generate secure keys
ENCRYPTION_KEY=$(openssl rand -hex 32)
JWT_SECRET=$(openssl rand -hex 32)
SESSION_SECRET=$(openssl rand -hex 32)
```

## 📋 Configuration Categories

### Agent Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_NAME` | `VecSecAgent` | Name of the AI agent |
| `AGENT_VERSION` | `1.0.0` | Agent version |
| `AGENT_MODE` | `production` | Agent mode (production/development) |

### Proxy Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `PROXY_URL` | `http://localhost:8080` | VecSec proxy URL |
| `PROXY_TIMEOUT` | `30` | Request timeout in seconds |
| `PROXY_MAX_RETRIES` | `3` | Maximum retry attempts |

### Malware Detection
| Variable | Default | Description |
|----------|---------|-------------|
| `MALWARE_MODEL_PATH` | `malware_bert_model.pth` | Path to BERT model |
| `MALWARE_DETECTION_ENABLED` | `true` | Enable malware detection |
| `MALWARE_ML_ENABLED` | `true` | Enable ML-based detection |

### Security Thresholds
| Variable | Default | Description |
|----------|---------|-------------|
| `AUTO_BLOCK_THRESHOLD` | `0.8` | Auto-block confidence threshold |
| `ESCALATION_THRESHOLD` | `0.9` | Escalation confidence threshold |
| `INVESTIGATION_THRESHOLD` | `0.3` | Investigation confidence threshold |

### Agent Memory
| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_MEMORY_FILE` | `agent_memory.json` | Memory storage file |
| `AGENT_MAX_MEMORY_SIZE` | `10000` | Maximum memory entries |
| `AGENT_LEARNING_ENABLED` | `true` | Enable learning capabilities |

### Logging Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging level |
| `LOG_FILE` | `agent.log` | Log file path |
| `LOG_MAX_SIZE` | `10MB` | Maximum log file size |
| `LOG_BACKUP_COUNT` | `5` | Number of backup files |

## 🔐 Security Configuration

### API Keys and Secrets
```bash
# OpenAI Integration (optional)
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic Integration (optional)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Azure OpenAI (optional)
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint_here
```

### Security Settings
```bash
# Generate secure keys
ENCRYPTION_KEY=your_encryption_key_here
JWT_SECRET=your_jwt_secret_here
SESSION_SECRET=your_session_secret_here
```

## 🗄️ Database Configuration

### SQLite (Default)
```bash
DATABASE_URL=sqlite:///agent.db
```

### PostgreSQL
```bash
DATABASE_URL=postgresql://user:password@localhost:5432/vecsec
```

### Redis
```bash
REDIS_URL=redis://localhost:6379
```

## 🌐 External Services

### SIEM Integration
```bash
SIEM_ENDPOINT=https://your-siem.com/api
```

### SOAR Integration
```bash
SOAR_ENDPOINT=https://your-soar.com/api
```

### Threat Intelligence
```bash
THREAT_INTEL_API=https://your-threat-intel.com/api
```

## ⚡ Performance Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_CONCURRENT_ANALYSES` | `10` | Maximum concurrent analyses |
| `ANALYSIS_TIMEOUT` | `60` | Analysis timeout in seconds |
| `CACHE_TTL` | `3600` | Cache time-to-live in seconds |

## 🛠️ Development Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `DEBUG` | `false` | Enable debug mode |
| `TESTING` | `false` | Enable testing mode |
| `MOCK_EXTERNAL_SERVICES` | `false` | Mock external services |

## 📖 Usage Examples

### Basic Configuration
```python
from agent_config import config

# Access configuration
print(f"Agent Name: {config.agent_name}")
print(f"Proxy URL: {config.proxy_url}")
print(f"Auto Block Threshold: {config.auto_block_threshold}")
```

### Validation
```python
from agent_config import config

# Validate configuration
if config.validate_config():
    print("Configuration is valid")
else:
    print("Configuration has errors")
```

### Get Configuration Dictionary
```python
from agent_config import config

# Get all configuration (excluding secrets)
config_dict = config.get_config_dict()
print(config_dict)

# Get secrets status (masked)
secrets = config.get_secrets_dict()
print(secrets)
```

## 🔒 Security Best Practices

### 1. Environment File Security
```bash
# Set proper permissions
chmod 600 .env

# Add to .gitignore
echo ".env" >> .gitignore
```

### 2. Secret Management
```bash
# Generate secure keys
ENCRYPTION_KEY=$(openssl rand -hex 32)
JWT_SECRET=$(openssl rand -hex 32)
SESSION_SECRET=$(openssl rand -hex 32)
```

### 3. Production Deployment
```bash
# Use environment variables in production
export AGENT_MODE=production
export DEBUG=false
export LOG_LEVEL=WARNING
```

## 🚨 Common Issues

### 1. Configuration Not Loading
```bash
# Check if .env file exists
ls -la .env

# Check file permissions
chmod 644 .env
```

### 2. Invalid Thresholds
```bash
# Ensure thresholds are between 0 and 1
AUTO_BLOCK_THRESHOLD=0.8
ESCALATION_THRESHOLD=0.9
INVESTIGATION_THRESHOLD=0.3
```

### 3. Missing Dependencies
```bash
# Install required packages
uv sync
# or
pip install python-dotenv
```

## 📚 Advanced Configuration

### Custom Configuration Class
```python
from agent_config import AgentConfig

# Create custom configuration
custom_config = AgentConfig(env_file="custom.env")

# Validate custom configuration
if custom_config.validate_config():
    print("Custom configuration is valid")
```

### Environment-Specific Configuration
```bash
# Development
AGENT_MODE=development
DEBUG=true
LOG_LEVEL=DEBUG

# Production
AGENT_MODE=production
DEBUG=false
LOG_LEVEL=WARNING
```

## 🔄 Configuration Updates

### Runtime Configuration Changes
```python
# Update configuration at runtime
config.auto_block_threshold = 0.7
config.escalation_threshold = 0.8

# Validate updated configuration
if config.validate_config():
    print("Updated configuration is valid")
```

### Configuration Persistence
```python
# Save configuration to file
config_dict = config.get_config_dict()
with open('config_backup.json', 'w') as f:
    json.dump(config_dict, f, indent=2)
```

## 📞 Support

For configuration issues:

1. Check the validation output
2. Verify environment variables
3. Review the configuration template
4. Check file permissions
5. Validate threshold values

---

*This configuration system provides a secure, flexible way to manage the VecSec AI Agent settings while keeping sensitive data separate from code.*
