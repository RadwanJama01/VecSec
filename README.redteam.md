# 🛡️ VecSec Red Team Framework

A comprehensive adversarial testing framework for continuous security evaluation of VecSec VectorSecurityAgent using Llama-based attack generation and isolated sandbox execution.

## 🚀 Features

### Core Components

- **🤖 Llama Attack Generator**: Fine-tuned Llama models for sophisticated attack generation
- **🎯 Adversarial Test Orchestrator**: Multi-worker attack execution with adaptive rate limiting
- **🏗️ Sandbox Manager**: Docker-based isolated environments for safe attack execution
- **📊 Attack Logger**: Comprehensive audit trail with vulnerability detection
- **🎲 Synthetic Data Generator**: Safe testing data without exposing real information
- **📈 Analytics Dashboard**: Real-time visualization and monitoring

### Attack Modes

- **PRE_EMBEDDING**: Query-based attacks targeting RLS policies and semantic boundaries
- **POST_EMBEDDING**: Vector manipulation and embedding space attacks
- **HYBRID**: Multi-stage attack chains with adaptive techniques

### Security Features

- **Isolated Execution**: All attacks run in Docker sandboxes
- **Resource Limits**: CPU, memory, and network constraints
- **Security Profiles**: Seccomp and capability restrictions
- **Audit Trail**: Complete logging of all attack attempts
- **Vulnerability Detection**: Automatic identification of security weaknesses

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Llama Model   │───▶│   Orchestrator   │───▶│   Sandbox       │
│   (Attack Gen)  │    │   (Queue Mgmt)   │    │   (Isolation)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Analytics     │◀───│   Attack Logger  │◀───│   VecSec Agent  │
│   Dashboard     │    │   (Audit Trail)  │    │   (Target)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.9+
- CUDA-capable GPU (recommended for Llama model)
- 8GB+ RAM

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd VecSec
```

2. **Install dependencies**:
```bash
pip install -r requirements.redteam.txt
```

3. **Build Docker images**:
```bash
docker-compose -f docker-compose.redteam.yml build
```

4. **Start the framework**:
```bash
docker-compose -f docker-compose.redteam.yml up -d
```

### Manual Setup

1. **Configure the framework**:
```bash
cp red_team_config.json.example red_team_config.json
# Edit configuration as needed
```

2. **Generate synthetic data**:
```bash
python synthetic_data_generator.py
```

3. **Start continuous testing**:
```bash
python vecsec_red_team_framework.py
```

## 📊 Usage

### Running Attack Campaigns

```python
from vecsec_red_team_framework import VecSecRedTeamFramework

# Initialize framework
framework = VecSecRedTeamFramework(config)

# Run a custom campaign
campaign_config = {
    'name': 'RLS Policy Testing',
    'modes': [
        {'mode': 'PRE_EMBEDDING', 'count': 20, 'target': 'RLS_policies'},
        {'mode': 'POST_EMBEDDING', 'count': 10, 'target': 'embedding_security'}
    ]
}

campaign_id = await framework.run_attack_campaign(campaign_config)
```

### Single Attack Execution

```python
# Run a single attack
attack_id = await framework.run_single_attack(
    attack_prompt="Generate attacks targeting financial data access",
    mode="PRE_EMBEDDING",
    priority="high"
)
```

### Analytics Dashboard

Access the dashboard at `http://localhost:5001` to view:
- Real-time attack statistics
- Vulnerability discoveries
- Performance metrics
- Interactive visualizations

## 🔧 Configuration

### Main Configuration (`red_team_config.json`)

```json
{
  "llama_model_path": "meta-llama/Llama-3.1-8B-Instruct",
  "target_endpoint": "http://localhost:8080",
  "sandbox": {
    "mem_limit": "4g",
    "cpu_quota": 200000,
    "pids_limit": 100
  },
  "orchestrator": {
    "initial_rate": 10,
    "max_rate": 50,
    "num_workers": 5
  }
}
```

### Environment Variables

- `TARGET_ENDPOINT`: VecSec agent endpoint
- `LLAMA_MODEL_PATH`: Path to Llama model
- `DEVICE`: Device for model execution (cuda/cpu/auto)
- `LOG_LEVEL`: Logging level (DEBUG/INFO/WARNING/ERROR)

## 🎯 Attack Types

### PRE_EMBEDDING Attacks

- **Prompt Injection**: Inject malicious instructions into queries
- **Topic Evasion**: Bypass topic-based access controls
- **Authority Escalation**: Attempt privilege escalation
- **Semantic Obfuscation**: Hide malicious intent in queries

### POST_EMBEDDING Attacks

- **Vector Poisoning**: Inject malicious embeddings
- **Metadata Injection**: Manipulate embedding metadata
- **Retrieval Manipulation**: Manipulate search results
- **Boundary Exploitation**: Exploit embedding space boundaries

### HYBRID Attacks

- **Multi-Stage**: Complex attack chains
- **Adaptive**: Attacks that adapt based on responses
- **Context Building**: Progressive exploitation
- **Delayed Payload**: Time-delayed malicious actions

## 📈 Monitoring and Analytics

### Key Metrics

- **Attack Success Rate**: Percentage of successful attacks
- **Evasion Rate**: Percentage of attacks that bypass detection
- **Detection Rate**: Percentage of attacks detected as malicious
- **Vulnerability Count**: Number of discovered vulnerabilities
- **Performance Metrics**: Execution times and resource usage

### Alerts

The system automatically generates alerts for:
- Critical vulnerabilities discovered
- High evasion rates (>30%)
- Low detection rates (<50%)
- System performance issues

### Export Capabilities

- Attack execution data (JSON)
- Vulnerability reports
- Performance metrics
- Audit logs

## 🛡️ Security Considerations

### Sandbox Isolation

- All attacks run in isolated Docker containers
- Network isolation prevents external communication
- Resource limits prevent system overload
- Security profiles restrict dangerous operations

### Data Protection

- No real data is exposed during testing
- Synthetic data generation for safe testing
- Complete audit trail for compliance
- Secure logging and data storage

### Access Control

- Framework runs with minimal privileges
- Network access restricted to target endpoint
- File system access limited to necessary directories
- Process isolation prevents privilege escalation

## 🔍 Troubleshooting

### Common Issues

1. **Docker Permission Errors**:
```bash
sudo usermod -aG docker $USER
# Log out and back in
```

2. **CUDA Out of Memory**:
```bash
# Reduce batch size or use CPU
export DEVICE=cpu
```

3. **Sandbox Creation Fails**:
```bash
# Check Docker daemon status
docker system info
```

4. **Model Loading Issues**:
```bash
# Verify model path and permissions
ls -la /path/to/model
```

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
python vecsec_red_team_framework.py
```

## 📚 API Reference

### VecSecRedTeamFramework

```python
class VecSecRedTeamFramework:
    async def start_continuous_testing()
    async def stop_continuous_testing()
    async def run_attack_campaign(campaign_config: Dict) -> str
    async def run_single_attack(prompt: str, mode: str, priority: str) -> str
    async def generate_synthetic_dataset() -> Dict[str, str]
    async def get_framework_statistics() -> Dict
```

### AttackLogger

```python
class AttackLogger:
    async def log_attack(attack_spec: AttackSpec, result: AttackResult, telemetry: Dict)
    async def get_attack_statistics(start_time: datetime, end_time: datetime) -> Dict
    async def get_top_vulnerabilities(limit: int) -> List[Dict]
    async def export_attack_data(output_path: str, start_time: datetime, end_time: datetime)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Documentation**: [Wiki](link-to-wiki)
- **Issues**: [GitHub Issues](link-to-issues)
- **Discussions**: [GitHub Discussions](link-to-discussions)

## 🙏 Acknowledgments

- Meta AI for the Llama models
- Hugging Face for the Transformers library
- Docker for containerization
- The open-source community for various dependencies

---

**⚠️ Disclaimer**: This framework is designed for authorized security testing only. Use responsibly and in accordance with applicable laws and regulations.
