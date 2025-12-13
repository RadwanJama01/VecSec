# 🔐 VecSec - Security Testing Framework

**Automated Security Testing with Pattern-Based Detection & RLS Enforcement**

VecSec is a security testing framework for RLS-protected vector databases with pattern-based threat detection, role-based access control, and comprehensive attack generation.

## ✨ Features

- 🛡️ **7+ Attack Types**: Prompt injection, data exfiltration, privilege escalation, and more
- 🔍 **Pattern-Based Detection**: Keyword and rule-based threat detection
- 🔐 **RLS Enforcement**: Multi-tenant isolation, role-based access, clearance levels
- 🤖 **Automated Testing**: Blind tests, batch testing, comprehensive reporting
- 📊 **Real-Time Metrics**: Prometheus + Grafana integration (when configured)
- ⚡ **Vector Storage**: ChromaDB Cloud/Local with InMemory fallback

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start everything (Docker + metrics)
./scripts/START_EVERYTHING.sh

# Run demo
python3 src/SHOWCASE_DEMO.py
```

## 🎯 Usage

### Generate & Test Attacks
```bash
# Generate attacks
python3 src/Evil_Agent.py --attack-type prompt_injection

# Test security
python3 src/Sec_Agent.py "query" --role analyst --clearance INTERNAL

# Run tests
python3 src/Good_Vs_Evil.py --test-type blind --blind-tests 50
```

### Training & Metrics
```bash
# Training iterations
python3 src/train_security_agent.py --iterations 5

# View metrics
python3 src/SIMPLE_METRICS_VIEWER.py
cat data/training/vecsec_metrics.json
```

## 🔐 Security Features

### Clearance Levels
```
PUBLIC (1) < INTERNAL (2) < CONFIDENTIAL (3) < SECRET (4)
```

### Roles
- **guest**: PUBLIC clearance
- **analyst**: INTERNAL clearance
- **superuser**: CONFIDENTIAL clearance
- **admin**: SECRET clearance

### Attack Types
1. Prompt Injection
2. Data Exfiltration
3. Social Engineering
4. Obfuscation
5. Jailbreak
6. Privilege Escalation
7. Data Poisoning

## ⚙️ Configuration

### Environment Variables (.env)
```env
# ChromaDB (optional)
USE_CHROMA=true
CHROMA_API_KEY=your_key          # For cloud
CHROMA_TENANT=your_tenant
CHROMA_DATABASE=your_database
CHROMA_PATH=./chroma_db          # For local

# Optional: Baseten for semantic detection
BASETEN_MODEL_ID=your_model
BASETEN_API_KEY=your_key

# Metrics (optional)
METRICS_PORT=8080
```

### Vector Storage Options
1. **InMemory** (default): Fast, no persistence
2. **ChromaDB Local**: Persistent storage in `./chroma_db/`
3. **ChromaDB Cloud**: Cloud-hosted with API key

## 🧪 Testing

```bash
# Run full CI pipeline locally
./scripts/run_ci_locally.sh

# Individual tests
ruff check src/ scripts/
pytest src/sec_agent/tests/ -v

# Integration tests
export USE_CHROMA=true
pytest src/sec_agent/tests/test_rag_orchestrator.py -v
```

## 📊 Architecture

```
User Query → Query Parser → Vector Store → RLS Enforcer → Response
                ↓                              ↓
         Threat Detector              Policy Manager
```

## 📁 Key Files

- `src/Sec_Agent.py` - Security enforcement
- `src/Evil_Agent.py` - Attack generator
- `src/Good_Vs_Evil.py` - Test framework
- `src/sec_agent/` - Core security modules

## 📚 Documentation

- `QUICK_START.md` - Quick reference
- `TASKS.md` - Development tasks
- `PROJECT_STRUCTURE.md` - File organization
- `docs/` - Detailed documentation

## ⚠️ Disclaimer

**For authorized security testing only.** Unauthorized use against systems you don't own is illegal and unethical.

---

**🚀 Start demo: `python3 src/SHOWCASE_DEMO.py`**
