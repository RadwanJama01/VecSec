# 🔐 VecSec - Security Testing Framework

**Automated Security Testing with Pattern-Based Detection & RLS Enforcement**

VecSec is a security testing framework for RLS-protected vector databases with pattern-based threat detection, role-based access control, and comprehensive attack generation.

## ✨ Features

- 🛡️ **7+ Attack Types**: Prompt injection, data exfiltration, privilege escalation, and more
- 🔍 **Pattern-Based Detection**: Keyword and rule-based threat detection
- 🔐 **RLS Enforcement**: Multi-tenant isolation, role-based access, clearance levels
- 🤖 **Automated Testing**: Blind tests, batch testing, comprehensive reporting
- 📊 **Real-Time Metrics**: Prometheus + Grafana integration (when configured)
- ⚡ **Batch Processing**: Optional Baseten embedding integration for semantic detection
- 📈 **Attack Generation**: Static attack templates for security testing

## 🚀 Quick Start

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Or use the automated script
./scripts/install_dependencies.sh
```

### Start Everything
```bash
# One command to start everything:
./scripts/START_EVERYTHING.sh

# This starts:
# - ✅ Docker (Prometheus + Grafana)
# - ✅ Metrics on localhost:3000
# - ✅ Prometheus on localhost:9090
```

### Run Demo
```bash
# Run security tests
python3 src/Good_Vs_Evil.py --test-type blind --blind-tests 20

# View metrics
python3 src/SIMPLE_METRICS_VIEWER.py
```

## 🎯 Usage

### 1. Generate Attacks
```bash
python3 src/Evil_Agent.py --attack-type prompt_injection
python3 src/Evil_Agent.py --attack-type jailbreak --role guest
```

### 2. Test Security
```bash
python3 src/Sec_Agent.py "malicious query" --role guest --clearance PUBLIC
python3 src/Sec_Agent.py "legitimate query" --role analyst --clearance INTERNAL
```

### 3. Run Tests
```bash
# Blind testing
python3 src/Good_Vs_Evil.py --test-type blind --blind-tests 50

# All attack types
python3 src/Good_Vs_Evil.py --test-type all --role analyst

# Specific attack
python3 src/Good_Vs_Evil.py --test-type single --attack-type privilege_escalation
```

### 4. Training & Metrics
```bash
# Run training iterations (tracks failures for analysis)
python3 src/train_security_agent.py --iterations 5

# Check training progress
cat data/training/training_iteration_*.json
```

## 📊 Metrics & Monitoring

### Tracked Metrics (When Metrics Exporter Configured)
- **Detection Results**: Attacks blocked vs allowed
- **Response Time**: Request processing duration
- **Request Volume**: Total requests processed
- **Threat Detection**: Threats detected by type
- **Threats Blocked**: Attacks blocked over time
- **Training Events**: Learning events tracked

### Data Persistence
1. **Prometheus Volume**: `/var/lib/docker/volumes/vecsec_prometheus_data` - Historical metrics (when Docker running)
2. **Grafana Volume**: `/var/lib/docker/volumes/vecsec_grafana_data` - Dashboard configs
3. **JSON Backup**: `data/training/vecsec_metrics.json` - Auto-saves every 30s

### View Dashboards
```bash
# Grafana - Full dashboards (requires Docker)
open http://localhost:3000  # Login: admin/vecsec_admin

# Prometheus - Raw metrics (requires Docker)
open http://localhost:9090

# Simple viewer (works without Docker)
python3 src/SIMPLE_METRICS_VIEWER.py

# JSON backup
cat data/training/vecsec_metrics.json
```

## 📊 Training & Analysis

The system tracks test results and failures for analysis:

**How it works:**
1. Runs tests → Finds failures
2. Tracks failures → Stores in training data
3. Generates reports → Analysis of security gaps
4. Pattern tracking → Logs attack patterns for review

**Training Data:**
- `data/training/training_iteration_*.json` - Per-iteration results
- `data/training/vecsec_metrics.json` - Overall metrics
- `data/attacks/malicious_inputs_*.json` - Exported attack data

**Note**: Current implementation tracks failures but doesn't automatically improve detection. Patterns are logged for manual analysis and system improvement. See `TASKS.md` for planned learning improvements.

## 🏗️ Architecture

```
VecSec/
├── src/                     # 🐍 Core Python Files
│   ├── Evil_Agent.py        # 🔴 Attack generator
│   ├── Sec_Agent.py         # 🛡️ Security enforcement
│   ├── Good_Vs_Evil.py      # ⚔️ Test framework
│   ├── Legitimate_Agent.py   # 🟢 Legitimate queries (FP testing)
│   ├── train_security_agent.py # 📊 Training tracker
│   ├── metrics_exporter.py # 📈 Prometheus metrics
│   └── ...
│
├── data/                     # 📊 Data Files
│   ├── attacks/             # Attack exports
│   ├── training/             # Training iterations
│   └── test_results/        # Test outputs
│
├── docs/                     # 📚 Documentation
├── monitoring/              # 📊 Grafana dashboards
└── scripts/                 # 🔧 Shell scripts
```

## 🔐 Security Features

### Role-Based Access Control
- **guest**: PUBLIC clearance
- **analyst**: INTERNAL clearance
- **superuser**: CONFIDENTIAL clearance
- **admin**: SECRET clearance

### Threat Detection
**Always Blocked:**
- Prompt injection
- Obfuscation
- Jailbreak
- Privilege escalation

**Role-Dependent:**
- Data exfiltration (admin/superuser allowed)
- Social engineering (admin/superuser allowed)

### Clearance Enforcement
Users cannot access data beyond their clearance level:
```
PUBLIC (1) < INTERNAL (2) < CONFIDENTIAL (3) < SECRET (4)
```

## ⚡ Performance & Configuration

### Pattern-Based Detection (Default)
- **Works Out of Box**: Pattern matching against known attack keywords
- **Fast**: No external API calls required
- **Reliable**: Consistent threat detection based on rules

### Baseten Integration (Optional)
- **Requires Configuration**: Set `BASETEN_MODEL_ID` and `BASETEN_API_KEY`
- **Semantic Detection**: Uses Qwen3 8B model for embedding-based similarity
- **Batch Processing**: Processes multiple embeddings in single API call (when configured)
- **Performance**: Significantly faster when batching enabled (requires BaseTen setup)

**Note**: Without BaseTen API keys, the system uses pattern-based detection only. Semantic similarity detection requires BaseTen credentials.

## 📈 Demo Results

After running `src/SHOWCASE_DEMO.py`:

```bash
# View training metrics
cat data/training/training_iteration_*.json

# View test results
cat data/test_results/test_*.json
```

## 🎯 Key Features

### 1. Security Enforcement
- **Pattern-Based Detection**: Keyword matching for known attacks
- **RLS Enforcement**: Clearance level and role-based access control
- **Multi-Tenant Isolation**: Cross-tenant access prevention
- **7+ Attack Types**: Comprehensive attack coverage

### 2. Testing Framework
- **Blind Testing**: Mixed legitimate and malicious queries
- **Automated Testing**: Batch testing across attack types
- **False Positive Detection**: Tests legitimate queries to ensure they're allowed
- **Comprehensive Reporting**: Detailed security assessment reports

### 3. Monitoring (When Configured)
- **Prometheus Integration**: Metrics collection (requires Docker)
- **Grafana Dashboards**: Visualization (requires Docker)
- **JSON Metrics**: Simple metrics export without Docker
- **Real-Time Tracking**: Request and threat metrics

### 4. Attack Generation
- **7 Attack Types**: Prompt injection, jailbreak, privilege escalation, etc.
- **Static Templates**: Reliable, predictable attack patterns
- **LLM Generation**: Optional LLM-based attack generation (requires API keys)
- **Privilege Escalation**: Tests role/clearance violations

## 📁 File Structure

```
VecSec/
├── src/                          # 🐍 Core Python Files
│   ├── Sec_Agent.py              # Security enforcement
│   ├── Evil_Agent.py             # Attack generator
│   ├── Legitimate_Agent.py       # Legitimate queries (FP testing)
│   ├── Good_Vs_Evil.py           # Testing framework
│   ├── train_security_agent.py   # Training tracker
│   └── ...
│
├── data/                         # 📊 Data Files
│   ├── attacks/                  # Attack exports
│   ├── training/                 # Training iterations & metrics
│   └── test_results/             # Test outputs
│
├── docs/                         # 📚 Documentation
├── monitoring/                   # 📊 Grafana/Prometheus configs
└── scripts/                      # 🔧 Setup scripts
```

See `PROJECT_STRUCTURE.md` for complete structure details.

## 🧪 Testing

### Test Examples
```bash
# Single attack test
python3 src/Good_Vs_Evil.py --test-type single --attack-type jailbreak --role guest

# Blind testing (mixed legitimate + malicious)
python3 src/Good_Vs_Evil.py --test-type blind --blind-tests 100

# All attack types
python3 src/Good_Vs_Evil.py --test-type all --role analyst

# Clearance enforcement
python3 src/test_clearance_enforcement.py
```

## 🔄 Continuous Integration

VecSec uses GitHub Actions for automated quality gates on every pull request and push to main.

### CI Pipeline Structure

The CI pipeline (`.github/workflows/ci.yml`) runs three jobs in parallel:

1. **Lint & Type Check** (`lint`)
   - Runs `ruff` for fast linting and formatting checks
   - Runs `mypy src/` for static type checking (note: use `mypy src/` not `mypy .`)
   - Fast feedback (< 2 minutes)

2. **Unit Tests** (`unit-tests`)
   - Runs pytest with coverage reporting
   - Uses mock retrieval (`USE_REAL_VECTOR_RETRIEVAL=false`)
   - Tests all unit tests in `src/sec_agent/tests/`
   - Generates coverage reports

3. **Integration Smoke Tests** (`integration-smoke`)
   - Runs on main branch and PRs
   - Uses real vector retrieval (`USE_REAL_VECTOR_RETRIEVAL=true`)
   - Tests migration and integration scenarios
   - Uses ChromaDB for real vector store testing

### Running Tests Locally

#### Quick: Run Full CI Pipeline Locally
```bash
# Run the entire CI pipeline locally (matches GitHub Actions)
./scripts/run_ci_locally.sh
```

This script runs:
1. **Lint & Type Check**: `ruff check`, `ruff format --check`, `mypy`
2. **Unit Tests**: Pytest with mock retrieval and coverage
3. **Integration Tests**: Pytest with real ChromaDB vector store

#### Manual: Individual CI Steps

**1. Lint & Type Check (matches CI lint job)**
```bash
ruff check src/ scripts/ --output-format=github
ruff format --check src/ scripts/
mypy . --ignore-missing-imports --no-strict-optional
```

**2. Unit Tests (matches CI unit-tests job)**
```bash
export USE_REAL_VECTOR_RETRIEVAL=false
export LOG_LEVEL=WARNING
pytest src/sec_agent/tests/ \
  --maxfail=1 \
  --disable-warnings \
  --cov=src/sec_agent \
  --cov-report=term-missing \
  -v
```

**3. Integration Smoke Tests (matches CI integration-smoke job)**
```bash
export USE_REAL_VECTOR_RETRIEVAL=true
export USE_CHROMA=true
export CHROMA_PATH=./chroma_db_ci
pytest src/sec_agent/tests/test_rag_orchestrator_migration.py \
  src/sec_agent/tests/test_metadata_generator_real.py \
  -v
```

#### All Tests (Quick)
```bash
# Run all tests (unit + integration)
pytest src/sec_agent/tests/ -v
```

### Adding New CI Jobs

To add a new job to the CI pipeline:

1. Edit `.github/workflows/ci.yml`
2. Add a new job under the `jobs:` section
3. Use the same Python setup pattern:
   ```yaml
   new-job:
     name: New Job Name
     runs-on: ubuntu-latest
     steps:
       - uses: actions/checkout@v4
       - uses: actions/setup-python@v5
         with:
           python-version: '3.11'
       - run: pip install -r requirements.txt
       - run: # Your test command
   ```
4. Add the job to `test-summary` job's `needs:` list

### CI Requirements

**All pull requests must pass CI before merging:**
- ✅ Lint checks must pass (no ruff errors)
- ✅ Type checks must pass (mypy warnings are non-blocking)
- ✅ All unit tests must pass
- ✅ Integration smoke tests must pass (on main branch)

### Security Scanning

- **CodeQL**: Automated security scanning runs weekly and on every push to main
- **Dependabot**: Weekly dependency update PRs for pip packages
- See `.github/workflows/codeql.yml` and `.github/dependabot.yml`

### CI Performance

- **Target**: < 7 minutes total
- **Caching**: Pip dependencies are cached
- **Concurrency**: Stale runs are automatically cancelled

## 📊 Attack Types

1. **Prompt Injection**: Override system instructions
2. **Data Exfiltration**: Unauthorized data export
3. **Social Engineering**: Authority impersonation
4. **Obfuscation**: Encoded script execution
5. **Jailbreak**: Developer mode activation
6. **Privilege Escalation**: Role/clearance mismatch
7. **Data Poisoning**: Malicious training data

## 🔧 Configuration

### Environment Variables (.env)
```env
GOOGLE_API_KEY=your_key
OPENAI_API_KEY=your_key
BASETEN_MODEL_ID=your_model
BASETEN_API_KEY=your_key

# Optional: Use ChromaDB for persistent vector storage
USE_CHROMA=true
CHROMA_API_KEY=your_chroma_key  # For cloud storage
CHROMA_HOST=chromadb.net
```

### Vector Storage Options
1. **InMemory (Default)**: Fast, no persistence
2. **ChromaDB (Persistent)**: Local storage in `./chroma_db/`
3. **ChromaDB Cloud**: Cloud-hosted with API key

### Customize Policies
Edit `src/Sec_Agent.py`:
```python
ROLE_POLICIES = {
    "admin": {"max_clearance": "SECRET", "cross_tenant_access": True},
    "analyst": {"max_clearance": "INTERNAL", "cross_tenant_access": False},
}
```

## 📚 Documentation

- **README.md** (this file) - Complete guide
- **TASKS.md** - Development tasks and tickets
- **PROJECT_STRUCTURE.md** - File organization guide
- **QUICK_START.md** - Quick reference
- **docs/VECSEC_README.md** - Advanced technical docs
- **docs/ISSUES_FOUND.md** - Known issues and solutions
- **docs/MONITORING.md** - Prometheus/Grafana setup

## 🎬 Demo Commands

### Quick Demo (2 minutes)
```bash
python3 src/SHOWCASE_DEMO.py
```

### Training Iterations (10 minutes)
```bash
python3 src/train_security_agent.py --iterations 5 --delay 30
```

### Test All Features (15 minutes)
```bash
python3 src/Good_Vs_Evil.py --test-type blind --blind-tests 100 --export-malicious
```

## 🎯 Use Cases

1. **Security Validation**: Test RLS implementations
2. **Penetration Testing**: Simulate real-world attacks  
3. **Compliance Auditing**: Validate access controls
4. **Continuous Security**: Automated testing
5. **Training Teams**: Security awareness

## 📞 Getting Help

- **Demo**: `python3 src/SHOWCASE_DEMO.py`
- **Tasks**: See `TASKS.md` for known issues and planned improvements
- **Structure**: See `PROJECT_STRUCTURE.md` for file organization
- **Metrics**: Check `data/training/vecsec_metrics.json`
- **Docs**: See `docs/` folder for detailed documentation

## ⚠️ Disclaimer

**For authorized security testing only.** Unauthorized use against systems you don't own is illegal and unethical.

## 📜 License

MIT License - See LICENSE file for details.

---

**🚀 Start your demo: `python3 src/SHOWCASE_DEMO.py`**

**📋 See tasks: `cat TASKS.md`**
