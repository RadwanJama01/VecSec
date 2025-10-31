# VecSec Project Structure

## 📁 Directory Organization

```
VecSec/
├── src/                          # 🐍 CORE PYTHON FILES
│   ├── __init__.py
│   ├── Sec_Agent.py             # Security enforcement agent
│   ├── Evil_Agent.py            # Attack generator
│   ├── Legitimate_Agent.py      # Legitimate query generator
│   ├── Good_Vs_Evil.py          # Security testing framework
│   ├── train_security_agent.py  # Continuous learning
│   ├── run_training.py          # Training runner
│   ├── metrics_exporter.py      # Prometheus metrics
│   ├── SHOWCASE_DEMO.py         # Demo script
│   ├── SIMPLE_METRICS_VIEWER.py # Metrics viewer
│   ├── test_clearance_enforcement.py # Tests
│   └── CHROMA_CLOUD_SETUP.py    # ChromaDB setup
│
├── data/                         # 📊 DATA FILES
│   ├── attacks/                  # Attack data
│   │   ├── malicious_inputs_*.json
│   │   └── my_attacks.json
│   ├── training/                 # Training data
│   │   ├── training_iteration_*.json
│   │   └── vecsec_metrics.json
│   ├── test_results/             # Test outputs
│   │   └── test_*.json
│   └── *.txt                     # Status/change logs
│
├── docs/                         # 📚 DOCUMENTATION
│   ├── README.md                 # (stays at root)
│   ├── VECSEC_README.md
│   ├── ISSUES_FOUND.md
│   ├── TRAINING_REQUIREMENTS.md
│   ├── BATCHING_OPTIMIZATION.md
│   ├── CODEBASE_ANALYSIS.md
│   ├── MONITORING.md
│   └── PERFORMANCE_ISSUES.txt
│
├── scripts/                      # 🔧 SHELL SCRIPTS
│   ├── install_dependencies.sh
│   └── START_EVERYTHING.sh
│
├── monitoring/                   # 📊 MONITORING CONFIG
│   ├── prometheus.yml
│   ├── vecsec_rules.yml
│   └── grafana/
│
├── chroma_db/                    # 💾 DATABASE
│   └── ... (vector database files)
│
├── README.md                     # Main documentation
├── requirements.txt              # Dependencies
├── docker-compose.monitoring.yml # Docker config
├── Dockerfile                    # Docker image
└── .env                          # Environment variables
```

## 🚀 Quick Reference

### Core Python Files (src/)
- **Sec_Agent.py** - Main security agent with RLS enforcement
- **Evil_Agent.py** - Generates malicious attack queries
- **Legitimate_Agent.py** - Generates legitimate queries for FP testing
- **Good_Vs_Evil.py** - Security testing framework
- **train_security_agent.py** - Continuous learning system

### Running Core Files

```bash
# From project root, run with:
python3 src/Sec_Agent.py "query" --role analyst --clearance INTERNAL
python3 src/Evil_Agent.py --attack-type prompt_injection
python3 src/Good_Vs_Evil.py --test-type blind --blind-tests 20
python3 src/train_security_agent.py --iterations 5
```

### Data Files (data/)
- `data/attacks/` - Exported attack queries
- `data/training/` - Training iterations and metrics
- `data/test_results/` - Test output JSON files

### Documentation (docs/)
- Main docs in `docs/` folder
- README.md stays at root for GitHub visibility

### Scripts (scripts/)
- Shell scripts for setup and automation

## 📝 Note on Imports

If you need to import between modules in src/, use:
```python
from src.Sec_Agent import rag_with_rlsa
from src.Evil_Agent import generate_attack
```

Or add src to PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

