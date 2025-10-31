# 🚀 VecSec Quick Start Guide

## New Project Structure

All core Python files are now in `src/` directory for better organization:

```
VecSec/
├── src/              # 🐍 All Python core files here
├── data/             # 📊 All JSON/txt data files here
├── docs/             # 📚 All documentation here
├── scripts/          # 🔧 Shell scripts here
└── README.md         # Main docs
```

## Running Core Files

All commands now use `src/` prefix:

```bash
# Security Agent
python3 src/Sec_Agent.py "query" --role analyst --clearance INTERNAL

# Generate Attacks
python3 src/Evil_Agent.py --attack-type prompt_injection

# Test Security
python3 src/Good_Vs_Evil.py --test-type blind --blind-tests 20

# Training
python3 src/train_security_agent.py --iterations 5

# Metrics
python3 src/SIMPLE_METRICS_VIEWER.py
```

## File Locations

- **Core Python**: `src/` directory
- **Training Data**: `data/training/`
- **Attack Data**: `data/attacks/`
- **Test Results**: `data/test_results/`
- **Documentation**: `docs/` directory

## See Full Structure

```bash
cat PROJECT_STRUCTURE.md
```

