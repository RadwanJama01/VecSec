# ✅ Refactoring Complete - New Project Structure

## 📁 What Changed

### ✅ Core Python Files → `src/`
All Python files are now organized in `src/` directory:
- `Sec_Agent.py` → `src/Sec_Agent.py`
- `Evil_Agent.py` → `src/Evil_Agent.py`
- `Legitimate_Agent.py` → `src/Legitimate_Agent.py`
- `Good_Vs_Evil.py` → `src/Good_Vs_Evil.py`
- `train_security_agent.py` → `src/train_security_agent.py`
- `metrics_exporter.py` → `src/metrics_exporter.py`
- And all other Python files...

### ✅ Data Files → `data/`
- Attack data → `data/attacks/`
- Training data → `data/training/`
- Test results → `data/test_results/`
- Status logs → `data/`

### ✅ Documentation → `docs/`
- All `.md` files (except README.md) → `docs/`

### ✅ Scripts → `scripts/`
- Shell scripts → `scripts/`

## 🚀 How to Use

### Running Core Files
```bash
# All commands now use src/ prefix
python3 src/Sec_Agent.py "query" --role analyst
python3 src/Good_Vs_Evil.py --test-type blind --blind-tests 20
python3 src/train_security_agent.py --iterations 5
```

### Data Files
```bash
# Training data
ls data/training/

# Attack data
ls data/attacks/

# Test results
ls data/test_results/
```

## 📋 Updated Imports

All internal imports have been updated with fallbacks:
- Primary: `from src.Module import ...`
- Fallback: `from Module import ...` (if in same directory)

## ✨ Benefits

1. **Clear Separation**: Python files vs data vs docs
2. **Easy Navigation**: All core logic in one place (`src/`)
3. **Better Organization**: Data files organized by purpose
4. **Cleaner Root**: Only essential files at root level

## 📖 See Structure

```bash
cat PROJECT_STRUCTURE.md
cat QUICK_START.md
```

