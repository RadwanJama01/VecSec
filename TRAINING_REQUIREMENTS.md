# 🎓 Training Requirements - Everything You Need Running

## ✅ What's Needed for Training

### 1. Docker Services (Required) ✅
```bash
# Start monitoring services
./START_EVERYTHING.sh

# Or manually:
docker-compose -f docker-compose.monitoring.yml up -d
```

**Required Services:**
- ✅ Prometheus (localhost:9090)
- ✅ Grafana (localhost:3000)
- Both are now running

### 2. Metrics Exporter (Required) ✅
```bash
# Start metrics exporter (keep running!)
python3 start_metrics_exporter.py &

# Or background:
nohup python3 start_metrics_exporter.py > metrics.log 2>&1 &
```

**Status:** ✅ Running on port 9091

### 3. ChromaDB (Optional but Recommended) ✅
```bash
# Status: NOW ENABLED
USE_CHROMA=true in .env
```

**Why ChromaDB for Training:**
- Stores learned threat patterns persistently
- Allows continuous learning across sessions
- Better than in-memory storage

**Status:** ✅ Installed and ready
- Location: `./chroma_db/`
- Collection: `vecsec_threat_patterns`

## 🎯 Running Training

### Quick Training
```bash
python3 train_security_agent.py --iterations 3
```

### Full Training
```bash
python3 train_security_agent.py --iterations 10 --delay 30
```

### What Happens During Training:
1. **Run Tests** - Generates attacks and tests security
2. **Identify Failures** - Finds false positives/negatives
3. **Learn Patterns** - Adds to ChromaDB threat patterns
4. **Update Metrics** - Tracks progress in Prometheus
5. **Save to JSON** - Exports results

## 📊 Data Flow

```
Training Script
    ↓
Generates Attacks (Evil_Agent)
    ↓
Tests Security (Sec_Agent)
    ↓
Tracks Metrics → Prometheus → Grafana
    ↓
Learns Patterns → ChromaDB (persistent)
    ↓
Saves Results → JSON files
```

## ✅ Current Setup Status

| Component | Status | Location |
|-----------|--------|----------|
| Prometheus | ✅ Running | localhost:9090 |
| Grafana | ✅ Running | localhost:3000 |
| Metrics Exporter | ✅ Running | localhost:9091 |
| ChromaDB | ✅ Enabled | ./chroma_db/ |
| Vector Storage | ✅ Persisting | ChromaDB collection |

## 🧪 Test Training Now

```bash
# Start everything if not running
./START_EVERYTHING.sh

# Keep metrics exporter running
python3 start_metrics_exporter.py &

# Run training
python3 train_security_agent.py --iterations 3

# View results
cat learning_metrics.json
cat training_iteration_*.json

# Check ChromaDB
ls -lh chroma_db/
```

## 📈 View Results

### In Grafana:
```bash
open http://localhost:3000
# Login: admin / vecsec_admin
```

### In JSON Files:
```bash
cat learning_metrics.json
cat training_iteration_*.json
```

### Check ChromaDB:
```bash
python3 -c "
import chromadb
client = chromadb.PersistentClient(path='./chroma_db')
collection = client.get_collection('vecsec_threat_patterns')
print(f'Patterns learned: {collection.count()}')
"
```

## ✅ Summary

**Everything Required is Running:**
- ✅ Docker (Prometheus + Grafana)
- ✅ Metrics Exporter
- ✅ ChromaDB (just enabled!)
- ✅ All integrations working

**Ready to train!** Just run:
```bash
python3 train_security_agent.py --iterations 3
```

