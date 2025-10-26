# VecSec - Start Here! 🚀

## YOU DON'T NEED DOCKER!

Everything works without it. Here's what you can do RIGHT NOW:

### Quick Test (1 minute)
```bash
python3 Evil_Agent.py --attack-type prompt_injection
```

### Full Blind Testing (5 minutes)
```bash
# Run comprehensive security tests
python3 Good_Vs_Evil.py --test-type blind --blind-tests 20

# Output includes:
# - Each attack tested
# - Timing for each request (ms)
# - Security score
# - Summary report
```

### Export Results (10 minutes)
```bash
# Generate lots of test data
python3 Good_Vs_Evil.py --test-type blind --blind-tests 100 --export-malicious

# Results saved to: malicious_inputs_TIMESTAMP.json
```

## What Works Now (No Docker!)

✅ **Evil_Agent**: Generates attack patterns
✅ **Sec_Agent**: Security enforcement system  
✅ **Legitimate_Agent**: Generates legitimate queries
✅ **Good_Vs_Evil**: Blind testing with timing
✅ **Timing Stats**: Every request shows timing
✅ **Reports**: JSON export for analysis

## Adding Monitoring Later

If you want dashboards later:

1. Open Docker Desktop manually
2. Run: `docker-compose -f docker-compose.monitoring.yml up -d`
3. Visit: http://localhost:3000

But you don't need it to test!

## File Structure

```
VecSec/
├── Evil_Agent.py          ✅ Attack generator
├── Sec_Agent.py           ✅ Security system  
├── Legitimate_Agent.py    ✅ Legitimate queries
├── Good_Vs_Evil.py        ✅ Test framework
├── metrics_exporter.py    🔜 Optional monitoring
└── monitoring/            🔜 Optional dashboards
```

## Commands You Can Use Now

```bash
# Test a single attack
python3 Evil_Agent.py --attack-type jailbreak

# Test the security system
python3 Sec_Agent.py "your malicious query" --role analyst

# Run blind tests
python3 Good_Vs_Evil.py --test-type blind --blind-tests 50

# Test specific attack
python3 Good_Vs_Evil.py --test-type single --attack-type prompt_injection

# Export results
python3 Good_Vs_Evil.py --test-type blind --export-malicious
```

## What Each Command Does

### Evil_Agent.py
- Generates malicious prompts
- Uses static templates (no API needed)
- Multiple attack types

### Sec_Agent.py  
- Security enforcement system
- Blocks malicious queries
- Uses pattern matching

### Good_Vs_Evil.py
- Blind testing framework
- Mixes legitimate + malicious inputs
- Shows timing for each request
- Generates security reports

## Next Steps

1. ✅ Run tests: `python3 Good_Vs_Evil.py --test-type blind --blind-tests 20`
2. ✅ Check timing stats in output
3. ✅ Review security reports
4. 🔜 Add BaseTen API for semantic detection (optional)
5. 🔜 Add Docker for monitoring dashboards (optional)

## No Docker Required!

Start testing now:
```bash
python3 Good_Vs_Evil.py --test-type blind --blind-tests 10
```

