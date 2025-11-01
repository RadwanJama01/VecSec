# VecSec Test Suite

Unit tests and integration tests for VecSec security framework.

## Running Tests

```bash
# Run all tests
python3 -m pytest tests/

# Run specific test file
python3 tests/test_sec_agent.py

# Run with verbose output
python3 -m pytest tests/ -v

# Run with coverage
python3 -m pytest tests/ --cov=src --cov-report=html
```

## Test Files

- `test_sec_agent.py` - Tests for security enforcement and RLS
- `test_evil_agent.py` - Tests for attack generation
- `test_legitimate_agent.py` - Tests for legitimate query generation
- `test_good_vs_evil.py` - Tests for testing framework
- `test_integration.py` - Integration tests for component interaction

## Test Coverage Goals

- [ ] Security enforcement (RLS, RBAC)
- [ ] Attack generation (all types)
- [ ] Threat detection patterns
- [ ] Embedding client (when fixed)
- [ ] Learning system (when implemented)
- [ ] Integration between components

## Known Test Limitations

- Embedding tests disabled until TICKET-001 is fixed
- Learning tests disabled until TICKET-002 is implemented
- Some tests require BaseTen API (marked with @skipIf)

