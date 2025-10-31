# 🎫 VecSec Development Tasks & Tickets

## 🔴 Critical Priority

### [TICKET-001] Fix Random Embeddings in Semantic Detection
**Priority**: CRITICAL 🔴  
**Status**: OPEN  
**Files**: `src/Sec_Agent.py`

**Problem**:
- Returns `np.random.rand(768)` when BaseTen API not configured (line 124)
- Returns random embeddings when batch not full (line 141)
- Returns random embeddings after training complete (line 118)
- Semantic similarity checks become meaningless with random vectors

**Impact**: Security checks based on random embeddings don't work properly

**Solution**:
- Raise `ValueError` when embeddings cannot be generated
- Fix batch processing to flush immediately instead of returning random
- Update `check_semantic_threat()` to handle embedding failures gracefully
- Remove all `np.random.rand(768)` fallbacks

**Acceptance Criteria**:
- [ ] No random embeddings returned as fallback
- [ ] Proper error messages when BaseTen not configured
- [ ] Batch processing returns real embeddings or errors
- [ ] Semantic detection disabled gracefully when embeddings unavailable

---

### [TICKET-002] Implement Actual Learning (Not Just Tracking)
**Priority**: CRITICAL 🔴  
**Status**: OPEN  
**Files**: `src/train_security_agent.py`, `src/Sec_Agent.py`

**Problem**:
- `train_security_agent.py` only appends failures to `learning_data`
- Doesn't actually improve detection rules or patterns
- "Learning" is just logging, not improvement

**Impact**: System doesn't actually learn or improve accuracy

**Solution**:
- Use learned patterns to update detection rules
- Feed learned patterns back into `ContextualThreatEmbedding`
- Update threat detection patterns based on failures
- Implement actual pattern matching improvement

**Acceptance Criteria**:
- [ ] Failures are analyzed and patterns extracted
- [ ] Learned patterns are used to improve detection
- [ ] Accuracy improves over iterations (measurable)
- [ ] Patterns stored and reused in future detection

---

## 🟠 High Priority

### [TICKET-003] Make Privilege Escalation Generation Algorithmic
**Priority**: HIGH 🟠  
**Status**: OPEN  
**Files**: `src/Evil_Agent.py`

**Problem**:
- Hardcoded list of only 6 escalation scenarios (lines 198-205)
- Doesn't generate all possible role/clearance/tenant combinations
- Missing many potential attack vectors

**Solution**:
- Algorithmically generate ALL role combinations:
  - Roles: guest → analyst → superuser → admin (all paths)
  - Clearances: PUBLIC → INTERNAL → CONFIDENTIAL → SECRET (all violations)
  - Tenants: tenantA ↔ tenantB (both directions)
- Generate all permutations programmatically

**Acceptance Criteria**:
- [ ] Generates all role escalation combinations (not just 6)
- [ ] Generates all clearance level violations
- [ ] Generates all cross-tenant access attempts
- [ ] Removes hardcoded escalation_scenarios list

---

### [TICKET-004] Fix Batch Processing Race Condition
**Priority**: HIGH 🟠  
**Status**: OPEN  
**Files**: `src/Sec_Agent.py`

**Problem**:
- Returns random embeddings while waiting for batch to fill (line 141)
- Early requests get random vectors before batch processes
- Security checks use random data

**Solution**:
- Flush batch immediately instead of waiting
- OR implement proper async/await for batch processing
- OR reduce batch_size to 1 for immediate processing
- Ensure real embeddings always returned

**Acceptance Criteria**:
- [ ] No random embeddings while batch filling
- [ ] All requests get real embeddings
- [ ] Batch processing works correctly
- [ ] Performance acceptable with immediate flushing

---

### [TICKET-005] Clarify Multi-LLM Provider Usage
**Priority**: HIGH 🟠  
**Status**: OPEN  
**Files**: `src/Evil_Agent.py`

**Problem**:
- Supports Google, OpenAI, Flash - but only one used at a time
- No clear reason for having multiple providers
- Unclear if they should be:
  - Fallback mechanism?
  - Diversity for attack generation?
  - Redundant?

**Solution**:
- Document why multiple providers exist
- Implement fallback if one fails
- OR use multiple for attack diversity
- OR remove unused providers

**Acceptance Criteria**:
- [ ] Clear documentation of multi-provider purpose
- [ ] Fallback mechanism OR diversity usage OR removal
- [ ] No unused code

---

## 🟡 Medium Priority

### [TICKET-006] Make Attack Generation More Dynamic
**Priority**: MEDIUM 🟡  
**Status**: OPEN  
**Files**: `src/Evil_Agent.py`

**Problem**:
- Static `ATTACK_TYPES` dict with hardcoded strings (lines 132-177)
- Even LLM generation uses static templates (lines 309-315)
- Limited variation, predictable patterns

**Solution**:
- Use LLM to generate truly novel attacks (not template-based)
- Learn from successful attack patterns
- Evolve attacks based on what bypasses security
- Implement attack mutation/variation

**Acceptance Criteria**:
- [ ] LLM generates novel attacks, not template-based
- [ ] Attacks evolve based on success/failure
- [ ] Less predictable attack patterns
- [ ] Attack catalog can grow dynamically

---

### [TICKET-007] Add Input Sanitization
**Priority**: MEDIUM 🟡  
**Status**: OPEN  
**Files**: `src/Sec_Agent.py`

**Problem**:
- User queries passed directly without sanitization
- No validation of input length, characters, etc.
- Potential injection vulnerabilities

**Solution**:
- Sanitize and truncate input queries
- Remove dangerous characters
- Validate input length
- Normalize whitespace

**Acceptance Criteria**:
- [ ] All user input sanitized
- [ ] Max length enforced
- [ ] Dangerous characters removed
- [ ] Input validation in place

---

### [TICKET-008] Add Rate Limiting
**Priority**: MEDIUM 🟡  
**Status**: OPEN  
**Files**: `src/Sec_Agent.py`

**Problem**:
- No protection against API abuse or DoS
- Attackers could flood system with requests

**Solution**:
- Implement rate limiting per user/tenant
- Track requests per time window
- Block excessive requests
- Log rate limit violations

**Acceptance Criteria**:
- [ ] Rate limiting per user/tenant
- [ ] Configurable limits
- [ ] Proper error responses
- [ ] Metrics tracking rate limits

---

## 🟢 Low Priority

### [TICKET-009] Add Unit Tests
**Priority**: LOW 🟢  
**Status**: OPEN  
**Files**: New test files needed

**Problem**:
- No unit tests for core security functions
- No way to verify fixes don't break functionality

**Solution**:
- Create `tests/` directory
- Add unit tests for:
  - `QwenEmbeddingClient.get_embedding()`
  - `rlsa_guard_comprehensive()`
  - `generate_attack()`
  - `check_semantic_threat()`

**Acceptance Criteria**:
- [ ] Test suite for core functions
- [ ] Tests cover edge cases
- [ ] CI/CD integration
- [ ] Good test coverage (>80%)

---

### [TICKET-010] Improve Error Handling
**Priority**: LOW 🟢  
**Status**: OPEN  
**Files**: Multiple

**Problem**:
- Uses `print()` instead of proper logging
- Silent failures in many places
- Difficult to debug in production

**Solution**:
- Replace print statements with logging
- Proper error handling with exceptions
- Log levels (DEBUG, INFO, WARNING, ERROR)
- Structured logging

**Acceptance Criteria**:
- [ ] Proper logging infrastructure
- [ ] No silent failures
- [ ] Log files for debugging
- [ ] Structured log format

---

### [TICKET-011] Add Type Hints Throughout
**Priority**: LOW 🟢  
**Status**: OPEN  
**Files**: All Python files

**Problem**:
- Missing type hints in many functions
- Reduces code maintainability

**Solution**:
- Add complete type hints to all functions
- Use `typing` module properly
- Improve IDE autocomplete

**Acceptance Criteria**:
- [ ] All functions have type hints
- [ ] Type checking passes
- [ ] Better IDE support

---

## 📋 Task Summary

| Priority | Count | Tickets |
|----------|-------|---------|
| 🔴 Critical | 2 | TICKET-001, TICKET-002 |
| 🟠 High | 3 | TICKET-003, TICKET-004, TICKET-005 |
| 🟡 Medium | 3 | TICKET-006, TICKET-007, TICKET-008 |
| 🟢 Low | 3 | TICKET-009, TICKET-010, TICKET-011 |

**Total**: 11 tickets

---

## 🎯 Quick Wins (Start Here)

These can be fixed quickly (< 1 hour each):

1. **TICKET-001** (partially) - Remove random embeddings, add proper errors
2. **TICKET-007** - Add input sanitization (simple function)
3. **TICKET-010** - Replace prints with logging (quick refactor)

---

## 📝 Notes

- All tickets reference actual code locations
- Priority based on security impact
- Acceptance criteria clearly defined
- Some tickets may be blocked by others (e.g., TICKET-002 needs TICKET-001)

