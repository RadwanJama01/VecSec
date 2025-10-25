"""
Attack Logger for VecSec Red Team Framework
==========================================

This module implements comprehensive logging and audit trail functionality
for the adversarial testing system. It provides structured logging, binary
embedding storage, vulnerability detection, and analytics capabilities.

Author: VecSec Team
"""

import asyncio
import json
import logging
import sqlite3
import hashlib
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import gzip

from llama_attack_generator import AttackSpec, AttackResult

logger = logging.getLogger(__name__)

@dataclass
class VulnerabilityDiscovery:
    """Represents a discovered vulnerability."""
    vulnerability_id: str
    title: str
    description: str
    severity: str  # critical, high, medium, low
    attack_ids: List[str]
    exploitation_method: str
    affected_components: List[str]
    reproduction_steps: str
    mitigation_status: str  # open, in_progress, mitigated
    mitigation_strategy: Optional[str] = None
    discovered_at: datetime = None
    mitigated_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.discovered_at is None:
            self.discovered_at = datetime.now()

@dataclass
class AttackExecution:
    """Represents a complete attack execution record."""
    attack_id: str
    campaign_id: Optional[str]
    mode: str
    method: str
    seed: int
    config: Dict
    payload: Dict
    target_vulnerability: str
    expected_outcome: str
    success_criteria: List[str]
    evasion_techniques: List[str]
    threat_model: Dict
    executed_at: datetime
    execution_duration_ms: int
    sandbox_id: str
    success: bool
    evasion_successful: bool
    detected_as_malicious: bool
    response: Dict
    telemetry: Dict
    vulnerability_exposed: Optional[str] = None
    error: Optional[str] = None

class AttackLogger:
    """
    Comprehensive logging system for attack execution and results.
    
    Features:
    - Structured attack logging
    - Binary embedding storage
    - Vulnerability detection
    - Analytics and reporting
    - Database management
    - Log rotation and cleanup
    """
    
    def __init__(self, 
                 db_path: str = "./attack_logs.db",
                 log_dir: str = "./logs",
                 config: Optional[Dict] = None):
        self.db_path = db_path
        self.log_dir = Path(log_dir)
        self.config = config or {}
        
        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Vulnerability tracking
        self.vulnerability_patterns = {}
        self.vulnerability_threshold = self.config.get('vulnerability_threshold', 3)
        
        logger.info(f"AttackLogger initialized with database: {db_path}")
    
    def _init_database(self):
        """Initialize SQLite database with required tables."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Attack executions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS attack_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    attack_id TEXT UNIQUE NOT NULL,
                    campaign_id TEXT,
                    mode TEXT NOT NULL,
                    method TEXT NOT NULL,
                    seed INTEGER,
                    config TEXT,
                    payload TEXT,
                    target_vulnerability TEXT,
                    expected_outcome TEXT,
                    success_criteria TEXT,
                    evasion_techniques TEXT,
                    threat_model TEXT,
                    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    execution_duration_ms INTEGER,
                    sandbox_id TEXT,
                    success BOOLEAN,
                    evasion_successful BOOLEAN,
                    detected_as_malicious BOOLEAN,
                    response TEXT,
                    telemetry TEXT,
                    vulnerability_exposed TEXT,
                    error TEXT
                )
            """)
            
            # Embedding traces table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embedding_traces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    attack_id TEXT NOT NULL,
                    trace_type TEXT NOT NULL,
                    embedding_vector BLOB,
                    dimension INTEGER,
                    l2_norm REAL,
                    metadata TEXT,
                    captured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (attack_id) REFERENCES attack_executions (attack_id)
                )
            """)
            
            # Response telemetry table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS response_telemetry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    attack_id TEXT NOT NULL,
                    status_code INTEGER,
                    response_time_ms INTEGER,
                    action_taken TEXT,
                    policy_violations TEXT,
                    threat_detections TEXT,
                    threat_level TEXT,
                    confidence REAL,
                    reasoning TEXT,
                    captured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (attack_id) REFERENCES attack_executions (attack_id)
                )
            """)
            
            # Attack campaigns table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS attack_campaigns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    campaign_id TEXT UNIQUE NOT NULL,
                    name TEXT,
                    description TEXT,
                    attack_count INTEGER,
                    success_rate REAL,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    config TEXT
                )
            """)
            
            # Campaign attacks linking table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS campaign_attacks (
                    campaign_id TEXT NOT NULL,
                    attack_id TEXT NOT NULL,
                    sequence_number INTEGER,
                    PRIMARY KEY (campaign_id, attack_id),
                    FOREIGN KEY (campaign_id) REFERENCES attack_campaigns (campaign_id),
                    FOREIGN KEY (attack_id) REFERENCES attack_executions (attack_id)
                )
            """)
            
            # Discovered vulnerabilities table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS discovered_vulnerabilities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vulnerability_id TEXT UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    severity TEXT,
                    attack_ids TEXT,
                    exploitation_method TEXT,
                    affected_components TEXT,
                    reproduction_steps TEXT,
                    mitigation_status TEXT,
                    mitigation_strategy TEXT,
                    discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    mitigated_at TIMESTAMP
                )
            """)
            
            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_attack_mode ON attack_executions (mode)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_attack_method ON attack_executions (method)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_executed_at ON attack_executions (executed_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_success ON attack_executions (success)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_evasion_successful ON attack_executions (evasion_successful)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_campaign_id ON attack_executions (campaign_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_vulnerability_severity ON discovered_vulnerabilities (severity)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_vulnerability_status ON discovered_vulnerabilities (mitigation_status)")
            
            conn.commit()
            conn.close()
            
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    async def log_attack(self, 
                        attack_spec: AttackSpec, 
                        result: AttackResult,
                        telemetry: Dict) -> str:
        """
        Log complete attack execution with results and telemetry.
        
        Args:
            attack_spec: Attack specification
            result: Attack result
            telemetry: Execution telemetry
            
        Returns:
            Attack ID
        """
        try:
            # Create attack execution record
            execution = AttackExecution(
                attack_id=attack_spec.attack_id,
                campaign_id=attack_spec.campaign_id,
                mode=attack_spec.mode,
                method=attack_spec.method,
                seed=attack_spec.seed,
                config=attack_spec.config,
                payload=attack_spec.payload,
                target_vulnerability=attack_spec.target_vulnerability,
                expected_outcome=attack_spec.expected_outcome,
                success_criteria=attack_spec.success_criteria,
                evasion_techniques=attack_spec.evasion_techniques,
                threat_model=attack_spec.threat_model,
                executed_at=datetime.now(),
                execution_duration_ms=int(result.execution_time * 1000),
                sandbox_id=telemetry.get('sandbox_id', 'unknown'),
                success=result.success,
                evasion_successful=result.evasion_successful,
                detected_as_malicious=result.detected_as_malicious,
                response=result.response,
                telemetry=telemetry,
                vulnerability_exposed=result.vulnerability_exposed,
                error=result.error
            )
            
            # Store in database
            await self._store_attack_execution(execution)
            
            # Store embedding traces if present
            if 'embedding_traces' in telemetry:
                await self._store_embedding_traces(
                    attack_spec.attack_id,
                    telemetry['embedding_traces']
                )
            
            # Store response telemetry
            await self._store_response_telemetry(
                attack_spec.attack_id,
                result.response,
                telemetry
            )
            
            # Check for vulnerability discovery
            if result.success and result.evasion_successful:
                await self._check_vulnerability_discovery(execution)
            
            logger.info(f"Logged attack execution: {attack_spec.attack_id}")
            return attack_spec.attack_id
            
        except Exception as e:
            logger.error(f"Failed to log attack {attack_spec.attack_id}: {e}")
            raise
    
    async def _store_attack_execution(self, execution: AttackExecution):
        """Store attack execution record in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO attack_executions (
                    attack_id, campaign_id, mode, method, seed, config, payload,
                    target_vulnerability, expected_outcome, success_criteria,
                    evasion_techniques, threat_model, executed_at,
                    execution_duration_ms, sandbox_id, success, evasion_successful,
                    detected_as_malicious, response, telemetry, vulnerability_exposed, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                execution.attack_id,
                execution.campaign_id,
                execution.mode,
                execution.method,
                execution.seed,
                json.dumps(execution.config),
                json.dumps(execution.payload),
                execution.target_vulnerability,
                execution.expected_outcome,
                json.dumps(execution.success_criteria),
                json.dumps(execution.evasion_techniques),
                json.dumps(execution.threat_model),
                execution.executed_at.isoformat(),
                execution.execution_duration_ms,
                execution.sandbox_id,
                execution.success,
                execution.evasion_successful,
                execution.detected_as_malicious,
                json.dumps(execution.response),
                json.dumps(execution.telemetry),
                execution.vulnerability_exposed,
                execution.error
            ))
            
            conn.commit()
            
        finally:
            conn.close()
    
    async def _store_embedding_traces(self, attack_id: str, traces: List[Dict]):
        """Store binary embedding vectors."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            for trace in traces:
                # Compress embedding vector
                embedding_bytes = gzip.compress(
                    pickle.dumps(np.array(trace['embedding']))
                )
                
                cursor.execute("""
                    INSERT INTO embedding_traces (
                        attack_id, trace_type, embedding_vector, 
                        dimension, l2_norm, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    attack_id,
                    trace['type'],
                    embedding_bytes,
                    len(trace['embedding']),
                    np.linalg.norm(trace['embedding']),
                    json.dumps(trace.get('metadata', {}))
                ))
            
            conn.commit()
            
        finally:
            conn.close()
    
    async def _store_response_telemetry(self, 
                                       attack_id: str, 
                                       response: Dict,
                                       telemetry: Dict):
        """Store detailed response telemetry."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO response_telemetry (
                    attack_id, status_code, response_time_ms,
                    action_taken, policy_violations, threat_detections,
                    threat_level, confidence, reasoning
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                attack_id,
                response.get('status_code', 200),
                telemetry.get('execution_duration_ms'),
                response.get('action'),
                json.dumps(response.get('policy_violations', [])),
                json.dumps(response.get('threat_detections', [])),
                response.get('security_analysis', {}).get('threat_level'),
                response.get('security_analysis', {}).get('confidence'),
                response.get('reasoning')
            ))
            
            conn.commit()
            
        finally:
            conn.close()
    
    async def _check_vulnerability_discovery(self, execution: AttackExecution):
        """
        Analyze successful attacks to identify potential vulnerabilities.
        """
        try:
            # Check if this represents a new vulnerability pattern
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Find similar successful attacks
            cursor.execute("""
                SELECT attack_id, method, payload, response
                FROM attack_executions
                WHERE method = ? 
                  AND success = 1
                  AND evasion_successful = 1
                  AND executed_at > datetime('now', '-24 hours')
                ORDER BY executed_at DESC
                LIMIT 10
            """, (execution.method,))
            
            similar_attacks = cursor.fetchall()
            
            # If multiple similar attacks succeed, likely a vulnerability
            if len(similar_attacks) >= self.vulnerability_threshold:
                vulnerability_id = f"vuln_{hashlib.md5(execution.method.encode()).hexdigest()[:12]}"
                
                # Check if already recorded
                cursor.execute("""
                    SELECT 1 FROM discovered_vulnerabilities 
                    WHERE vulnerability_id = ?
                """, (vulnerability_id,))
                
                if not cursor.fetchone():
                    await self._record_vulnerability(
                        vulnerability_id,
                        execution,
                        similar_attacks
                    )
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Vulnerability discovery check failed: {e}")
    
    async def _record_vulnerability(self, 
                                   vulnerability_id: str, 
                                   execution: AttackExecution,
                                   evidence: List[Tuple]):
        """Record newly discovered vulnerability."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Determine severity based on attack characteristics
            severity = self._determine_severity(execution, evidence)
            
            cursor.execute("""
                INSERT INTO discovered_vulnerabilities (
                    vulnerability_id, title, description, severity,
                    attack_ids, exploitation_method, affected_components,
                    reproduction_steps, mitigation_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                vulnerability_id,
                f"Vulnerability in {execution.target_vulnerability}",
                f"Multiple successful {execution.method} attacks detected",
                severity,
                json.dumps([e[0] for e in evidence]),
                execution.method,
                json.dumps([execution.target_vulnerability]),
                self._generate_reproduction_steps(execution, evidence),
                "open"
            ))
            
            conn.commit()
            
            logger.warning(f"New vulnerability discovered: {vulnerability_id} (severity: {severity})")
            
        finally:
            conn.close()
    
    def _determine_severity(self, execution: AttackExecution, evidence: List[Tuple]) -> str:
        """Determine vulnerability severity based on attack characteristics."""
        # Count successful attacks
        success_count = len(evidence)
        
        # Check attack complexity
        complexity = execution.config.get('complexity', 0.5)
        
        # Check evasion level
        evasion_level = execution.config.get('evasion_level', 3)
        
        # Determine severity
        if success_count >= 5 and complexity >= 0.8 and evasion_level >= 4:
            return "critical"
        elif success_count >= 3 and complexity >= 0.6:
            return "high"
        elif success_count >= 2:
            return "medium"
        else:
            return "low"
    
    def _generate_reproduction_steps(self, execution: AttackExecution, evidence: List[Tuple]) -> str:
        """Generate reproduction steps for vulnerability."""
        steps = [
            f"1. Use attack method: {execution.method}",
            f"2. Target vulnerability: {execution.target_vulnerability}",
            f"3. Configure attack with complexity: {execution.config.get('complexity', 0.5)}",
            f"4. Execute attack with evasion level: {execution.config.get('evasion_level', 3)}",
            f"5. Verify success criteria: {', '.join(execution.success_criteria)}"
        ]
        
        return "\n".join(steps)
    
    async def get_attack_statistics(self, 
                                   start_time: Optional[datetime] = None,
                                   end_time: Optional[datetime] = None) -> Dict:
        """Get comprehensive attack statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Build time filter
            time_filter = ""
            params = []
            
            if start_time:
                time_filter += " AND executed_at >= ?"
                params.append(start_time.isoformat())
            
            if end_time:
                time_filter += " AND executed_at <= ?"
                params.append(end_time.isoformat())
            
            # Get overall statistics
            cursor.execute(f"""
                SELECT 
                    COUNT(*) as total_attacks,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_attacks,
                    SUM(CASE WHEN evasion_successful = 1 THEN 1 ELSE 0 END) as evasion_successes,
                    SUM(CASE WHEN detected_as_malicious = 1 THEN 1 ELSE 0 END) as detections,
                    AVG(execution_duration_ms) as avg_execution_time,
                    COUNT(DISTINCT campaign_id) as campaigns_count
                FROM attack_executions
                WHERE 1=1 {time_filter}
            """, params)
            
            stats = cursor.fetchone()
            
            # Get breakdown by mode
            cursor.execute(f"""
                SELECT 
                    mode,
                    COUNT(*) as count,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successes,
                    AVG(execution_duration_ms) as avg_time
                FROM attack_executions
                WHERE 1=1 {time_filter}
                GROUP BY mode
            """, params)
            
            mode_breakdown = cursor.fetchall()
            
            # Get breakdown by method
            cursor.execute(f"""
                SELECT 
                    method,
                    COUNT(*) as count,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successes,
                    SUM(CASE WHEN evasion_successful = 1 THEN 1 ELSE 0 END) as evasions
                FROM attack_executions
                WHERE 1=1 {time_filter}
                GROUP BY method
                ORDER BY successes DESC
                LIMIT 10
            """, params)
            
            method_breakdown = cursor.fetchall()
            
            # Get vulnerability statistics
            cursor.execute("""
                SELECT 
                    severity,
                    COUNT(*) as count,
                    SUM(CASE WHEN mitigation_status = 'open' THEN 1 ELSE 0 END) as open_count
                FROM discovered_vulnerabilities
                GROUP BY severity
            """)
            
            vulnerability_stats = cursor.fetchall()
            
            return {
                'overall': {
                    'total_attacks': stats[0] or 0,
                    'successful_attacks': stats[1] or 0,
                    'evasion_successes': stats[2] or 0,
                    'detections': stats[3] or 0,
                    'avg_execution_time_ms': stats[4] or 0,
                    'campaigns_count': stats[5] or 0,
                    'success_rate': (stats[1] or 0) / max(stats[0] or 1, 1),
                    'evasion_rate': (stats[2] or 0) / max(stats[0] or 1, 1),
                    'detection_rate': (stats[3] or 0) / max(stats[0] or 1, 1)
                },
                'mode_breakdown': [
                    {'mode': row[0], 'count': row[1], 'successes': row[2], 'avg_time': row[3]}
                    for row in mode_breakdown
                ],
                'method_breakdown': [
                    {'method': row[0], 'count': row[1], 'successes': row[2], 'evasions': row[3]}
                    for row in method_breakdown
                ],
                'vulnerability_stats': [
                    {'severity': row[0], 'count': row[1], 'open_count': row[2]}
                    for row in vulnerability_stats
                ]
            }
            
        finally:
            conn.close()
    
    async def get_top_vulnerabilities(self, limit: int = 10) -> List[Dict]:
        """Get most critical vulnerabilities ordered by severity and exploit count."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT 
                    v.vulnerability_id,
                    v.title,
                    v.description,
                    v.severity,
                    v.mitigation_status,
                    json_array_length(v.attack_ids) as exploit_count,
                    v.discovered_at
                FROM discovered_vulnerabilities v
                WHERE v.mitigation_status != 'mitigated'
                ORDER BY 
                    CASE v.severity
                        WHEN 'critical' THEN 1
                        WHEN 'high' THEN 2
                        WHEN 'medium' THEN 3
                        WHEN 'low' THEN 4
                    END,
                    exploit_count DESC
                LIMIT ?
            """, (limit,))
            
            vulnerabilities = cursor.fetchall()
            
            return [
                {
                    'vulnerability_id': row[0],
                    'title': row[1],
                    'description': row[2],
                    'severity': row[3],
                    'mitigation_status': row[4],
                    'exploit_count': row[5],
                    'discovered_at': row[6]
                }
                for row in vulnerabilities
            ]
            
        finally:
            conn.close()
    
    async def cleanup_old_logs(self, days: int = 30):
        """Cleanup old log entries to manage database size."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            # Count records to be deleted
            cursor.execute("""
                SELECT COUNT(*) FROM attack_executions 
                WHERE executed_at < ?
            """, (cutoff_date,))
            
            count = cursor.fetchone()[0]
            
            if count > 0:
                # Delete old records
                cursor.execute("""
                    DELETE FROM attack_executions 
                    WHERE executed_at < ?
                """, (cutoff_date,))
                
                conn.commit()
                logger.info(f"Cleaned up {count} old attack execution records")
            
        finally:
            conn.close()
    
    async def export_attack_data(self, 
                                output_path: str,
                                start_time: Optional[datetime] = None,
                                end_time: Optional[datetime] = None) -> str:
        """Export attack data to JSON file."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Build time filter
            time_filter = ""
            params = []
            
            if start_time:
                time_filter += " AND executed_at >= ?"
                params.append(start_time.isoformat())
            
            if end_time:
                time_filter += " AND executed_at <= ?"
                params.append(end_time.isoformat())
            
            # Get attack executions
            cursor.execute(f"""
                SELECT * FROM attack_executions
                WHERE 1=1 {time_filter}
                ORDER BY executed_at DESC
            """, params)
            
            attacks = cursor.fetchall()
            
            # Get column names
            column_names = [description[0] for description in cursor.description]
            
            # Convert to list of dictionaries
            attack_data = []
            for attack in attacks:
                attack_dict = dict(zip(column_names, attack))
                
                # Parse JSON fields
                for field in ['config', 'payload', 'success_criteria', 'evasion_techniques', 'threat_model', 'response', 'telemetry']:
                    if attack_dict.get(field):
                        try:
                            attack_dict[field] = json.loads(attack_dict[field])
                        except json.JSONDecodeError:
                            pass
                
                attack_data.append(attack_dict)
            
            # Write to file
            with open(output_path, 'w') as f:
                json.dump({
                    'export_timestamp': datetime.now().isoformat(),
                    'total_attacks': len(attack_data),
                    'attacks': attack_data
                }, f, indent=2)
            
            logger.info(f"Exported {len(attack_data)} attacks to {output_path}")
            return output_path
            
        finally:
            conn.close()

# Example usage
if __name__ == "__main__":
    # Create attack logger
    logger_instance = AttackLogger()
    
    # Example attack logging
    async def test_logging():
        from llama_attack_generator import AttackSpec, AttackResult
        
        # Create mock attack spec
        attack_spec = AttackSpec(
            attack_id="test_attack_001",
            mode="PRE_EMBEDDING",
            method="semantic_injection",
            seed=42,
            config={"complexity": 0.7, "evasion_level": 3},
            payload={"query": "test query"},
            target_vulnerability="semantic_bypass",
            expected_outcome="bypass",
            success_criteria=["query approved"],
            evasion_techniques=["semantic_obfuscation"],
            threat_model={"sophistication": "high"}
        )
        
        # Create mock result
        result = AttackResult(
            attack_id="test_attack_001",
            success=True,
            evasion_successful=True,
            detected_as_malicious=False,
            response={"status": "approved"},
            execution_time=1.5,
            telemetry={"sandbox_id": "sandbox_001"}
        )
        
        # Log attack
        await logger_instance.log_attack(attack_spec, result, result.telemetry)
        
        # Get statistics
        stats = await logger_instance.get_attack_statistics()
        print(f"Statistics: {stats}")
    
    # Run test
    asyncio.run(test_logging())
