"""
AI Agent for VecSec - Cybersecurity Analysis Agent
==================================================

This AI agent acts as a cybersecurity analyst with read, write, and label capabilities.
It can analyze threats, generate reports, and make security decisions autonomously.
"""

import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import requests
from pathlib import Path

from malware_bert import MalwareBERTDetector, ThreatLevel, DetectionResult
from agent_config import config

class AgentCapability(Enum):
    """Agent capabilities"""
    READ = "read"
    WRITE = "write" 
    LABEL = "label"
    ANALYZE = "analyze"
    REPORT = "report"
    DECIDE = "decide"

class IncidentSeverity(Enum):
    """Incident severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AgentAction(Enum):
    """Actions the agent can take"""
    BLOCK = "block"
    ALLOW = "allow"
    QUARANTINE = "quarantine"
    INVESTIGATE = "investigate"
    REPORT = "report"
    ESCALATE = "escalate"

@dataclass
class SecurityIncident:
    """Represents a security incident"""
    id: str
    timestamp: datetime
    source_ip: str
    target_url: str
    threat_level: ThreatLevel
    severity: IncidentSeverity
    indicators: List[str]
    patterns_found: List[str]
    agent_decision: AgentAction
    confidence: float
    risk_score: float
    raw_data: str
    analysis_notes: str = ""

@dataclass
class AgentMemory:
    """Agent's memory for learning and context"""
    incident_history: List[SecurityIncident]
    threat_patterns: Dict[str, List[str]]
    false_positives: List[str]
    false_negatives: List[str]
    learned_rules: List[Dict[str, Any]]
    performance_metrics: Dict[str, float]

class VecSecAgent:
    """
    AI Agent for VecSec Cybersecurity Platform
    
    This agent can:
    - Read and analyze network traffic, files, and logs
    - Write security reports and policies
    - Label and classify threats
    - Make autonomous security decisions
    - Learn from past incidents
    """
    
    def __init__(self, 
                 proxy_url: Optional[str] = None,
                 model_path: Optional[str] = None,
                 memory_file: Optional[str] = None):
        # Use configuration values
        self.proxy_url = proxy_url or config.proxy_url
        self.memory_file = Path(memory_file or config.agent_memory_file)
        self.logger = self._setup_logging()
        
        # Initialize malware detector
        model_path = model_path or config.malware_model_path
        self.malware_detector = MalwareBERTDetector(model_path)
        
        # Initialize agent memory
        self.memory = self._load_memory()
        
        # Agent configuration from config
        self.config = {
            "auto_block_threshold": config.auto_block_threshold,
            "escalation_threshold": config.escalation_threshold,
            "investigation_threshold": config.investigation_threshold,
            "learning_enabled": config.agent_learning_enabled,
            "max_memory_size": config.agent_max_memory_size,
            "report_generation": True,
            "real_time_analysis": True,
            "proxy_timeout": config.proxy_timeout,
            "max_concurrent_analyses": config.max_concurrent_analyses,
            "analysis_timeout": config.analysis_timeout
        }
        
        self.logger.info(f"VecSec AI Agent initialized successfully (mode: {config.agent_mode})")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the agent"""
        logger = logging.getLogger("VecSecAgent")
        
        # Set log level from config
        log_level = getattr(logging, config.log_level.upper(), logging.INFO)
        logger.setLevel(log_level)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler if configured
        if config.log_file and config.log_file != 'agent.log':
            try:
                file_handler = logging.FileHandler(config.log_file)
                file_handler.setFormatter(console_formatter)
                logger.addHandler(file_handler)
            except Exception as e:
                logger.warning(f"Failed to setup file logging: {e}")
        
        return logger
    
    def _load_memory(self) -> AgentMemory:
        """Load agent memory from file"""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                return AgentMemory(**data)
            except Exception as e:
                self.logger.warning(f"Failed to load memory: {e}")
        
        return AgentMemory(
            incident_history=[],
            threat_patterns={},
            false_positives=[],
            false_negatives=[],
            learned_rules=[],
            performance_metrics={}
        )
    
    def _save_memory(self):
        """Save agent memory to file"""
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(asdict(self.memory), f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save memory: {e}")
    
    # READ CAPABILITIES
    async def read_network_traffic(self, duration: int = 60) -> List[Dict[str, Any]]:
        """Read and analyze network traffic"""
        self.logger.info(f"Reading network traffic for {duration} seconds")
        
        # In a real implementation, this would interface with network monitoring tools
        # For now, we'll simulate by analyzing proxy logs
        traffic_data = []
        
        try:
            # Get proxy configuration to understand current setup
            response = requests.get(f"{self.proxy_url}/config", timeout=self.config["proxy_timeout"])
            if response.status_code == 200:
                proxy_config = response.json()
                traffic_data.append({
                    "type": "proxy_config",
                    "data": proxy_config,
                    "timestamp": datetime.now().isoformat()
                })
        except Exception as e:
            self.logger.error(f"Failed to read network traffic: {e}")
        
        return traffic_data
    
    async def read_file_content(self, file_path: str) -> Dict[str, Any]:
        """Read and analyze file content"""
        self.logger.info(f"Reading file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Analyze the file for malware
            analysis = self.malware_detector.detect_malware(content)
            
            return {
                "file_path": file_path,
                "content_length": len(content),
                "analysis": {
                    "threat_level": analysis.threat_level.value,
                    "confidence": analysis.confidence,
                    "risk_score": analysis.risk_score,
                    "indicators": analysis.indicators,
                    "patterns_found": analysis.patterns_found
                },
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Failed to read file {file_path}: {e}")
            return {"error": str(e)}
    
    async def read_system_logs(self, log_path: str = "/var/log") -> List[Dict[str, Any]]:
        """Read and analyze system logs"""
        self.logger.info(f"Reading system logs from {log_path}")
        
        # In a real implementation, this would parse actual log files
        # For now, we'll return a simulated structure
        return [{
            "log_file": "auth.log",
            "entries": [],
            "suspicious_activity": [],
            "timestamp": datetime.now().isoformat()
        }]
    
    # WRITE CAPABILITIES
    async def write_security_report(self, incident: SecurityIncident) -> str:
        """Write a comprehensive security report"""
        self.logger.info(f"Writing security report for incident {incident.id}")
        
        report = {
            "incident_id": incident.id,
            "timestamp": incident.timestamp.isoformat(),
            "summary": {
                "threat_level": incident.threat_level.value,
                "severity": incident.severity.value,
                "confidence": incident.confidence,
                "risk_score": incident.risk_score
            },
            "technical_details": {
                "source_ip": incident.source_ip,
                "target_url": incident.target_url,
                "indicators": incident.indicators,
                "patterns_found": incident.patterns_found
            },
            "agent_analysis": {
                "decision": incident.agent_decision.value,
                "reasoning": incident.analysis_notes,
                "recommended_actions": self._generate_recommendations(incident)
            },
            "recommendations": self._generate_recommendations(incident)
        }
        
        # Save report to file
        report_file = f"security_report_{incident.id}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report_file
    
    async def write_detection_rules(self, patterns: List[str]) -> str:
        """Write new detection rules based on learned patterns"""
        self.logger.info(f"Writing {len(patterns)} new detection rules")
        
        rules = {
            "version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "rules": []
        }
        
        for pattern in patterns:
            rule = {
                "id": f"rule_{len(rules['rules']) + 1}",
                "pattern": pattern,
                "category": self._categorize_pattern(pattern),
                "severity": self._assess_pattern_severity(pattern),
                "created_by": "VecSecAgent",
                "created_at": datetime.now().isoformat()
            }
            rules["rules"].append(rule)
        
        rules_file = f"detection_rules_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(rules_file, 'w') as f:
            json.dump(rules, f, indent=2)
        
        return rules_file
    
    async def write_policy_recommendations(self, incidents: List[SecurityIncident]) -> str:
        """Write security policy recommendations based on incident analysis"""
        self.logger.info(f"Writing policy recommendations based on {len(incidents)} incidents")
        
        # Analyze incidents to identify patterns
        threat_analysis = self._analyze_threat_patterns(incidents)
        
        policy = {
            "version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "analysis_summary": threat_analysis,
            "recommendations": [
                {
                    "category": "Network Security",
                    "priority": "High",
                    "recommendation": "Implement stricter URL filtering",
                    "rationale": f"Detected {threat_analysis['suspicious_urls']} suspicious URLs"
                },
                {
                    "category": "File Security", 
                    "priority": "Medium",
                    "recommendation": "Enhance file scanning for encoded payloads",
                    "rationale": f"Found {threat_analysis['encoded_payloads']} encoded payloads"
                }
            ]
        }
        
        policy_file = f"security_policy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(policy_file, 'w') as f:
            json.dump(policy, f, indent=2)
        
        return policy_file
    
    # LABEL CAPABILITIES
    async def label_threat_level(self, content: str) -> Tuple[ThreatLevel, float, List[str]]:
        """Label content with threat level and confidence"""
        self.logger.info("Labeling threat level for content")
        
        analysis = self.malware_detector.detect_malware(content)
        
        # Update agent memory with this classification
        if self.config["learning_enabled"]:
            self._update_learning_memory(content, analysis)
        
        return analysis.threat_level, analysis.confidence, analysis.indicators
    
    async def label_incident_severity(self, incident: SecurityIncident) -> IncidentSeverity:
        """Label incident with severity level"""
        self.logger.info(f"Labeling severity for incident {incident.id}")
        
        # Determine severity based on threat level and risk score
        if incident.threat_level == ThreatLevel.MALICIOUS and incident.risk_score > 0.8:
            severity = IncidentSeverity.CRITICAL
        elif incident.threat_level == ThreatLevel.MALICIOUS and incident.risk_score > 0.5:
            severity = IncidentSeverity.HIGH
        elif incident.threat_level == ThreatLevel.SUSPICIOUS and incident.risk_score > 0.6:
            severity = IncidentSeverity.MEDIUM
        else:
            severity = IncidentSeverity.LOW
        
        return severity
    
    async def label_attack_type(self, patterns: List[str]) -> List[str]:
        """Label attack types based on detected patterns"""
        self.logger.info("Labeling attack types")
        
        attack_types = []
        
        for pattern in patterns:
            if "rm -rf" in pattern or "del /s" in pattern:
                attack_types.append("Data Destruction")
            elif "curl" in pattern or "wget" in pattern:
                attack_types.append("Data Exfiltration")
            elif "nc -l" in pattern or "netcat" in pattern:
                attack_types.append("Backdoor/Listener")
            elif "eval(" in pattern or "exec(" in pattern:
                attack_types.append("Code Injection")
            elif "base64" in pattern or "hex" in pattern:
                attack_types.append("Obfuscated Payload")
            elif "script" in pattern.lower():
                attack_types.append("Script Injection")
        
        return list(set(attack_types))  # Remove duplicates
    
    # ANALYZE CAPABILITIES
    async def analyze_real_time_traffic(self, target_url: str) -> SecurityIncident:
        """Analyze real-time traffic through the proxy"""
        self.logger.info(f"Analyzing real-time traffic to {target_url}")
        
        # Make a request through the proxy to analyze it
        try:
            response = requests.get(f"{self.proxy_url}/?target={target_url}", timeout=self.config["proxy_timeout"])
            
            # Extract analysis from response headers
            malware_analysis = None
            if 'X-Malware-Analysis' in response.headers:
                malware_analysis = json.loads(response.headers['X-Malware-Analysis'])
            
            # Create incident
            incident_id = f"incident_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Determine threat level from analysis
            if malware_analysis:
                threat_level = ThreatLevel(malware_analysis['threat_level'])
                risk_score = malware_analysis['risk_score']
            else:
                threat_level = ThreatLevel.CLEAN
                risk_score = 0.0
            
            incident = SecurityIncident(
                id=incident_id,
                timestamp=datetime.now(),
                source_ip="unknown",
                target_url=target_url,
                threat_level=threat_level,
                severity=await self.label_incident_severity(
                    SecurityIncident(
                        id=incident_id,
                        timestamp=datetime.now(),
                        source_ip="unknown",
                        target_url=target_url,
                        threat_level=threat_level,
                        severity=IncidentSeverity.LOW,
                        indicators=[],
                        patterns_found=[],
                        agent_decision=AgentAction.ALLOW,
                        confidence=0.0,
                        risk_score=risk_score,
                        raw_data=""
                    )
                ),
                indicators=[],
                patterns_found=[],
                agent_decision=self._make_decision(threat_level, risk_score),
                confidence=malware_analysis['risk_score'] if malware_analysis else 0.0,
                risk_score=risk_score,
                raw_data=response.text[:1000]  # First 1000 chars
            )
            
            # Store in memory
            self.memory.incident_history.append(incident)
            if len(self.memory.incident_history) > self.config["max_memory_size"]:
                self.memory.incident_history = self.memory.incident_history[-self.config["max_memory_size"]:]
            
            return incident
            
        except Exception as e:
            self.logger.error(f"Failed to analyze traffic: {e}")
            raise
    
    # DECISION MAKING
    def _make_decision(self, threat_level: ThreatLevel, risk_score: float) -> AgentAction:
        """Make autonomous security decision"""
        if threat_level == ThreatLevel.MALICIOUS and risk_score > self.config["auto_block_threshold"]:
            return AgentAction.BLOCK
        elif threat_level == ThreatLevel.MALICIOUS and risk_score > self.config["escalation_threshold"]:
            return AgentAction.ESCALATE
        elif threat_level == ThreatLevel.SUSPICIOUS:
            return AgentAction.INVESTIGATE
        else:
            return AgentAction.ALLOW
    
    def _generate_recommendations(self, incident: SecurityIncident) -> List[str]:
        """Generate security recommendations based on incident"""
        recommendations = []
        
        if incident.threat_level == ThreatLevel.MALICIOUS:
            recommendations.extend([
                "Immediately block the source IP",
                "Quarantine any affected systems",
                "Review network logs for similar patterns",
                "Update firewall rules to prevent similar attacks"
            ])
        
        if "curl" in str(incident.patterns_found) or "wget" in str(incident.patterns_found):
            recommendations.append("Implement outbound traffic monitoring")
        
        if "base64" in str(incident.patterns_found):
            recommendations.append("Enhance detection for encoded payloads")
        
        return recommendations
    
    def _categorize_pattern(self, pattern: str) -> str:
        """Categorize a detection pattern"""
        if "rm -rf" in pattern or "del" in pattern:
            return "destructive_command"
        elif "curl" in pattern or "wget" in pattern:
            return "data_exfiltration"
        elif "nc" in pattern or "netcat" in pattern:
            return "backdoor"
        elif "eval" in pattern or "exec" in pattern:
            return "code_injection"
        else:
            return "suspicious_activity"
    
    def _assess_pattern_severity(self, pattern: str) -> str:
        """Assess the severity of a pattern"""
        if any(cmd in pattern for cmd in ["rm -rf", "del /s", "format"]):
            return "critical"
        elif any(cmd in pattern for cmd in ["curl", "wget", "nc -l"]):
            return "high"
        elif any(cmd in pattern for cmd in ["eval", "exec", "system"]):
            return "medium"
        else:
            return "low"
    
    def _analyze_threat_patterns(self, incidents: List[SecurityIncident]) -> Dict[str, Any]:
        """Analyze patterns across multiple incidents"""
        analysis = {
            "total_incidents": len(incidents),
            "malicious_count": len([i for i in incidents if i.threat_level == ThreatLevel.MALICIOUS]),
            "suspicious_count": len([i for i in incidents if i.threat_level == ThreatLevel.SUSPICIOUS]),
            "suspicious_urls": 0,
            "encoded_payloads": 0,
            "shell_commands": 0,
            "common_patterns": []
        }
        
        for incident in incidents:
            for pattern in incident.patterns_found:
                if "url" in pattern.lower():
                    analysis["suspicious_urls"] += 1
                elif "base64" in pattern or "hex" in pattern:
                    analysis["encoded_payloads"] += 1
                elif any(cmd in pattern for cmd in ["curl", "wget", "rm", "nc"]):
                    analysis["shell_commands"] += 1
        
        return analysis
    
    def _update_learning_memory(self, content: str, analysis: DetectionResult):
        """Update agent's learning memory"""
        # Store patterns for learning
        if analysis.patterns_found:
            for pattern in analysis.patterns_found:
                pattern_type = self._categorize_pattern(pattern)
                if pattern_type not in self.memory.threat_patterns:
                    self.memory.threat_patterns[pattern_type] = []
                self.memory.threat_patterns[pattern_type].append(pattern)
        
        # Save memory
        self._save_memory()
    
    # MAIN AGENT INTERFACE
    async def run_autonomous_analysis(self, target_url: str) -> Dict[str, Any]:
        """Run autonomous analysis on a target URL"""
        self.logger.info(f"Starting autonomous analysis of {target_url}")
        
        try:
            # Analyze the traffic
            incident = await self.analyze_real_time_traffic(target_url)
            
            # Generate report
            report_file = await self.write_security_report(incident)
            
            # Make decision
            decision = self._make_decision(incident.threat_level, incident.risk_score)
            
            result = {
                "incident": asdict(incident),
                "decision": decision.value,
                "report_file": report_file,
                "recommendations": self._generate_recommendations(incident),
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"Analysis complete. Decision: {decision.value}")
            return result
            
        except Exception as e:
            self.logger.error(f"Autonomous analysis failed: {e}")
            return {"error": str(e)}
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status and capabilities"""
        return {
            "capabilities": [cap.value for cap in AgentCapability],
            "memory_size": len(self.memory.incident_history),
            "learned_patterns": len(self.memory.threat_patterns),
            "performance_metrics": self.memory.performance_metrics,
            "config": self.config,
            "status": "active"
        }

# Example usage and testing
async def main():
    """Example usage of the VecSec AI Agent"""
    agent = VecSecAgent()
    
    # Test the agent's capabilities
    print("VecSec AI Agent Demo")
    print("=" * 50)
    
    # Get agent status
    status = await agent.get_agent_status()
    print(f"Agent Status: {json.dumps(status, indent=2)}")
    
    # Test autonomous analysis
    test_url = "https://httpbin.org/get"
    print(f"\nAnalyzing: {test_url}")
    
    result = await agent.run_autonomous_analysis(test_url)
    print(f"Analysis Result: {json.dumps(result, indent=2, default=str)}")

if __name__ == "__main__":
    asyncio.run(main())
