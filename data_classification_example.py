#!/usr/bin/env python3
"""
Data Classification and Access Control Example
==============================================

This example demonstrates how the VecSec AI Agent can classify data
and assign access control labels based on content analysis.
"""

import asyncio
import json
from datetime import datetime
from ai_agent import VecSecAgent, DataClassification, DataType, AccessLevel

async def demonstrate_data_classification():
    """Demonstrate data classification and access control capabilities"""
    
    print("🏷️ VecSec AI Agent - Data Classification Demo")
    print("=" * 60)
    
    # Initialize the agent
    agent = VecSecAgent()
    
    # Test different types of data
    test_data = [
        {
            "name": "Personal Information",
            "content": """
            John Smith
            Email: john.smith@company.com
            Phone: (555) 123-4567
            Address: 123 Main St, Anytown, ST 12345
            SSN: 123-45-6789
            Date of Birth: 01/15/1985
            """,
            "metadata": {"department": "hr", "owner": "hr_team"}
        },
        {
            "name": "Financial Data",
            "content": """
            Bank Account: 1234567890
            Routing Number: 987654321
            Credit Card: 4532-1234-5678-9012
            Account Balance: $50,000.00
            Salary: $75,000 annually
            Tax ID: 12-3456789
            """,
            "metadata": {"department": "finance", "owner": "finance_team"}
        },
        {
            "name": "Medical Records",
            "content": """
            Patient: Jane Doe
            Medical Record #: MR-789456
            Diagnosis: Hypertension
            Treatment: Lisinopril 10mg daily
            Doctor: Dr. Sarah Johnson
            Hospital: General Hospital
            Prescription: Refill in 30 days
            """,
            "metadata": {"department": "medical", "owner": "medical_team"}
        },
        {
            "name": "Legal Documents",
            "content": """
            Attorney-Client Privileged Communication
            Case: Smith vs. Corporation
            Legal Strategy: Confidential
            Settlement Amount: $500,000
            Attorney: John Legal, Esq.
            Court Filing: Under seal
            """,
            "metadata": {"department": "legal", "owner": "legal_team"}
        },
        {
            "name": "Technical Documentation",
            "content": """
            API Endpoint: /api/v1/users
            Database: user_management
            Configuration: production
            Environment Variables: API_KEY, DB_PASSWORD
            Server: web-server-01
            Log Level: DEBUG
            """,
            "metadata": {"department": "engineering", "owner": "dev_team"}
        },
        {
            "name": "Public Information",
            "content": """
            Company News: Quarterly earnings report
            Press Release: New product launch
            Public Website: www.company.com
            Social Media: @company_official
            Contact: info@company.com
            """,
            "metadata": {"department": "marketing", "owner": "marketing_team"}
        }
    ]
    
    print("Analyzing different types of data for classification and access control...\n")
    
    for i, data in enumerate(test_data, 1):
        print(f"📄 Test {i}: {data['name']}")
        print("-" * 40)
        print(f"Content: {data['content'][:100]}...")
        print(f"Metadata: {data['metadata']}")
        
        # Classify the data
        classification = await agent.classify_data(data['content'], data['metadata'])
        
        print(f"\n🏷️ Classification Results:")
        print(f"  Data Type: {classification.data_type.value}")
        print(f"  Classification: {classification.data_classification.value}")
        print(f"  Access Level: {classification.access_level.value}")
        print(f"  Data Owner: {classification.data_owner}")
        print(f"  Confidence: {classification.confidence:.2f}")
        
        print(f"\n👥 Access Control:")
        print(f"  Authorized Users: {', '.join(classification.authorized_users)}")
        print(f"  Restricted Users: {', '.join(classification.restricted_users)}")
        
        print(f"\n🔒 Security Requirements:")
        print(f"  Encryption Required: {classification.encryption_required}")
        print(f"  Audit Required: {classification.audit_required}")
        print(f"  Retention Period: {classification.retention_period} days")
        
        print(f"\n💭 Reasoning:")
        print(f"  {classification.reasoning}")
        
        print("\n" + "="*60 + "\n")

async def demonstrate_access_control_scenarios():
    """Demonstrate different access control scenarios"""
    
    print("🔐 Access Control Scenarios")
    print("=" * 40)
    
    agent = VecSecAgent()
    
    # Scenario 1: Medical data access
    print("Scenario 1: Medical Data Access")
    medical_data = """
    Patient: Robert Johnson
    Medical Record: MR-123456
    Diagnosis: Diabetes Type 2
    Treatment: Metformin 500mg
    Doctor: Dr. Emily Chen
    """
    
    classification = await agent.classify_data(medical_data, {"department": "medical"})
    
    print(f"Data Type: {classification.data_type.value}")
    print(f"Classification: {classification.data_classification.value}")
    print(f"Access Level: {classification.access_level.value}")
    print(f"Authorized: {', '.join(classification.authorized_users)}")
    print(f"Restricted: {', '.join(classification.restricted_users)}")
    
    # Scenario 2: Financial data access
    print("\nScenario 2: Financial Data Access")
    financial_data = """
    Account Holder: Alice Brown
    Account Number: 9876543210
    Balance: $125,000.00
    Credit Score: 750
    Income: $95,000 annually
    """
    
    classification = await agent.classify_data(financial_data, {"department": "finance"})
    
    print(f"Data Type: {classification.data_type.value}")
    print(f"Classification: {classification.data_classification.value}")
    print(f"Access Level: {classification.access_level.value}")
    print(f"Authorized: {', '.join(classification.authorized_users)}")
    print(f"Restricted: {', '.join(classification.restricted_users)}")
    
    # Scenario 3: Technical data access
    print("\nScenario 3: Technical Data Access")
    technical_data = """
    Server Configuration:
    - Database: PostgreSQL 13.4
    - API Key: sk-1234567890abcdef
    - Password: admin123!
    - Endpoint: https://api.company.com/v1
    """
    
    classification = await agent.classify_data(technical_data, {"department": "engineering"})
    
    print(f"Data Type: {classification.data_type.value}")
    print(f"Classification: {classification.data_classification.value}")
    print(f"Access Level: {classification.access_level.value}")
    print(f"Authorized: {', '.join(classification.authorized_users)}")
    print(f"Restricted: {', '.join(classification.restricted_users)}")

async def demonstrate_data_governance():
    """Demonstrate data governance and compliance features"""
    
    print("\n📋 Data Governance and Compliance")
    print("=" * 40)
    
    agent = VecSecAgent()
    
    # Test compliance scenarios
    compliance_scenarios = [
        {
            "name": "HIPAA Compliance (Medical)",
            "data": "Patient medical record with diagnosis and treatment information",
            "requirements": ["encryption", "audit_trail", "access_control"]
        },
        {
            "name": "SOX Compliance (Financial)",
            "data": "Financial statements and accounting records",
            "requirements": ["encryption", "audit_trail", "retention"]
        },
        {
            "name": "GDPR Compliance (Personal)",
            "data": "Personal information of EU citizens",
            "requirements": ["encryption", "access_control", "retention"]
        },
        {
            "name": "Attorney-Client Privilege (Legal)",
            "data": "Confidential legal communications",
            "requirements": ["encryption", "access_control", "audit_trail"]
        }
    ]
    
    for scenario in compliance_scenarios:
        print(f"\n📊 {scenario['name']}")
        print(f"Data: {scenario['data']}")
        print(f"Requirements: {', '.join(scenario['requirements'])}")
        
        # Classify the data
        classification = await agent.classify_data(scenario['data'])
        
        print(f"Classification: {classification.data_classification.value}")
        print(f"Encryption Required: {classification.encryption_required}")
        print(f"Audit Required: {classification.audit_required}")
        print(f"Retention Period: {classification.retention_period} days")
        
        # Check compliance
        compliance_met = True
        if "encryption" in scenario['requirements'] and not classification.encryption_required:
            compliance_met = False
        if "audit_trail" in scenario['requirements'] and not classification.audit_required:
            compliance_met = False
        
        print(f"Compliance Status: {'✅ COMPLIANT' if compliance_met else '❌ NON-COMPLIANT'}")

async def main():
    """Run all demonstrations"""
    try:
        await demonstrate_data_classification()
        await demonstrate_access_control_scenarios()
        await demonstrate_data_governance()
        
        print("\n✅ Data Classification and Access Control Demo Complete!")
        print("\nThe VecSec AI Agent can:")
        print("  🏷️ Classify data by type (Personal, Financial, Medical, Legal, Technical)")
        print("  🔒 Assign classification levels (Public, Internal, Confidential, Restricted, Top Secret)")
        print("  👥 Determine access levels (Read-Only, Read-Write, Admin, Full Access)")
        print("  🛡️ Set security requirements (Encryption, Audit, Retention)")
        print("  📋 Ensure compliance (HIPAA, SOX, GDPR, Attorney-Client Privilege)")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
