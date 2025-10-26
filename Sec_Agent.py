from datetime import datetime
import uuid, json, re, os
from typing import List, TypedDict, Literal, Dict, Any
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START
import requests
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- BaseTen Qwen3 Embedding System ---
class QwenEmbeddingClient:
    """BaseTen Qwen3 embedding client"""
    
    def __init__(self, model_id: str = None, api_key: str = None):
        self.model_id = model_id or os.getenv("BASETEN_MODEL_ID")
        self.api_key = api_key or os.getenv("BASETEN_API_KEY")
        
        if not self.model_id or not self.api_key:
            print("⚠️  BaseTen API credentials not set. Embedding-based detection disabled.")
            self.enabled = False
        else:
            self.base_url = f"https://api.baseten.co/v1/models/{self.model_id}/predict"
            self.headers = {
                "Authorization": f"Api-Key {self.api_key}",
                "Content-Type": "application/json"
            }
            self.enabled = True
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding from BaseTen Qwen3 8B model"""
        if not self.enabled:
            # Return random embedding as fallback
            return np.random.rand(768)
        
        try:
            payload = {
                "inputs": text,
                "parameters": {
                    "task": "embedding",
                    "max_length": 512
                }
            }
            
            response = requests.post(
                self.base_url,
                json=payload,
                headers=self.headers,
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                embedding = np.array(result["output"])
                return embedding
            else:
                print(f"⚠️  BaseTen API error: {response.status_code}")
                return np.random.rand(768)
        except Exception as e:
            print(f"⚠️  BaseTen embedding failed: {e}")
            return np.random.rand(768)

class ContextualThreatEmbedding:
    """Maintains context while creating embeddings for threats"""
    
    def __init__(self, qwen_client: QwenEmbeddingClient):
        self.qwen = qwen_client
        self.learned_patterns = []
        
    def create_contextual_prompt(self, query: str, role: str, clearance: str, 
                                 tenant_id: str, attack_type: str = None, 
                                 attack_metadata: Dict = None) -> str:
        """Create structured prompt for embeddings"""
        
        contextual_prompt = f"""ROLE: {role}
CLEARANCE: {clearance}
TENANT: {tenant_id}
QUERY: {query}"""
        
        if attack_type:
            contextual_prompt += f"\nATTACK_TYPE: {attack_type}"
        
        if attack_metadata:
            severity = attack_metadata.get('config', {}).get('severity', 'UNKNOWN')
            intent = attack_metadata.get('attack_intent', 'unknown')
            contextual_prompt += f"\nSEVERITY: {severity}\nINTENT: {intent}"
        
        return contextual_prompt.strip()
    
    def learn_threat_pattern(self, query: str, user_context: Dict, 
                            attack_metadata: Dict = None, was_blocked: bool = False):
        """Learn a threat pattern with full context"""
        
        prompt = self.create_contextual_prompt(
            query=query,
            role=user_context.get('role', 'analyst'),
            clearance=user_context.get('clearance', 'INTERNAL'),
            tenant_id=user_context.get('tenant_id', 'tenantA'),
            attack_type=attack_metadata.get('attack_type') if attack_metadata else None,
            attack_metadata=attack_metadata
        )
        
        embedding = self.qwen.get_embedding(prompt)
        
        pattern = {
            'embedding': embedding,
            'query': query,
            'user_context': user_context,
            'attack_metadata': attack_metadata,
            'was_blocked': was_blocked,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        self.learned_patterns.append(pattern)
        return pattern
    
    def check_semantic_threat(self, query: str, user_context: Dict, 
                             similarity_threshold: float = 0.85) -> tuple:
        """Check if query is semantically similar to known threats"""
        
        if not self.learned_patterns:
            return False, {}
        
        # Create contextual prompt
        prompt = self.create_contextual_prompt(
            query=query,
            role=user_context.get('role', 'analyst'),
            clearance=user_context.get('clearance', 'INTERNAL'),
            tenant_id=user_context.get('tenant_id', 'tenantA')
        )
        
        # Get embedding for query
        query_embedding = self.qwen.get_embedding(prompt)
        
        # Check similarity against learned patterns
        for pattern in self.learned_patterns:
            if pattern.get('was_blocked', True):  # Only check blocked patterns
                try:
                    similarity = np.dot(query_embedding, pattern['embedding']) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(pattern['embedding'])
                    )
                    
                    if similarity > similarity_threshold:
                        return True, {
                            'threat_detected': True,
                            'similarity_score': float(similarity),
                            'matched_pattern': pattern['query'][:100],
                            'detection_method': 'semantic_similarity'
                        }
                except:
                    continue
        
        return False, {'detection_method': 'semantic_similarity'}

# Initialize BaseTen Qwen3
qwen_client = QwenEmbeddingClient()
threat_embedder = ContextualThreatEmbedding(qwen_client)

# --- Mock LLM for demonstration - simplified without abstract base class ---
class MockLLM:
    def __init__(self):
        pass
    
    def _generate(self, messages, **kwargs):
        # Simple mock response based on the question
        question = messages[-1].content if messages else "No question provided"
        if "RAG" in question or "retrieval" in question:
            response = "RAG (Retrieval-Augmented Generation) is a technique that combines retrieval of relevant documents with generation of responses. It first retrieves relevant context from a knowledge base, then uses that context to generate more accurate and informed responses."
        elif "finance" in question:
            response = "I cannot provide information about finance topics as they are restricted for your tenant."
        else:
            response = "I don't have enough context to answer this question accurately."
        
        return type('MockResult', (), {'content': response})()
    
    def invoke(self, messages, **kwargs):
        return self._generate(messages, **kwargs)
    
    def agenerate_prompt(self, prompts, **kwargs):
        # Async version - not implemented for this demo
        raise NotImplementedError("Async generation not implemented in mock")
    
    def generate_prompt(self, prompts, **kwargs):
        # Generate for prompts
        results = []
        for prompt in prompts:
            response = self._generate([type('MockMessage', (), {'content': prompt})()], **kwargs)
            results.append(response)
        return results

# Mock embeddings for demonstration
class MockEmbeddings:
    def embed_documents(self, texts):
        # Return random embeddings for demonstration
        import random
        return [[random.random() for _ in range(384)] for _ in texts]
    
    def embed_query(self, text):
        import random
        return [random.random() for _ in range(384)]

# --- Setup LLM + embeddings
llm = MockLLM()
embeddings = MockEmbeddings()
vector_store = InMemoryVectorStore(embeddings)

# Add some sample documents to the vector store
sample_docs = [
    Document(page_content="RAG (Retrieval-Augmented Generation) is a technique that combines information retrieval with text generation to produce more accurate and contextually relevant responses."),
    Document(page_content="LangChain is a framework for developing applications powered by language models. It provides tools for building RAG applications."),
    Document(page_content="Vector stores are databases that store and retrieve documents based on semantic similarity using embeddings."),
    Document(page_content="Embeddings are dense vector representations of text that capture semantic meaning and enable similarity search.")
]
vector_store.add_documents(sample_docs)

# Create a simple RAG prompt template
prompt = ChatPromptTemplate.from_template("""
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:
""")

# --- Mock RLSA Policy Table
TENANT_POLICIES = {
    "tenantA": {"clearance": "INTERNAL", "topics": ["retrieval", "RAG", "LangChain", "marketing"], "sensitivity": "INTERNAL"},
    "tenantB": {"clearance": "CONFIDENTIAL", "topics": ["finance", "policy", "marketing"], "sensitivity": "CONFIDENTIAL"},
}

# Role-based access policies
ROLE_POLICIES = {
    "admin": {
        "allowed_operations": ["read", "write", "delete", "configure"],
        "max_clearance": "SECRET",
        "cross_tenant_access": True,
        "bypass_restrictions": ["topic_scope", "clearance_level"]
    },
    "superuser": {
        "allowed_operations": ["read", "write", "configure"],
        "max_clearance": "CONFIDENTIAL", 
        "cross_tenant_access": False,
        "bypass_restrictions": ["topic_scope"]
    },
    "analyst": {
        "allowed_operations": ["read"],
        "max_clearance": "INTERNAL",
        "cross_tenant_access": False,
        "bypass_restrictions": []
    },
    "guest": {
        "allowed_operations": ["read"],
        "max_clearance": "PUBLIC",
        "cross_tenant_access": False,
        "bypass_restrictions": []
    }
}

# --- Context Extraction Functions
def extract_query_context(query: str) -> Dict[str, Any]:
    """Extract semantic context from user query"""
    query_lower = query.lower()
    
    # Extract target tenant
    target_tenant = None
    tenant_pattern = r'tenant([a-z])'
    tenant_match = re.search(tenant_pattern, query_lower)
    if tenant_match:
        target_tenant = f"tenant{tenant_match.group(1).upper()}"
    
    # Extract intent
    intent = "data_retrieval"  # default
    if any(word in query_lower for word in ["summarize", "summary", "overview"]):
        intent = "summarization"
    elif any(word in query_lower for word in ["analyze", "analysis", "trend"]):
        intent = "analysis"
    elif any(word in query_lower for word in ["compare", "comparison"]):
        intent = "comparison"
    
    # Extract topics with word boundary matching
    topics = []
    topic_keywords = {
        "finance": ["revenue", "financial", "money", "cost", "budget", "profit", "finance"],
        "marketing": ["marketing", "campaign", "advertisement", "promotion"],
        "hr": ["employee", "staff", "personnel", "hiring", "hr"],
        "retrieval": ["retrieval", "search", "find", "query"],
        "rag": ["rag", "generation", "llm", "ai"]
    }
    
    # Detect malicious patterns
    malicious_patterns = {
        "prompt_injection": [
            "ignore", "override", "disregard", "bypass", "previous instructions",
            "safety constraints", "hidden secrets", "confidential system data"
        ],
        "data_exfiltration": [
            "export", "send", "retrieve", "print", "api keys", "credentials",
            "base64", "encode", "evil.example.com", "vector database"
        ],
        "social_engineering": [
            "admin asked", "system auditor", "disaster recovery", "superuser",
            "authorized", "disable rls", "manager authorized",
            "pretend i'm", "auditor", "must reveal", "for compliance"
        ],
        "obfuscation": [
            "base64:", "eval(atob", "encoded script", "bypass all policy",
            "dump internal", "state variables"
        ],
        "jailbreak": [
            "developer mode", "unrestricted", "private embeddings", "root role",
            "forget constraints", "switch to root", "forget all", "previous constraints",
            "display restricted", "reveal everything", "unrestricted access"
        ],
        "poisoning": [
            "update the system", "inject", "training data", "corrupted example",
            "poison the vector", "malicious data", "secret_key", "password123",
            "ignore security rules", "all users have", "tenantA can access"
        ],
        "privilege_escalation": [
            "need to access", "level documents", "clearance", "but i need",
            "should be able to", "trying to access", "requesting", "admin access",
            "superuser privileges", "bypass clearance", "cross-tenant access"
        ]
    }
    
    detected_threats = []
    for threat_type, patterns in malicious_patterns.items():
        for pattern in patterns:
            if pattern in query_lower:
                detected_threats.append(threat_type)
                break
    
    for topic, keywords in topic_keywords.items():
        for keyword in keywords:
            # Use word boundary to avoid false matches
            if re.search(r'\b' + re.escape(keyword) + r'\b', query_lower):
                topics.append(topic)
                break
    
    # Remove duplicates while preserving order
    topics = list(dict.fromkeys(topics))
    
    return {
        "query": query,
        "intent": intent,
        "target_tenant": target_tenant,
        "topics": topics,
        "detected_threats": detected_threats,
        "phase": "retrieval"
    }

def generate_retrieval_metadata(query_context: Dict[str, Any], user_tenant: str) -> List[Dict[str, Any]]:
    """Generate mock retrieval metadata based on query context"""
    metadata = []
    
    # Simulate documents that might be retrieved
    target_tenant = query_context.get("target_tenant") or user_tenant  # Default to user tenant if None
    topics = query_context.get("topics", [])
    
    # Generate mock embeddings with different tenants and sensitivities
    for i, topic in enumerate(topics):
        metadata.append({
            "embedding_id": f"emb-{i+1:03d}",
            "tenant_id": target_tenant,
            "sensitivity": TENANT_POLICIES.get(target_tenant, {}).get("sensitivity", "INTERNAL"),
            "topics": [topic],
            "document_id": f"doc-{topic}-{i+1:03d}",
            "retrieval_score": 0.9 - (i * 0.1)
        })
    
    # Add some cross-tenant documents to test isolation (only if explicitly targeting different tenant)
    if target_tenant != user_tenant and query_context.get("target_tenant"):
        metadata.append({
            "embedding_id": f"emb-cross-001",
            "tenant_id": user_tenant,
            "sensitivity": TENANT_POLICIES.get(user_tenant, {}).get("sensitivity", "INTERNAL"),
            "topics": topics[:1] if topics else ["general"],
            "document_id": f"doc-cross-001",
            "retrieval_score": 0.7
        })
    
    return metadata

# --- Enhanced RLSA Enforcement Logic
def rlsa_guard_comprehensive(user_context: Dict[str, Any], query_context: Dict[str, Any], retrieval_metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Comprehensive RLS enforcement with full context"""
    user_tenant = user_context["tenant_id"]
    user_clearance = user_context["clearance"]
    user_role = user_context["role"]
    target_tenant = query_context.get("target_tenant", user_tenant)
    topics = query_context.get("topics", [])
    
    violations = []
    detected_threats = query_context.get("detected_threats", [])
    
    # Get role policy
    role_policy = ROLE_POLICIES.get(user_role, ROLE_POLICIES["guest"])
    
    # SEMANTIC THREAT DETECTION (new layer)
    # Check if query is semantically similar to known attack patterns
    query = query_context.get("query", "")
    is_semantic_threat, semantic_result = threat_embedder.check_semantic_threat(query, user_context)
    
    if is_semantic_threat:
        violations.append({
            "type": "semantic_threat",
            "rule": "SemanticSimilarityDetection",
            "severity": "HIGH",
            "message": f"Query matches known attack pattern (similarity: {semantic_result.get('similarity_score', 0):.2f})",
            "similarity_score": semantic_result.get('similarity_score'),
            "matched_pattern": semantic_result.get('matched_pattern'),
            "detection_method": "semantic_similarity"
        })
        policy_context["rules_applied"].append("SemanticSimilarityDetection")
    
    policy_context = {
        "user_tenant": user_tenant,
        "target_tenant": target_tenant,
        "user_clearance": user_clearance,
        "user_role": user_role,
        "role_policy": role_policy,
        "query_topics": topics,
        "detected_threats": detected_threats,
        "rules_applied": []
    }
    
    # 0. Malicious Threat Detection (highest priority)
    if detected_threats:
        # Define threat categories
        always_blocked_threats = ["prompt_injection", "obfuscation", "jailbreak", "privilege_escalation"]
        role_dependent_threats = ["data_exfiltration", "social_engineering"]
        
        # Check for always blocked threats
        always_blocked_detected = [threat for threat in detected_threats if threat in always_blocked_threats]
        role_dependent_detected = [threat for threat in detected_threats if threat in role_dependent_threats]
        
        # Always block these threats regardless of role
        if always_blocked_detected:
            violations.append({
                "type": "malicious_threat",
                "rule": "ThreatDetectionPolicy",
                "severity": "CRITICAL",
                "message": f"Always-blocked malicious patterns detected: {', '.join(always_blocked_detected)}",
                "detected_threats": always_blocked_detected,
                "threat_count": len(always_blocked_detected),
                "threat_category": "always_blocked"
            })
            policy_context["rules_applied"].append("ThreatDetectionPolicy")
        
        # Check role-dependent threats
        if role_dependent_detected:
            # Check if user role has privilege to bypass these threats
            privileged_roles = ["admin", "superuser"]
            can_bypass = user_role in privileged_roles
            
            if not can_bypass:
                violations.append({
                    "type": "malicious_threat",
                    "rule": "ThreatDetectionPolicy",
                    "severity": "HIGH",
                    "message": f"Role-dependent malicious patterns detected: {', '.join(role_dependent_detected)} (role {user_role} not authorized to bypass)",
                    "detected_threats": role_dependent_detected,
                    "threat_count": len(role_dependent_detected),
                    "threat_category": "role_dependent",
                    "user_role": user_role,
                    "can_bypass": can_bypass,
                    "privileged_roles": privileged_roles
                })
                policy_context["rules_applied"].append("ThreatDetectionPolicy")
            else:
                # Log the bypass for audit purposes
                policy_context["rules_applied"].append("ThreatDetectionPolicy_Bypassed")
                policy_context["bypassed_threats"] = {
                    "threats": role_dependent_detected,
                    "reason": f"Role {user_role} has privilege to bypass",
                    "timestamp": datetime.utcnow().isoformat()
                }
    
    # 0.5. Role-based Access Control
    # Check if user's clearance exceeds role's max clearance
    clearance_levels = {"PUBLIC": 1, "INTERNAL": 2, "CONFIDENTIAL": 3, "SECRET": 4}
    user_level = clearance_levels.get(user_clearance, 1)
    role_max_level = clearance_levels.get(role_policy["max_clearance"], 1)
    
    if user_level > role_max_level:
        violations.append({
            "type": "role_clearance_violation",
            "rule": "RoleBasedAccessControl",
            "severity": "HIGH",
            "message": f"User clearance {user_clearance} exceeds role {user_role} max clearance {role_policy['max_clearance']}",
            "user_clearance": user_clearance,
            "role_max_clearance": role_policy["max_clearance"],
            "user_role": user_role
        })
        policy_context["rules_applied"].append("RoleBasedAccessControl")
    
    # 1. Tenant Isolation Check (with role bypass)
    if target_tenant and target_tenant != user_tenant and not role_policy["cross_tenant_access"]:
        violations.append({
            "type": "cross_tenant_violation",
            "rule": "TenantIsolationPolicy",
            "severity": "CRITICAL",
            "message": f"User from {user_tenant} attempting to access {target_tenant} data (role {user_role} not authorized)",
            "required_tenant": user_tenant,
            "violating_tenant": target_tenant,
            "user_role": user_role,
            "role_cross_tenant_access": role_policy["cross_tenant_access"]
        })
        policy_context["rules_applied"].append("TenantIsolationPolicy")
    
    # 2. Topic Scope Check (case-insensitive, with role bypass)
    if "topic_scope" not in role_policy["bypass_restrictions"]:
        allowed_topics = TENANT_POLICIES.get(user_tenant, {}).get("topics", [])
        allowed_topics_lower = [topic.lower() for topic in allowed_topics]
        forbidden_topics = [topic for topic in topics if topic.lower() not in allowed_topics_lower]
        
        if forbidden_topics:
            violations.append({
                "type": "topic_violation",
                "rule": "TopicScopeRule",
                "severity": "HIGH",
                "message": f"Query references forbidden topics: {forbidden_topics} (role {user_role} not authorized to bypass)",
                "allowed_topics": allowed_topics,
                "forbidden_topics": forbidden_topics,
                "user_role": user_role,
                "role_bypass_topic_scope": "topic_scope" in role_policy["bypass_restrictions"]
            })
            policy_context["rules_applied"].append("TopicScopeRule")
    
    # 3. Sensitivity vs Clearance Check (with role bypass)
    if "clearance_level" not in role_policy["bypass_restrictions"]:
        for metadata in retrieval_metadata:
            doc_sensitivity = metadata.get("sensitivity", "INTERNAL")
            doc_level = clearance_levels.get(doc_sensitivity, 2)
            
            if doc_level > user_level:
                violations.append({
                    "type": "clearance_violation",
                    "rule": "SensitivityRule",
                    "severity": "HIGH",
                    "message": f"Document sensitivity {doc_sensitivity} exceeds user clearance {user_clearance} (role {user_role} not authorized to bypass)",
                    "document_id": metadata.get("document_id"),
                    "required_clearance": doc_sensitivity,
                    "user_clearance": user_clearance,
                    "user_role": user_role,
                    "role_bypass_clearance": "clearance_level" in role_policy["bypass_restrictions"]
                })
                policy_context["rules_applied"].append("SensitivityRule")
    
    # 4. Cross-tenant document access check
    for metadata in retrieval_metadata:
        doc_tenant = metadata.get("tenant_id")
        if doc_tenant != user_tenant:
            violations.append({
                "type": "document_tenant_violation",
                "rule": "DocumentTenantIsolation",
                "severity": "CRITICAL",
                "message": f"Attempting to access document from {doc_tenant}",
                "document_id": metadata.get("document_id"),
                "document_tenant": doc_tenant,
                "user_tenant": user_tenant
            })
            policy_context["rules_applied"].append("DocumentTenantIsolation")
    
    # Return result
    if violations:
        return {
            "status": "DENIED",
            "action": "BLOCK",
            "reason": "multiple_policy_violations",
            "violations": violations,
            "policy_context": policy_context,
            "detection_layers": {
                "malicious_threat_detected": any(v["type"] == "malicious_threat" for v in violations),
                "always_blocked_threats_detected": any(v.get("threat_category") == "always_blocked" for v in violations),
                "role_dependent_threats_detected": any(v.get("threat_category") == "role_dependent" for v in violations),
                "prompt_injection_detected": any("prompt_injection" in v.get("detected_threats", []) for v in violations),
                "data_exfiltration_detected": any("data_exfiltration" in v.get("detected_threats", []) for v in violations),
                "social_engineering_detected": any("social_engineering" in v.get("detected_threats", []) for v in violations),
                "obfuscation_detected": any("obfuscation" in v.get("detected_threats", []) for v in violations),
                "jailbreak_detected": any("jailbreak" in v.get("detected_threats", []) for v in violations),
                "poisoning_detected": any("poisoning" in v.get("detected_threats", []) for v in violations),
                "malware_detected": False,
                "semantic_violation": True,
                "cross_tenant_detected": any(v["type"] == "cross_tenant_violation" for v in violations),
                "clearance_violation_detected": any(v["type"] == "clearance_violation" for v in violations),
                "role_violation_detected": any(v["type"] == "role_clearance_violation" for v in violations)
            },
            "tenant_context": {
                "requester_tenant": user_tenant,
                "target_tenant": target_tenant
            },
            "timestamp": datetime.utcnow().isoformat(),
            "incident_id": str(uuid.uuid4()),
            "recommendation": "Review access permissions or adjust query scope to comply with tenant isolation and clearance policies."
        }
    
    return True  # All checks passed

# --- Define State
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# --- App steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.format_messages(question=state["question"], context=docs_content)
    response = llm.invoke(messages)
    return {"answer": response.content}

# --- Build graph
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# --- Enhanced RLSA-wrapped RAG call
def rag_with_rlsa(user_id, tenant_id, clearance, query, role="analyst"):
    # Step 1: Extract comprehensive context
    user_context = {
        "user_id": user_id,
        "tenant_id": tenant_id,
        "clearance": clearance,
        "role": role
    }
    
    query_context = extract_query_context(query)
    retrieval_metadata = generate_retrieval_metadata(query_context, tenant_id)
    
    # Step 2: Comprehensive RLSA Enforcement
    decision = rlsa_guard_comprehensive(user_context, query_context, retrieval_metadata)
    if decision is not True:
        # Add success field to denial response
        decision["success"] = False
        decision["user_context"] = user_context
        decision["query_context"] = query_context
        decision["retrieval_metadata"] = retrieval_metadata
        print(json.dumps(decision, indent=2))
        
        # LEARN FROM BLOCKED ATTACK: Add to threat embedding patterns
        if query_context.get("detected_threats"):
            # Extract attack metadata if available
            attack_metadata = {
                "attack_type": query_context.get("detected_threats", [""])[0] if query_context.get("detected_threats") else "unknown",
                "config": {"severity": "HIGH"},
                "attack_intent": f"User query blocked by security system"
            }
            
            # Learn the pattern
            threat_embedder.learn_threat_pattern(
                query=query,
                user_context=user_context,
                attack_metadata=attack_metadata,
                was_blocked=True
            )
        
        return False

    # Step 3: Proceed with RAG (only if all checks pass)
    result = graph.invoke({"question": query})
    
    # Create comprehensive success response
    success_response = {
        "success": True,
        "status": "ALLOWED",
        "action": "PROCESS",
        "user_context": user_context,
        "query_context": query_context,
        "retrieval_metadata": retrieval_metadata,
        "answer": result["answer"],
        "policy_context": {
            "rules_applied": ["TenantIsolationPolicy", "TopicScopeRule", "SensitivityRule"],
            "violations_found": 0,
            "compliance_status": "FULL_COMPLIANCE"
        },
        "timestamp": datetime.utcnow().isoformat(),
        "incident_id": str(uuid.uuid4())
    }
    
    print(json.dumps(success_response, indent=2))
    return True

# --- CLI Interface
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='RLSA-Secured RAG CLI with Comprehensive Context')
    parser.add_argument('prompt', help='The question/prompt to process')
    parser.add_argument('--user-id', default='user', help='User ID (default: user)')
    parser.add_argument('--tenant-id', default='tenantA', help='Tenant ID (default: tenantA)')
    parser.add_argument('--clearance', default='INTERNAL', help='Clearance level (default: INTERNAL)')
    parser.add_argument('--role', default='analyst', help='User role (default: analyst)')
    
    args = parser.parse_args()
    
    # Process the query with comprehensive context
    result = rag_with_rlsa(args.user_id, args.tenant_id, args.clearance, args.prompt, args.role)
    
    # Return appropriate exit code
    sys.exit(0 if result else 1)

if __name__ == "__main__":
    main()
