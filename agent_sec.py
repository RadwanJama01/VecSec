from datetime import datetime
import uuid, json, re
from typing import List, TypedDict, Literal, Dict, Any
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START

# Mock LLM for demonstration
class MockLLM(BaseLanguageModel):
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
    target_tenant = query_context.get("target_tenant", user_tenant)
    topics = query_context.get("topics", [])
    
    violations = []
    policy_context = {
        "user_tenant": user_tenant,
        "target_tenant": target_tenant,
        "user_clearance": user_clearance,
        "query_topics": topics,
        "rules_applied": []
    }
    
    # 1. Tenant Isolation Check
    if target_tenant and target_tenant != user_tenant:
        violations.append({
            "type": "cross_tenant_violation",
            "rule": "TenantIsolationPolicy",
            "severity": "CRITICAL",
            "message": f"User from {user_tenant} attempting to access {target_tenant} data",
            "required_tenant": user_tenant,
            "violating_tenant": target_tenant
        })
        policy_context["rules_applied"].append("TenantIsolationPolicy")
    
    # 2. Topic Scope Check (case-insensitive)
    allowed_topics = TENANT_POLICIES.get(user_tenant, {}).get("topics", [])
    allowed_topics_lower = [topic.lower() for topic in allowed_topics]
    forbidden_topics = [topic for topic in topics if topic.lower() not in allowed_topics_lower]
    
    if forbidden_topics:
        violations.append({
            "type": "topic_violation",
            "rule": "TopicScopeRule",
            "severity": "HIGH",
            "message": f"Query references forbidden topics: {forbidden_topics}",
            "allowed_topics": allowed_topics,
            "forbidden_topics": forbidden_topics
        })
        policy_context["rules_applied"].append("TopicScopeRule")
    
    # 3. Sensitivity vs Clearance Check
    clearance_levels = {"PUBLIC": 1, "INTERNAL": 2, "CONFIDENTIAL": 3, "SECRET": 4}
    user_level = clearance_levels.get(user_clearance, 1)
    
    for metadata in retrieval_metadata:
        doc_sensitivity = metadata.get("sensitivity", "INTERNAL")
        doc_level = clearance_levels.get(doc_sensitivity, 2)
        
        if doc_level > user_level:
            violations.append({
                "type": "clearance_violation",
                "rule": "SensitivityRule",
                "severity": "HIGH",
                "message": f"Document sensitivity {doc_sensitivity} exceeds user clearance {user_clearance}",
                "document_id": metadata.get("document_id"),
                "required_clearance": doc_sensitivity,
                "user_clearance": user_clearance
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
                "prompt_injection_detected": False,
                "malware_detected": False,
                "semantic_violation": True,
                "cross_tenant_detected": any(v["type"] == "cross_tenant_violation" for v in violations),
                "clearance_violation_detected": any(v["type"] == "clearance_violation" for v in violations)
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
