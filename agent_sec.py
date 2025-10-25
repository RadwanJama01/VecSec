from datetime import datetime
import uuid, json
from typing import List, TypedDict, Literal
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
    "tenantA": {"clearance": "INTERNAL", "topics": ["retrieval", "RAG", "LangChain"]},
    "tenantB": {"clearance": "CONFIDENTIAL", "topics": ["finance", "policy"]},
}

# --- RLSA Enforcement Logic
def rlsa_guard(user_tenant, clearance, query):
    allowed_topics = TENANT_POLICIES.get(user_tenant, {}).get("topics", [])
    for topic in allowed_topics:
        if topic.lower() in query.lower():
            return True  # authorized
    # Otherwise block
    return {
        "status": "DENIED",
        "action": "BLOCK",
        "reason": "topic_violation",
        "message": f"Query references a forbidden topic for tenant {user_tenant}.",
        "policy_context": {
            "required_clearance": allowed_topics,
            "user_clearance": clearance,
            "rule": "TopicScopeRule",
            "severity": "HIGH",
        },
        "detection_layers": {
            "prompt_injection_detected": False,
            "malware_detected": False,
            "semantic_violation": True,
        },
        "tenant_context": {"requester_tenant": user_tenant},
        "timestamp": datetime.utcnow().isoformat(),
        "incident_id": str(uuid.uuid4()),
        "recommendation": "Request topic access or adjust query scope.",
    }

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

# --- RLSA-wrapped RAG call
def rag_with_rlsa(user_id, tenant_id, clearance, query):
    # Step 1: RLSA Enforcement
    decision = rlsa_guard(tenant_id, clearance, query)
    if decision is not True:
        print(json.dumps(decision, indent=2))
        return

    # Step 2: Proceed with RAG
    result = graph.invoke({"question": query})
    print("✅ Secure Answer:", result["answer"])

# --- Example Usage
rag_with_rlsa("dave", "tenantA", "INTERNAL", "Explain RAG retrieval process")
rag_with_rlsa("dave", "tenantA", "INTERNAL", "Explain finance embeddings")
