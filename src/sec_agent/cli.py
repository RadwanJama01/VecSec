"""
CLI Interface for VecSec Security Agent
"""

import sys
import argparse
from .rag_orchestrator import RAGOrchestrator
from .embeddings_client import EmbeddingClient
from .threat_detector import ContextualThreatEmbedding
from .mock_llm import MockLLM, MockEmbeddings
from .config import (
    initialize_vector_store,
    initialize_sample_documents,
    create_rag_prompt_template
)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description='RLSA-Secured RAG CLI with Comprehensive Context')
    parser.add_argument('prompt', help='The question/prompt to process')
    parser.add_argument('--user-id', default='user', help='User ID (default: user)')
    parser.add_argument('--tenant-id', default='tenantA', help='Tenant ID (default: tenantA)')
    parser.add_argument('--clearance', default='INTERNAL', help='Clearance level (default: INTERNAL)')
    parser.add_argument('--role', default='analyst', help='User role (default: analyst)')
    
    args = parser.parse_args()
    
    # Initialize components
    embeddings = MockEmbeddings()
    vector_store = initialize_vector_store(embeddings)
    initialize_sample_documents(vector_store)
    llm = MockLLM()
    prompt_template = create_rag_prompt_template()
    
    # Initialize embedding and threat detection
    qwen_client = EmbeddingClient()
    threat_embedder = ContextualThreatEmbedding(qwen_client)
    
    # Create orchestrator
    orchestrator = RAGOrchestrator(
        vector_store=vector_store,
        llm=llm,
        prompt_template=prompt_template,
        threat_embedder=threat_embedder,
        qwen_client=qwen_client
    )
    
    # Process the query with comprehensive context
    result = orchestrator.rag_with_rlsa(
        args.user_id, 
        args.tenant_id, 
        args.clearance, 
        args.prompt, 
        args.role
    )
    
    # Return appropriate exit code
    # result=True means allowed (successful query), result=False means blocked (denied)
    # Exit code 0 = success (allowed), exit code 1 = failure (blocked)
    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()

