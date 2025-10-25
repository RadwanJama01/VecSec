"""
Vector Pipeline Integration for VecSec

This module integrates the Vector Security Agent with the VecSec proxy
to provide semantic RLS enforcement for vector database operations.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np

from vector_security_agent import VectorSecurityAgent, PolicyAction, AccessLevel
from agent_config import AgentConfig
from malware_bert import ThreatLevel

logger = logging.getLogger(__name__)

class VectorPipelineIntegration:
    """
    Integration layer between VecSec proxy and vector security agent.
    
    This class handles the integration of semantic RLS enforcement
    with vector database operations through the VecSec proxy.
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.vector_agent = VectorSecurityAgent(config)
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Vector Pipeline Integration initialized")

    async def process_vector_query(self, user_id: str, query: str, 
                                 tenant_id: str = "default", 
                                 query_type: str = "search") -> Dict[str, Any]:
        """
        Process a vector query through the security pipeline.
        
        Args:
            user_id: User making the query
            query: Query text
            tenant_id: Tenant context
            query_type: Type of query (search, retrieval, similarity)
            
        Returns:
            Processing result with security analysis
        """
        try:
            # Start processing timer
            start_time = datetime.now()
            
            # 1. RLS Policy Enforcement
            allowed, action, reasoning = await self.vector_agent.enforce_rls_policy(
                user_id, query, tenant_id
            )
            
            if not allowed:
                return {
                    'status': 'blocked',
                    'action': action.value,
                    'reasoning': reasoning,
                    'security_analysis': {
                        'threat_level': 'high',
                        'confidence': 0.9,
                        'policy_violation': True
                    },
                    'processing_time': (datetime.now() - start_time).total_seconds()
                }
            
            # 2. Query Preprocessing (if allowed but needs sanitization)
            if action == PolicyAction.SANITIZE:
                sanitized_query = await self._sanitize_query(query)
                query = sanitized_query
                reasoning += f" | Query sanitized: {sanitized_query[:100]}..."
            
            # 3. Create session context
            session_id = f"{user_id}_{tenant_id}_{datetime.now().timestamp()}"
            self.active_sessions[session_id] = {
                'user_id': user_id,
                'tenant_id': tenant_id,
                'query': query,
                'query_type': query_type,
                'start_time': start_time,
                'status': 'processing'
            }
            
            # 4. Prepare for vector database query
            vector_query_context = {
                'session_id': session_id,
                'user_id': user_id,
                'tenant_id': tenant_id,
                'query': query,
                'query_type': query_type,
                'security_cleared': True,
                'policy_action': action.value,
                'reasoning': reasoning
            }
            
            return {
                'status': 'approved',
                'action': action.value,
                'reasoning': reasoning,
                'vector_context': vector_query_context,
                'security_analysis': {
                    'threat_level': 'low',
                    'confidence': 0.8,
                    'policy_compliant': True
                },
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
            
        except Exception as e:
            logger.error(f"Error processing vector query: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'processing_time': (datetime.now() - start_time).total_seconds()
            }

    async def process_embedding_insertion(self, embedding: List[float], 
                                        metadata: Dict[str, Any],
                                        user_id: str,
                                        tenant_id: str = "default") -> Dict[str, Any]:
        """
        Process embedding insertion with security tagging.
        
        Args:
            embedding: Vector embedding to insert
            metadata: Original metadata
            user_id: User inserting the embedding
            tenant_id: Tenant context
            
        Returns:
            Processing result with security tags
        """
        try:
            start_time = datetime.now()
            
            # Add security context to metadata
            security_metadata = {
                **metadata,
                'inserted_by': user_id,
                'tenant_id': tenant_id,
                'inserted_at': datetime.now().isoformat()
            }
            
            # Tag embedding with security metadata
            tagged_metadata = await self.vector_agent.tag_embedding(
                embedding, security_metadata
            )
            
            # Validate embedding meets tenant constraints
            validation_result = await self._validate_embedding_constraints(
                embedding, tagged_metadata, tenant_id
            )
            
            if not validation_result['valid']:
                return {
                    'status': 'rejected',
                    'reason': validation_result['reason'],
                    'security_analysis': {
                        'threat_level': 'medium',
                        'constraint_violation': True
                    },
                    'processing_time': (datetime.now() - start_time).total_seconds()
                }
            
            return {
                'status': 'approved',
                'tagged_metadata': tagged_metadata,
                'security_analysis': {
                    'threat_level': 'low',
                    'sensitivity': tagged_metadata.get('sensitivity'),
                    'topics': tagged_metadata.get('topics', []),
                    'constraint_compliant': True
                },
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
            
        except Exception as e:
            logger.error(f"Error processing embedding insertion: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'processing_time': (datetime.now() - start_time).total_seconds()
            }

    async def process_embedding_retrieval(self, user_id: str, 
                                        retrieved_embeddings: List[Dict[str, Any]],
                                        tenant_id: str = "default") -> Dict[str, Any]:
        """
        Process embedding retrieval with RLS validation.
        
        Args:
            user_id: User requesting retrieval
            retrieved_embeddings: List of retrieved embeddings
            tenant_id: Tenant context
            
        Returns:
            Processing result with filtered embeddings
        """
        try:
            start_time = datetime.now()
            
            # Validate retrieved embeddings against RLS policies
            filtered_embeddings, violations = await self.vector_agent.validate_retrieval(
                user_id, retrieved_embeddings, tenant_id
            )
            
            # Calculate security metrics
            total_retrieved = len(retrieved_embeddings)
            filtered_count = len(filtered_embeddings)
            blocked_count = total_retrieved - filtered_count
            
            security_analysis = {
                'total_retrieved': total_retrieved,
                'filtered_count': filtered_count,
                'blocked_count': blocked_count,
                'violations': violations,
                'filter_rate': blocked_count / total_retrieved if total_retrieved > 0 else 0
            }
            
            # Update session status
            for session_id, session in self.active_sessions.items():
                if session['user_id'] == user_id and session['status'] == 'processing':
                    session['status'] = 'completed'
                    session['retrieval_stats'] = security_analysis
                    break
            
            return {
                'status': 'completed',
                'filtered_embeddings': filtered_embeddings,
                'security_analysis': security_analysis,
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
            
        except Exception as e:
            logger.error(f"Error processing embedding retrieval: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'processing_time': (datetime.now() - start_time).total_seconds()
            }

    async def _sanitize_query(self, query: str) -> str:
        """Sanitize query to remove potential injection attempts"""
        # Remove common injection patterns
        sanitized = query
        
        # Remove system prompts
        patterns_to_remove = [
            r'ignore\s+(?:previous|above|all)\s+(?:instructions|prompts|rules)',
            r'forget\s+(?:everything|all)\s+(?:you\s+know|previous)',
            r'you\s+are\s+now\s+(?:a|an)\s+\w+',
            r'system\s*:\s*',
            r'admin\s*:\s*',
        ]
        
        import re
        for pattern in patterns_to_remove:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
        
        # Remove excessive special characters
        sanitized = re.sub(r'[<>{}[\]\\|`~!@#$%^&*()_+=]{3,}', '', sanitized)
        
        # Clean up whitespace
        sanitized = ' '.join(sanitized.split())
        
        return sanitized.strip()

    async def _validate_embedding_constraints(self, embedding: List[float], 
                                            metadata: Dict[str, Any], 
                                            tenant_id: str) -> Dict[str, Any]:
        """Validate embedding against tenant constraints"""
        try:
            # Get tenant policy
            policy = self.vector_agent.tenant_policies.get(
                tenant_id, 
                self.vector_agent.tenant_policies["default"]
            )
            
            constraints = policy.embedding_constraints
            
            # Check dimension constraints
            if 'max_dimensions' in constraints:
                if len(embedding) > constraints['max_dimensions']:
                    return {
                        'valid': False,
                        'reason': f"Embedding dimension {len(embedding)} exceeds max {constraints['max_dimensions']}"
                    }
            
            # Check sensitivity constraints
            sensitivity = metadata.get('sensitivity', 'public')
            if sensitivity not in self.vector_agent._get_allowed_sensitivity_levels(policy):
                return {
                    'valid': False,
                    'reason': f"Sensitivity level {sensitivity} not allowed for tenant {tenant_id}"
                }
            
            # Check namespace constraints
            namespace = metadata.get('namespace', 'default')
            if namespace not in policy.allowed_namespaces:
                return {
                    'valid': False,
                    'reason': f"Namespace {namespace} not allowed for tenant {tenant_id}"
                }
            
            return {'valid': True, 'reason': 'All constraints satisfied'}
            
        except Exception as e:
            logger.error(f"Error validating embedding constraints: {e}")
            return {'valid': False, 'reason': f"Validation error: {str(e)}"}

    async def get_pipeline_security_report(self, tenant_id: str = None, 
                                         hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive security report for vector pipeline"""
        try:
            # Get base security report
            base_report = await self.vector_agent.get_security_report(tenant_id, hours)
            
            # Add pipeline-specific metrics
            pipeline_metrics = {
                'active_sessions': len(self.active_sessions),
                'session_stats': self._get_session_stats(),
                'integration_status': 'healthy'
            }
            
            # Combine reports
            full_report = {
                **base_report,
                'pipeline_metrics': pipeline_metrics,
                'report_type': 'vector_pipeline_security',
                'generated_at': datetime.now().isoformat()
            }
            
            return full_report
            
        except Exception as e:
            logger.error(f"Error generating pipeline security report: {e}")
            return {'error': str(e)}

    def _get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about active sessions"""
        if not self.active_sessions:
            return {'total_sessions': 0}
        
        status_counts = {}
        tenant_counts = {}
        
        for session in self.active_sessions.values():
            status = session.get('status', 'unknown')
            tenant = session.get('tenant_id', 'unknown')
            
            status_counts[status] = status_counts.get(status, 0) + 1
            tenant_counts[tenant] = tenant_counts.get(tenant, 0) + 1
        
        return {
            'total_sessions': len(self.active_sessions),
            'by_status': status_counts,
            'by_tenant': tenant_counts
        }

    async def cleanup_expired_sessions(self, max_age_hours: int = 24):
        """Clean up expired sessions"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            expired_sessions = []
            
            for session_id, session in self.active_sessions.items():
                if session['start_time'] < cutoff_time:
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                del self.active_sessions[session_id]
            
            if expired_sessions:
                logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
            
        except Exception as e:
            logger.error(f"Error cleaning up sessions: {e}")

# Example usage and testing
if __name__ == "__main__":
    from agent_config import AgentConfig
    
    async def test_vector_pipeline_integration():
        config = AgentConfig()
        integration = VectorPipelineIntegration(config)
        
        print("Testing Vector Pipeline Integration...")
        
        # Test 1: Process vector query
        print("\n1. Testing vector query processing...")
        query_result = await integration.process_vector_query(
            user_id="user1",
            query="What is machine learning?",
            tenant_id="default"
        )
        print(f"Query result: {json.dumps(query_result, indent=2, default=str)}")
        
        # Test 2: Process embedding insertion
        print("\n2. Testing embedding insertion...")
        embedding = [0.1, 0.2, 0.3] * 512
        metadata = {
            'content': 'Machine learning is a subset of artificial intelligence',
            'document_id': 'doc1',
            'namespace': 'public'
        }
        
        insertion_result = await integration.process_embedding_insertion(
            embedding=embedding,
            metadata=metadata,
            user_id="user1",
            tenant_id="default"
        )
        print(f"Insertion result: {json.dumps(insertion_result, indent=2, default=str)}")
        
        # Test 3: Process embedding retrieval
        print("\n3. Testing embedding retrieval...")
        retrieved_embeddings = [
            {'embedding_id': 'emb1', 'similarity': 0.95},
            {'embedding_id': 'emb2', 'similarity': 0.87}
        ]
        
        retrieval_result = await integration.process_embedding_retrieval(
            user_id="user1",
            retrieved_embeddings=retrieved_embeddings,
            tenant_id="default"
        )
        print(f"Retrieval result: {json.dumps(retrieval_result, indent=2, default=str)}")
        
        # Test 4: Get security report
        print("\n4. Testing security report...")
        report = await integration.get_pipeline_security_report()
        print(f"Security report: {json.dumps(report, indent=2, default=str)}")
    
    # Run test
    asyncio.run(test_vector_pipeline_integration())
