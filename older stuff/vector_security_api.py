"""
Vector Security API Endpoints for VecSec

This module provides REST API endpoints for vector security operations
integrated with the VecSec proxy.
"""

from flask import Blueprint, request, jsonify, current_app
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List

from vector_pipeline_integration import VectorPipelineIntegration
from agent_config import AgentConfig

logger = logging.getLogger(__name__)

# Create blueprint for vector security endpoints
vector_security_bp = Blueprint('vector_security', __name__, url_prefix='/api/vector')

# Global integration instance (will be initialized in app.py)
vector_integration: VectorPipelineIntegration = None

def init_vector_security_api(app):
    """Initialize the vector security API with the Flask app"""
    global vector_integration
    
    try:
        config = AgentConfig()
        vector_integration = VectorPipelineIntegration(config)
        app.register_blueprint(vector_security_bp)
        logger.info("Vector Security API initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Vector Security API: {e}")

@vector_security_bp.route('/query', methods=['POST'])
def process_vector_query():
    """Process a vector query through the security pipeline"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        user_id = data.get('user_id')
        query = data.get('query')
        tenant_id = data.get('tenant_id', 'default')
        query_type = data.get('query_type', 'search')
        
        if not user_id or not query:
            return jsonify({'error': 'user_id and query are required'}), 400
        
        # Process query asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                vector_integration.process_vector_query(
                    user_id=user_id,
                    query=query,
                    tenant_id=tenant_id,
                    query_type=query_type
                )
            )
            return jsonify(result)
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Error processing vector query: {e}")
        return jsonify({'error': str(e)}), 500

@vector_security_bp.route('/embedding/insert', methods=['POST'])
def insert_embedding():
    """Insert an embedding with security tagging"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        embedding = data.get('embedding')
        metadata = data.get('metadata', {})
        user_id = data.get('user_id')
        tenant_id = data.get('tenant_id', 'default')
        
        if not embedding or not user_id:
            return jsonify({'error': 'embedding and user_id are required'}), 400
        
        # Process embedding insertion asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                vector_integration.process_embedding_insertion(
                    embedding=embedding,
                    metadata=metadata,
                    user_id=user_id,
                    tenant_id=tenant_id
                )
            )
            return jsonify(result)
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Error inserting embedding: {e}")
        return jsonify({'error': str(e)}), 500

@vector_security_bp.route('/embedding/retrieve', methods=['POST'])
def retrieve_embeddings():
    """Retrieve embeddings with RLS validation"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        retrieved_embeddings = data.get('retrieved_embeddings', [])
        user_id = data.get('user_id')
        tenant_id = data.get('tenant_id', 'default')
        
        if not user_id:
            return jsonify({'error': 'user_id is required'}), 400
        
        # Process embedding retrieval asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                vector_integration.process_embedding_retrieval(
                    user_id=user_id,
                    retrieved_embeddings=retrieved_embeddings,
                    tenant_id=tenant_id
                )
            )
            return jsonify(result)
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Error retrieving embeddings: {e}")
        return jsonify({'error': str(e)}), 500

@vector_security_bp.route('/policy/create', methods=['POST'])
def create_tenant_policy():
    """Create a new tenant policy"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        tenant_id = data.get('tenant_id')
        policy_data = data.get('policy_data', {})
        
        if not tenant_id:
            return jsonify({'error': 'tenant_id is required'}), 400
        
        # Create policy asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            success = loop.run_until_complete(
                vector_integration.vector_agent.create_tenant_policy(
                    tenant_id=tenant_id,
                    policy_data=policy_data
                )
            )
            
            if success:
                return jsonify({'status': 'success', 'message': f'Policy created for tenant {tenant_id}'})
            else:
                return jsonify({'error': 'Failed to create policy'}), 500
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Error creating tenant policy: {e}")
        return jsonify({'error': str(e)}), 500

@vector_security_bp.route('/policy/<tenant_id>', methods=['GET'])
def get_tenant_policy(tenant_id):
    """Get tenant policy information"""
    try:
        if tenant_id not in vector_integration.vector_agent.tenant_policies:
            return jsonify({'error': 'Tenant policy not found'}), 404
        
        policy = vector_integration.vector_agent.tenant_policies[tenant_id]
        
        # Convert policy to JSON-serializable format
        policy_dict = {
            'tenant_id': policy.tenant_id,
            'allowed_topics': policy.allowed_topics,
            'blocked_topics': policy.blocked_topics,
            'sensitivity_level': policy.sensitivity_level.value,
            'max_query_complexity': policy.max_query_complexity,
            'allowed_namespaces': policy.allowed_namespaces,
            'embedding_constraints': policy.embedding_constraints,
            'created_at': policy.created_at.isoformat(),
            'updated_at': policy.updated_at.isoformat()
        }
        
        return jsonify(policy_dict)
        
    except Exception as e:
        logger.error(f"Error getting tenant policy: {e}")
        return jsonify({'error': str(e)}), 500

@vector_security_bp.route('/policy/<tenant_id>', methods=['PUT'])
def update_tenant_policy(tenant_id):
    """Update tenant policy"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        if tenant_id not in vector_integration.vector_agent.tenant_policies:
            return jsonify({'error': 'Tenant policy not found'}), 404
        
        policy = vector_integration.vector_agent.tenant_policies[tenant_id]
        
        # Update policy fields
        if 'allowed_topics' in data:
            policy.allowed_topics = data['allowed_topics']
        if 'blocked_topics' in data:
            policy.blocked_topics = data['blocked_topics']
        if 'sensitivity_level' in data:
            from vector_security_agent import AccessLevel
            policy.sensitivity_level = AccessLevel(data['sensitivity_level'])
        if 'max_query_complexity' in data:
            policy.max_query_complexity = data['max_query_complexity']
        if 'allowed_namespaces' in data:
            policy.allowed_namespaces = data['allowed_namespaces']
        if 'embedding_constraints' in data:
            policy.embedding_constraints = data['embedding_constraints']
        
        policy.updated_at = datetime.now()
        
        return jsonify({'status': 'success', 'message': f'Policy updated for tenant {tenant_id}'})
        
    except Exception as e:
        logger.error(f"Error updating tenant policy: {e}")
        return jsonify({'error': str(e)}), 500

@vector_security_bp.route('/security/report', methods=['GET'])
def get_security_report():
    """Get vector pipeline security report"""
    try:
        tenant_id = request.args.get('tenant_id')
        hours = int(request.args.get('hours', 24))
        
        # Get security report asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            report = loop.run_until_complete(
                vector_integration.get_pipeline_security_report(
                    tenant_id=tenant_id,
                    hours=hours
                )
            )
            return jsonify(report)
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Error getting security report: {e}")
        return jsonify({'error': str(e)}), 500

@vector_security_bp.route('/security/incidents', methods=['GET'])
def get_security_incidents():
    """Get recent security incidents"""
    try:
        tenant_id = request.args.get('tenant_id')
        limit = int(request.args.get('limit', 100))
        hours = int(request.args.get('hours', 24))
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter incidents
        if tenant_id:
            incidents = [
                incident for incident in vector_integration.vector_agent.incident_log
                if incident.tenant_id == tenant_id and incident.timestamp > cutoff_time
            ]
        else:
            incidents = [
                incident for incident in vector_integration.vector_agent.incident_log
                if incident.timestamp > cutoff_time
            ]
        
        # Sort by timestamp (newest first) and limit
        incidents.sort(key=lambda x: x.timestamp, reverse=True)
        incidents = incidents[:limit]
        
        # Convert to JSON-serializable format
        incidents_data = []
        for incident in incidents:
            incident_dict = {
                'incident_id': incident.incident_id,
                'user_id': incident.user_id,
                'tenant_id': incident.tenant_id,
                'query': incident.query,
                'action_taken': incident.action_taken.value,
                'threat_level': incident.threat_level.value,
                'confidence': incident.confidence,
                'reasoning': incident.reasoning,
                'metadata': incident.metadata,
                'timestamp': incident.timestamp.isoformat()
            }
            incidents_data.append(incident_dict)
        
        return jsonify({
            'incidents': incidents_data,
            'total_count': len(incidents_data),
            'time_range': {
                'start': cutoff_time.isoformat(),
                'end': datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting security incidents: {e}")
        return jsonify({'error': str(e)}), 500

@vector_security_bp.route('/sessions', methods=['GET'])
def get_active_sessions():
    """Get active session information"""
    try:
        tenant_id = request.args.get('tenant_id')
        
        sessions = vector_integration.active_sessions
        
        if tenant_id:
            sessions = {
                sid: session for sid, session in sessions.items()
                if session.get('tenant_id') == tenant_id
            }
        
        # Convert to JSON-serializable format
        sessions_data = []
        for session_id, session in sessions.items():
            session_dict = {
                'session_id': session_id,
                'user_id': session.get('user_id'),
                'tenant_id': session.get('tenant_id'),
                'query': session.get('query'),
                'query_type': session.get('query_type'),
                'status': session.get('status'),
                'start_time': session.get('start_time').isoformat() if session.get('start_time') else None,
                'retrieval_stats': session.get('retrieval_stats')
            }
            sessions_data.append(session_dict)
        
        return jsonify({
            'sessions': sessions_data,
            'total_count': len(sessions_data)
        })
        
    except Exception as e:
        logger.error(f"Error getting active sessions: {e}")
        return jsonify({'error': str(e)}), 500

@vector_security_bp.route('/threats/classify', methods=['POST'])
def classify_threats():
    """Classify threats in a query"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        query = data.get('query')
        user_id = data.get('user_id', 'unknown')
        tenant_id = data.get('tenant_id', 'default')
        
        if not query:
            return jsonify({'error': 'query is required'}), 400
        
        # Classify threats
        threat_detections = vector_integration.vector_agent.threat_classifier.classify_threat(
            query, {'user_id': user_id, 'tenant_id': tenant_id}
        )
        
        # Convert to JSON-serializable format
        detections_data = []
        for detection in threat_detections:
            detection_dict = {
                'threat_class': detection.threat_class.value,
                'attack_vector': detection.attack_vector.value,
                'confidence': detection.confidence,
                'indicators': detection.indicators,
                'detected_at': detection.detected_at.isoformat(),
                'user_id': detection.user_id,
                'tenant_id': detection.tenant_id,
                'query': detection.query,
                'metadata': detection.metadata
            }
            detections_data.append(detection_dict)
        
        return jsonify({
            'query': query,
            'threat_detections': detections_data,
            'total_threats': len(detections_data),
            'highest_severity': max([d['metadata']['severity'] for d in detections_data], default='none')
        })
        
    except Exception as e:
        logger.error(f"Error classifying threats: {e}")
        return jsonify({'error': str(e)}), 500

@vector_security_bp.route('/threats/summary', methods=['GET'])
def get_threat_summary():
    """Get comprehensive threat summary"""
    try:
        summary = vector_integration.vector_agent.threat_classifier.get_threat_summary()
        return jsonify(summary)
        
    except Exception as e:
        logger.error(f"Error getting threat summary: {e}")
        return jsonify({'error': str(e)}), 500

@vector_security_bp.route('/threats/definitions', methods=['GET'])
def get_threat_definitions():
    """Get detailed threat definitions"""
    try:
        threat_class = request.args.get('threat_class')
        
        if threat_class:
            # Get specific threat definition
            from threat_classification import ThreatClass
            try:
                threat_enum = ThreatClass(threat_class)
                definition = vector_integration.vector_agent.threat_classifier.threat_definitions[threat_enum]
                
                definition_dict = {
                    'threat_class': definition.threat_class.value,
                    'attack_vector': definition.attack_vector.value,
                    'description': definition.description,
                    'example_attack': definition.example_attack,
                    'defense_layers': definition.defense_layers,
                    'detection_methods': definition.detection_methods,
                    'mitigation_strategies': definition.mitigation_strategies,
                    'severity': definition.severity,
                    'likelihood': definition.likelihood
                }
                
                return jsonify(definition_dict)
            except ValueError:
                return jsonify({'error': f'Unknown threat class: {threat_class}'}), 400
        else:
            # Get all threat definitions
            definitions = {}
            for threat_class, definition in vector_integration.vector_agent.threat_classifier.threat_definitions.items():
                definitions[threat_class.value] = {
                    'threat_class': definition.threat_class.value,
                    'attack_vector': definition.attack_vector.value,
                    'description': definition.description,
                    'example_attack': definition.example_attack,
                    'defense_layers': definition.defense_layers,
                    'detection_methods': definition.detection_methods,
                    'mitigation_strategies': definition.mitigation_strategies,
                    'severity': definition.severity,
                    'likelihood': definition.likelihood
                }
            
            return jsonify(definitions)
        
    except Exception as e:
        logger.error(f"Error getting threat definitions: {e}")
        return jsonify({'error': str(e)}), 500

@vector_security_bp.route('/health', methods=['GET'])
def vector_security_health():
    """Health check for vector security API"""
    try:
        # Check if vector integration is available
        if not vector_integration:
            return jsonify({
                'status': 'unhealthy',
                'error': 'Vector integration not initialized'
            }), 503
        
        # Get basic stats
        stats = {
            'status': 'healthy',
            'active_sessions': len(vector_integration.active_sessions),
            'tenant_policies': len(vector_integration.vector_agent.tenant_policies),
            'total_incidents': len(vector_integration.vector_agent.incident_log),
            'embedding_metadata': len(vector_integration.vector_agent.embedding_metadata),
            'threat_classes': len(vector_integration.vector_agent.threat_classifier.threat_definitions)
        }
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error checking vector security health: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

# Error handlers
@vector_security_bp.errorhandler(400)
def bad_request(error):
    return jsonify({'error': 'Bad request', 'message': str(error)}), 400

@vector_security_bp.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found', 'message': str(error)}), 404

@vector_security_bp.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error', 'message': str(error)}), 500
