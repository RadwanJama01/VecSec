"""
Sandbox Manager for VecSec Red Team Framework
============================================

This module implements secure sandbox environments for executing attacks
against VecSec in isolated containers. It provides Docker-based isolation
with security constraints, resource limits, and comprehensive monitoring.

Author: VecSec Team
"""

import docker
import asyncio
import logging
import json
import uuid
import time
from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import aiohttp
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class SandboxEnvironment:
    """
    Represents an isolated sandbox environment for attack execution.
    """
    sandbox_id: str
    container: docker.models.containers.Container
    created_at: datetime
    last_used: datetime
    resource_stats: Dict = None
    
    def __post_init__(self):
        if self.resource_stats is None:
            self.resource_stats = {}

class SandboxManager:
    """
    Manages creation and destruction of isolated sandbox environments.
    
    Features:
    - Docker-based container isolation
    - Security constraints (seccomp, capabilities)
    - Resource limits (CPU, memory, network)
    - Automatic cleanup and monitoring
    - Network isolation
    - Read-only filesystem
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.docker_client = docker.from_env()
        self.active_sandboxes: Dict[str, SandboxEnvironment] = {}
        
        # Security configuration
        self.seccomp_profile_path = config.get(
            'seccomp_profile',
            './seccomp-profiles/attack-sandbox.json'
        )
        
        # Resource limits
        self.default_mem_limit = config.get('mem_limit', '4g')
        self.default_cpu_quota = config.get('cpu_quota', 200000)
        self.default_pids_limit = config.get('pids_limit', 100)
        
        # Network configuration
        self.sandbox_network = config.get('network', 'sandbox-network')
        self.target_endpoint = config.get('target_endpoint', 'http://localhost:8080')
        
        # Cleanup configuration
        self.max_sandbox_age = timedelta(hours=config.get('max_sandbox_age_hours', 2))
        self.cleanup_interval = config.get('cleanup_interval', 300)  # 5 minutes
        
        # Initialize Docker network
        self._ensure_network_exists()
        
        logger.info(f"SandboxManager initialized with {len(self.active_sandboxes)} active sandboxes")
    
    def _ensure_network_exists(self):
        """Ensure the sandbox network exists."""
        try:
            networks = self.docker_client.networks.list()
            network_names = [net.name for net in networks]
            
            if self.sandbox_network not in network_names:
                logger.info(f"Creating sandbox network: {self.sandbox_network}")
                self.docker_client.networks.create(
                    self.sandbox_network,
                    driver="bridge",
                    ipam=docker.types.IPAMConfig(
                        driver="default",
                        config=[docker.types.IPAMPool(subnet="172.25.0.0/16")]
                    )
                )
            else:
                logger.info(f"Sandbox network {self.sandbox_network} already exists")
                
        except Exception as e:
            logger.error(f"Failed to ensure network exists: {e}")
    
    async def create_isolated_environment(self) -> SandboxEnvironment:
        """
        Create a hardened sandbox environment for attack execution.
        
        Returns:
            SandboxEnvironment instance
        """
        sandbox_id = f"sandbox_{uuid.uuid4().hex[:8]}"
        
        try:
            # Container configuration
            container_config = {
                'image': self.config.get('sandbox_image', 'vecsec-attack-sandbox:latest'),
                'name': sandbox_id,
                'detach': True,
                'remove': False,  # Manual cleanup for inspection
                
                # Resource limits
                'mem_limit': self.default_mem_limit,
                'cpu_quota': self.default_cpu_quota,
                'cpu_period': 100000,
                'pids_limit': self.default_pids_limit,
                
                # Security options
                'security_opt': [
                    'no-new-privileges:true',
                    f'seccomp={self.seccomp_profile_path}'
                ],
                
                # Capability restrictions
                'cap_drop': ['ALL'],
                'cap_add': ['NET_BIND_SERVICE'],
                
                # Network
                'network': self.sandbox_network,
                
                # Filesystem
                'read_only': True,
                'tmpfs': {
                    '/tmp': 'rw,noexec,nosuid,size=100m',
                    '/var/tmp': 'rw,noexec,nosuid,size=50m'
                },
                
                # Environment
                'environment': {
                    'SANDBOX_ID': sandbox_id,
                    'TARGET_ENDPOINT': self.target_endpoint,
                    'MAX_EXECUTION_TIME': str(self.config.get('max_execution_time', 300)),
                    'PYTHONUNBUFFERED': '1',
                    'SANDBOX_MODE': 'true'
                },
                
                # Volumes (read-only where possible)
                'volumes': {
                    self.config.get('log_dir', './logs'): {
                        'bind': '/app/logs',
                        'mode': 'rw'
                    }
                }
            }
            
            # Create and start container
            container = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.docker_client.containers.create(**container_config)
            )
            
            await asyncio.get_event_loop().run_in_executor(
                None,
                container.start
            )
            
            # Wait for container to be ready
            await self._wait_for_container_ready(container)
            
            # Create sandbox environment
            sandbox = SandboxEnvironment(
                sandbox_id=sandbox_id,
                container=container,
                created_at=datetime.now(),
                last_used=datetime.now()
            )
            
            self.active_sandboxes[sandbox_id] = sandbox
            
            logger.info(f"Created sandbox: {sandbox_id}")
            return sandbox
            
        except Exception as e:
            logger.error(f"Failed to create sandbox: {e}")
            raise
    
    async def _wait_for_container_ready(self, container, timeout: int = 30):
        """Wait for container to be ready to accept requests."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Check if container is running
                container.reload()
                if container.status == 'running':
                    # Try to execute a simple command to check readiness
                    result = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: container.exec_run('echo "ready"', demux=True)
                    )
                    
                    if result[0] == 0:  # Exit code 0 means success
                        logger.debug(f"Container {container.name} is ready")
                        return
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.debug(f"Container readiness check failed: {e}")
                await asyncio.sleep(1)
        
        raise TimeoutError(f"Container {container.name} did not become ready within {timeout} seconds")
    
    async def destroy_environment(self, sandbox: SandboxEnvironment):
        """
        Safely destroy sandbox environment.
        
        Args:
            sandbox: SandboxEnvironment to destroy
        """
        try:
            # Stop container (graceful)
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: sandbox.container.stop(timeout=10)
            )
            
            # Get logs before removal (for forensics)
            if self.config.get('save_sandbox_logs', True):
                await self._save_sandbox_logs(sandbox)
            
            # Remove container
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: sandbox.container.remove(force=True)
            )
            
            # Remove from tracking
            if sandbox.sandbox_id in self.active_sandboxes:
                del self.active_sandboxes[sandbox.sandbox_id]
            
            logger.info(f"Destroyed sandbox: {sandbox.sandbox_id}")
            
        except Exception as e:
            logger.error(f"Error destroying sandbox {sandbox.sandbox_id}: {e}")
    
    async def _save_sandbox_logs(self, sandbox: SandboxEnvironment):
        """Save sandbox logs for forensic analysis."""
        try:
            logs = await asyncio.get_event_loop().run_in_executor(
                None,
                sandbox.container.logs
            )
            
            log_path = f"{self.config.get('log_dir', './logs')}/sandbox_{sandbox.sandbox_id}.log"
            
            # Write logs to file
            with open(log_path, 'wb') as f:
                f.write(logs)
            
            logger.info(f"Saved sandbox logs: {log_path}")
            
        except Exception as e:
            logger.error(f"Failed to save sandbox logs: {e}")
    
    async def cleanup_all_sandboxes(self):
        """Emergency cleanup of all sandboxes."""
        logger.warning("Cleaning up all sandboxes")
        
        for sandbox in list(self.active_sandboxes.values()):
            try:
                await self.destroy_environment(sandbox)
            except Exception as e:
                logger.error(f"Failed to cleanup sandbox {sandbox.sandbox_id}: {e}")
        
        logger.info("All sandboxes cleaned up")
    
    async def cleanup_old_sandboxes(self):
        """Cleanup sandboxes that are too old."""
        now = datetime.now()
        to_remove = []
        
        for sandbox_id, sandbox in self.active_sandboxes.items():
            age = now - sandbox.created_at
            
            if age > self.max_sandbox_age:
                to_remove.append(sandbox_id)
        
        for sandbox_id in to_remove:
            sandbox = self.active_sandboxes[sandbox_id]
            logger.info(f"Cleaning up old sandbox: {sandbox_id} (age: {now - sandbox.created_at})")
            
            try:
                await self.destroy_environment(sandbox)
            except Exception as e:
                logger.error(f"Failed to cleanup old sandbox {sandbox_id}: {e}")
    
    async def get_sandbox_stats(self) -> Dict:
        """Get statistics about active sandboxes."""
        stats = {
            'active_sandboxes': len(self.active_sandboxes),
            'total_created': 0,
            'total_destroyed': 0,
            'average_age_minutes': 0,
            'resource_usage': {
                'total_cpu_usage': 0,
                'total_memory_usage': 0,
                'total_network_bytes': 0
            }
        }
        
        if not self.active_sandboxes:
            return stats
        
        total_age = 0
        total_cpu = 0
        total_memory = 0
        total_network = 0
        
        for sandbox in self.active_sandboxes.values():
            age = datetime.now() - sandbox.created_at
            total_age += age.total_seconds()
            
            # Get resource stats
            resource_stats = await self.get_resource_stats(sandbox)
            total_cpu += resource_stats.get('cpu_usage', 0)
            total_memory += resource_stats.get('memory_usage', 0)
            total_network += resource_stats.get('network_bytes_sent', 0)
            total_network += resource_stats.get('network_bytes_received', 0)
        
        stats['average_age_minutes'] = total_age / len(self.active_sandboxes) / 60
        stats['resource_usage']['total_cpu_usage'] = total_cpu
        stats['resource_usage']['total_memory_usage'] = total_memory
        stats['resource_usage']['total_network_bytes'] = total_network
        
        return stats

# Enhanced SandboxEnvironment with HTTP client capabilities
class EnhancedSandboxEnvironment(SandboxEnvironment):
    """
    Enhanced sandbox environment with HTTP client capabilities.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=10)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def execute_secure_request(self,
                                    url: str,
                                    method: str = "GET",
                                    json_data: Optional[Dict] = None,
                                    headers: Optional[Dict] = None,
                                    timeout: int = 30) -> Dict:
        """
        Execute HTTP request from within sandbox.
        
        Args:
            url: Target URL
            method: HTTP method
            json_data: JSON data to send
            headers: Additional headers
            timeout: Request timeout in seconds
            
        Returns:
            Response dictionary
        """
        if not self.session:
            raise RuntimeError("Sandbox session not initialized. Use async context manager.")
        
        try:
            # Prepare headers
            request_headers = {
                'Content-Type': 'application/json',
                'User-Agent': f'VecSec-Sandbox/{self.sandbox_id}',
                **(headers or {})
            }
            
            # Execute request
            async with self.session.request(
                method=method,
                url=url,
                json=json_data,
                headers=request_headers,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                
                # Read response
                try:
                    response_data = await response.json()
                except aiohttp.ContentTypeError:
                    response_data = await response.text()
                
                # Update last used time
                self.last_used = datetime.now()
                
                return {
                    'status_code': response.status,
                    'status': 'success' if response.status < 400 else 'error',
                    'data': response_data,
                    'headers': dict(response.headers),
                    'url': str(response.url)
                }
                
        except asyncio.TimeoutError:
            return {
                'status_code': 408,
                'status': 'timeout',
                'error': 'Request timeout',
                'url': url
            }
        except Exception as e:
            logger.error(f"Sandbox request failed: {e}")
            return {
                'status_code': 500,
                'status': 'error',
                'error': str(e),
                'url': url
            }
    
    async def get_resource_stats(self) -> Dict:
        """Get resource usage statistics for sandbox."""
        try:
            # Get container stats
            stats = await asyncio.get_event_loop().run_in_executor(
                None,
                self.container.stats,
                False  # stream=False
            )
            
            # Parse stats
            cpu_stats = stats.get('cpu_stats', {})
            memory_stats = stats.get('memory_stats', {})
            network_stats = stats.get('networks', {})
            
            cpu_usage = self._calculate_cpu_percentage(cpu_stats)
            memory_usage = memory_stats.get('usage', 0)
            
            # Network stats
            network_bytes_sent = 0
            network_bytes_received = 0
            for interface_stats in network_stats.values():
                network_bytes_sent += interface_stats.get('tx_bytes', 0)
                network_bytes_received += interface_stats.get('rx_bytes', 0)
            
            resource_stats = {
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'network_bytes_sent': network_bytes_sent,
                'network_bytes_received': network_bytes_received,
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache stats
            self.resource_stats = resource_stats
            
            return resource_stats
            
        except Exception as e:
            logger.error(f"Failed to get resource stats: {e}")
            return self.resource_stats or {}
    
    def _calculate_cpu_percentage(self, cpu_stats: Dict) -> float:
        """Calculate CPU usage percentage."""
        try:
            cpu_delta = cpu_stats['cpu_usage']['total_usage'] - \
                       cpu_stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = cpu_stats['system_cpu_usage'] - \
                          cpu_stats['precpu_stats']['system_cpu_usage']
            
            if system_delta > 0:
                return (cpu_delta / system_delta) * 100.0
            return 0.0
        except KeyError:
            return 0.0

# Enhanced SandboxManager that creates EnhancedSandboxEnvironment
class EnhancedSandboxManager(SandboxManager):
    """
    Enhanced sandbox manager that creates enhanced sandbox environments.
    """
    
    async def create_isolated_environment(self) -> EnhancedSandboxEnvironment:
        """Create enhanced sandbox environment."""
        sandbox_id = f"sandbox_{uuid.uuid4().hex[:8]}"
        
        try:
            # Use parent class to create container
            container_config = {
                'image': self.config.get('sandbox_image', 'vecsec-attack-sandbox:latest'),
                'name': sandbox_id,
                'detach': True,
                'remove': False,
                'mem_limit': self.default_mem_limit,
                'cpu_quota': self.default_cpu_quota,
                'cpu_period': 100000,
                'pids_limit': self.default_pids_limit,
                'security_opt': [
                    'no-new-privileges:true',
                    f'seccomp={self.seccomp_profile_path}'
                ],
                'cap_drop': ['ALL'],
                'cap_add': ['NET_BIND_SERVICE'],
                'network': self.sandbox_network,
                'read_only': True,
                'tmpfs': {
                    '/tmp': 'rw,noexec,nosuid,size=100m',
                    '/var/tmp': 'rw,noexec,nosuid,size=50m'
                },
                'environment': {
                    'SANDBOX_ID': sandbox_id,
                    'TARGET_ENDPOINT': self.target_endpoint,
                    'MAX_EXECUTION_TIME': str(self.config.get('max_execution_time', 300)),
                    'PYTHONUNBUFFERED': '1',
                    'SANDBOX_MODE': 'true'
                },
                'volumes': {
                    self.config.get('log_dir', './logs'): {
                        'bind': '/app/logs',
                        'mode': 'rw'
                    }
                }
            }
            
            container = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.docker_client.containers.create(**container_config)
            )
            
            await asyncio.get_event_loop().run_in_executor(
                None,
                container.start
            )
            
            await self._wait_for_container_ready(container)
            
            # Create enhanced sandbox environment
            sandbox = EnhancedSandboxEnvironment(
                sandbox_id=sandbox_id,
                container=container,
                created_at=datetime.now(),
                last_used=datetime.now()
            )
            
            self.active_sandboxes[sandbox_id] = sandbox
            
            logger.info(f"Created enhanced sandbox: {sandbox_id}")
            return sandbox
            
        except Exception as e:
            logger.error(f"Failed to create enhanced sandbox: {e}")
            raise

# Example usage
if __name__ == "__main__":
    # Example configuration
    config = {
        'sandbox_image': 'vecsec-attack-sandbox:latest',
        'mem_limit': '2g',
        'cpu_quota': 100000,
        'pids_limit': 50,
        'network': 'sandbox-network',
        'target_endpoint': 'http://localhost:8080',
        'max_execution_time': 300,
        'log_dir': './logs',
        'save_sandbox_logs': True,
        'max_sandbox_age_hours': 2,
        'cleanup_interval': 300
    }
    
    # Create sandbox manager
    sandbox_manager = EnhancedSandboxManager(config)
    
    async def test_sandbox():
        # Create sandbox
        async with await sandbox_manager.create_isolated_environment() as sandbox:
            # Execute test request
            response = await sandbox.execute_secure_request(
                url="http://localhost:8080/health",
                method="GET"
            )
            
            print(f"Response: {response}")
            
            # Get resource stats
            stats = await sandbox.get_resource_stats()
            print(f"Resource stats: {stats}")
    
    # Run test
    asyncio.run(test_sandbox())
