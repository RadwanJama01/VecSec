"""
Adversarial Test Orchestrator for VecSec Red Team Framework
===========================================================

This module implements the orchestration layer for continuous adversarial
testing against VecSec. It manages attack execution, rate limiting, queue
management, and result processing in a scalable, production-ready manner.

Author: VecSec Team
"""

import asyncio
import logging
import time
import uuid
from asyncio import PriorityQueue, Queue
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json

from llama_attack_generator import AttackSpec, AttackResult, LlamaAttackGenerator

logger = logging.getLogger(__name__)

@dataclass
class AttackTask:
    """Task representing an attack to execute."""
    attack_spec: AttackSpec
    priority: int
    created_at: datetime = field(default_factory=datetime.now)
    retry_count: int = 0
    max_retries: int = 3
    scheduled_at: Optional[datetime] = None
    
    def __lt__(self, other):
        return self.priority < other.priority

@dataclass
class CampaignConfig:
    """Configuration for attack campaigns."""
    campaign_id: str
    name: str
    description: str
    modes: List[Dict]
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    max_attacks: Optional[int] = None
    rate_limit: Optional[int] = None

class AdaptiveRateLimiter:
    """
    Adaptive rate limiter that adjusts based on system response and performance.
    """
    
    def __init__(self, 
                 initial_rate: int = 10,
                 max_rate: int = 50,
                 min_rate: int = 1,
                 window: int = 60):
        self.current_rate = initial_rate
        self.max_rate = max_rate
        self.min_rate = min_rate
        self.window = window  # seconds
        
        # Request tracking
        self.request_times = deque()
        self.success_count = 0
        self.failure_count = 0
        self.detection_count = 0
        
        # Adaptive parameters
        self.success_threshold = 0.8  # Increase rate if success > 80%
        self.detection_threshold = 0.3  # Decrease rate if detection > 30%
        
        self.lock = asyncio.Lock()
        
        logger.info(f"Rate limiter initialized: {initial_rate}/min (max: {max_rate}, min: {min_rate})")
    
    async def acquire(self):
        """Acquire permission to make a request."""
        async with self.lock:
            now = time.time()
            
            # Remove old requests outside window
            while self.request_times and self.request_times[0] < now - self.window:
                self.request_times.popleft()
            
            # Wait if at limit
            if len(self.request_times) >= self.current_rate:
                sleep_time = self.window - (now - self.request_times[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            
            self.request_times.append(time.time())
    
    async def record_result(self, success: bool, detected: bool):
        """Record attack result for adaptive rate adjustment."""
        async with self.lock:
            if success:
                self.success_count += 1
            else:
                self.failure_count += 1
            
            if detected:
                self.detection_count += 1
            
            # Adjust rate based on recent performance
            await self._adjust_rate()
    
    async def _adjust_rate(self):
        """Adjust rate based on recent performance."""
        total_attacks = self.success_count + self.failure_count
        
        if total_attacks < 10:  # Need minimum sample size
            return
        
        success_rate = self.success_count / total_attacks
        detection_rate = self.detection_count / total_attacks
        
        # Increase rate if performing well
        if success_rate > self.success_threshold and detection_rate < self.detection_threshold:
            await self._increase_rate()
        # Decrease rate if being detected too much
        elif detection_rate > self.detection_threshold:
            await self._decrease_rate()
        
        # Reset counters periodically
        if total_attacks > 100:
            self.success_count = 0
            self.failure_count = 0
            self.detection_count = 0
    
    async def _increase_rate(self, factor: float = 1.1):
        """Increase rate limit."""
        old_rate = self.current_rate
        self.current_rate = min(int(self.current_rate * factor), self.max_rate)
        
        if self.current_rate != old_rate:
            logger.info(f"Rate limit increased: {old_rate} -> {self.current_rate}/min")
    
    async def _decrease_rate(self, factor: float = 0.8):
        """Decrease rate limit."""
        old_rate = self.current_rate
        self.current_rate = max(int(self.current_rate * factor), self.min_rate)
        
        if self.current_rate != old_rate:
            logger.info(f"Rate limit decreased: {old_rate} -> {self.current_rate}/min")
    
    def get_stats(self) -> Dict:
        """Get current rate limiter statistics."""
        return {
            'current_rate': self.current_rate,
            'max_rate': self.max_rate,
            'min_rate': self.min_rate,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'detection_count': self.detection_count,
            'total_requests': len(self.request_times)
        }

class AdversarialTestOrchestrator:
    """
    Orchestrates continuous adversarial testing with advanced features.
    
    Features:
    - Multi-worker attack execution
    - Priority-based queue management
    - Adaptive rate limiting
    - Campaign scheduling
    - Result processing and analysis
    - Vulnerability detection
    """
    
    def __init__(self,
                 attack_generator: LlamaAttackGenerator,
                 target_endpoint: str,
                 sandbox_manager,
                 attack_logger,
                 config: Dict):
        self.attack_generator = attack_generator
        self.target_endpoint = target_endpoint
        self.sandbox_manager = sandbox_manager
        self.attack_logger = attack_logger
        self.config = config
        
        # Queue management
        self.attack_queue = PriorityQueue()
        self.results_queue = Queue()
        self.active_attacks: Dict[str, asyncio.Task] = {}
        self.scheduled_campaigns: Dict[str, CampaignConfig] = {}
        
        # Rate limiting
        self.rate_limiter = AdaptiveRateLimiter(
            initial_rate=config.get('initial_rate', 10),
            max_rate=config.get('max_rate', 50),
            min_rate=config.get('min_rate', 1)
        )
        
        # Statistics
        self.stats = {
            'total_attacks': 0,
            'successful_attacks': 0,
            'failed_attacks': 0,
            'blocked_attacks': 0,
            'detected_attacks': 0,
            'vulnerabilities_found': 0,
            'campaigns_completed': 0,
            'start_time': datetime.now()
        }
        
        # Control flags
        self.running = False
        self.paused = False
        self.shutdown_event = asyncio.Event()
        
        # Worker management
        self.workers: List[asyncio.Task] = []
        self.num_workers = config.get('num_workers', 5)
        
        logger.info(f"AdversarialTestOrchestrator initialized with {self.num_workers} workers")
    
    async def start(self):
        """Start the orchestrator."""
        if self.running:
            logger.warning("Orchestrator already running")
            return
        
        self.running = True
        self.shutdown_event.clear()
        
        logger.info("Starting adversarial test orchestrator")
        
        # Start worker tasks
        self.workers = [
            asyncio.create_task(self._attack_worker(i))
            for i in range(self.num_workers)
        ]
        
        # Start background tasks
        background_tasks = [
            asyncio.create_task(self._process_results()),
            asyncio.create_task(self._campaign_scheduler()),
            asyncio.create_task(self._stats_reporter()),
            asyncio.create_task(self._cleanup_task())
        ]
        
        # Wait for shutdown signal
        try:
            await self.shutdown_event.wait()
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the orchestrator gracefully."""
        if not self.running:
            return
        
        logger.info("Stopping adversarial test orchestrator")
        self.running = False
        self.shutdown_event.set()
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to complete (with timeout)
        if self.workers:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self.workers, return_exceptions=True),
                    timeout=30
                )
            except asyncio.TimeoutError:
                logger.warning("Workers did not stop within timeout")
        
        # Wait for active attacks to complete
        if self.active_attacks:
            logger.info(f"Waiting for {len(self.active_attacks)} active attacks to complete")
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self.active_attacks.values(), return_exceptions=True),
                    timeout=60
                )
            except asyncio.TimeoutError:
                logger.warning("Active attacks did not complete within timeout")
        
        # Cleanup
        await self.sandbox_manager.cleanup_all_sandboxes()
        
        logger.info("Orchestrator stopped")
    
    async def schedule_attack_campaign(self, 
                                      campaign_config: CampaignConfig,
                                      start_immediately: bool = True) -> str:
        """
        Schedule a campaign of attacks.
        
        Args:
            campaign_config: Campaign configuration
            start_immediately: Whether to start executing immediately
            
        Returns:
            Campaign ID
        """
        campaign_id = campaign_config.campaign_id
        
        logger.info(f"Scheduling campaign: {campaign_config.name} ({campaign_id})")
        
        # Store campaign config
        self.scheduled_campaigns[campaign_id] = campaign_config
        
        if start_immediately:
            # Generate and enqueue attacks immediately
            await self._execute_campaign(campaign_config)
        else:
            # Schedule for later execution
            campaign_config.start_time = campaign_config.start_time or datetime.now()
        
        return campaign_id
    
    async def schedule_single_attack(self, 
                                   attack_spec: AttackSpec, 
                                   priority: str = 'medium',
                                   delay: Optional[float] = None) -> str:
        """Schedule a single attack for execution."""
        priority_value = self._get_priority_value(priority)
        
        task = AttackTask(
            attack_spec=attack_spec,
            priority=priority_value,
            scheduled_at=datetime.now() + timedelta(seconds=delay) if delay else None
        )
        
        await self.attack_queue.put(task)
        
        logger.info(f"Scheduled attack: {attack_spec.attack_id} (priority: {priority})")
        return attack_spec.attack_id
    
    async def pause(self):
        """Pause attack execution."""
        self.paused = True
        logger.info("Orchestrator paused")
    
    async def resume(self):
        """Resume attack execution."""
        self.paused = False
        logger.info("Orchestrator resumed")
    
    async def _execute_campaign(self, campaign_config: CampaignConfig):
        """Execute a campaign by generating and enqueueing attacks."""
        try:
            # Generate attacks using Llama model
            campaign_dict = {
                'campaign_id': campaign_config.campaign_id,
                'modes': campaign_config.modes
            }
            
            attacks = self.attack_generator.generate_campaign(campaign_dict)
            
            # Enqueue all attacks
            for attack in attacks:
                priority = attack.priority
                priority_value = self._get_priority_value(priority)
                
                task = AttackTask(
                    attack_spec=attack,
                    priority=priority_value
                )
                
                await self.attack_queue.put(task)
            
            logger.info(f"Enqueued {len(attacks)} attacks from campaign {campaign_config.campaign_id}")
            
        except Exception as e:
            logger.error(f"Failed to execute campaign {campaign_config.campaign_id}: {e}")
    
    async def _attack_worker(self, worker_id: int):
        """Worker coroutine that executes attacks from queue."""
        logger.info(f"Starting attack worker {worker_id}")
        
        while self.running:
            try:
                # Wait for rate limiter
                await self.rate_limiter.acquire()
                
                # Get attack from queue (with timeout)
                try:
                    task = await asyncio.wait_for(
                        self.attack_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Skip if paused
                if self.paused:
                    await self.attack_queue.put(task)
                    await asyncio.sleep(1)
                    continue
                
                # Check if attack is scheduled for later
                if task.scheduled_at and task.scheduled_at > datetime.now():
                    await self.attack_queue.put(task)
                    await asyncio.sleep(1)
                    continue
                
                # Execute attack
                attack_id = task.attack_spec.attack_id
                logger.info(f"Worker {worker_id} executing attack: {attack_id}")
                
                # Track active attack
                attack_task = asyncio.create_task(
                    self._execute_attack_with_retry(task)
                )
                self.active_attacks[attack_id] = attack_task
                
                try:
                    result = await attack_task
                    
                    # Queue result for processing
                    await self.results_queue.put((task, result))
                    
                    # Update stats
                    self.stats['total_attacks'] += 1
                    
                finally:
                    # Remove from active attacks
                    self.active_attacks.pop(attack_id, None)
                
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(1)
        
        logger.info(f"Attack worker {worker_id} stopped")
    
    async def _execute_attack_with_retry(self, task: AttackTask) -> AttackResult:
        """Execute attack with retry logic."""
        attack_spec = task.attack_spec
        
        while task.retry_count < task.max_retries:
            try:
                # Create sandbox
                sandbox = await self.sandbox_manager.create_isolated_environment()
                
                try:
                    # Execute based on mode
                    mode = attack_spec.mode
                    start_time = time.time()
                    
                    if mode == 'PRE_EMBEDDING':
                        result = await self._execute_pre_embedding_attack(
                            sandbox, attack_spec
                        )
                    elif mode == 'POST_EMBEDDING':
                        result = await self._execute_post_embedding_attack(
                            sandbox, attack_spec
                        )
                    else:  # HYBRID
                        result = await self._execute_hybrid_attack(
                            sandbox, attack_spec
                        )
                    
                    execution_time = time.time() - start_time
                    
                    # Collect telemetry
                    telemetry = await self._collect_telemetry(sandbox, execution_time)
                    
                    return AttackResult(
                        attack_id=attack_spec.attack_id,
                        success=result.get('success', False),
                        evasion_successful=result.get('evasion_successful', False),
                        detected_as_malicious=result.get('detected', False),
                        response=result.get('response', {}),
                        execution_time=execution_time,
                        telemetry=telemetry
                    )
                    
                finally:
                    # Always cleanup sandbox
                    await self.sandbox_manager.destroy_environment(sandbox)
                    
            except Exception as e:
                task.retry_count += 1
                logger.error(
                    f"Attack {attack_spec.attack_id} failed "
                    f"(attempt {task.retry_count}/{task.max_retries}): {e}"
                )
                
                if task.retry_count >= task.max_retries:
                    return AttackResult(
                        attack_id=attack_spec.attack_id,
                        success=False,
                        evasion_successful=False,
                        detected_as_malicious=False,
                        response={},
                        execution_time=0,
                        telemetry={},
                        error=str(e)
                    )
                
                # Exponential backoff
                await asyncio.sleep(2 ** task.retry_count)
        
        return AttackResult(
            attack_id=attack_spec.attack_id,
            success=False,
            evasion_successful=False,
            detected_as_malicious=False,
            response={},
            execution_time=0,
            telemetry={},
            error="Max retries exceeded"
        )
    
    async def _execute_pre_embedding_attack(self, 
                                           sandbox, 
                                           attack_spec: AttackSpec) -> Dict:
        """Execute pre-embedding (query-based) attack."""
        payload = attack_spec.payload
        query = payload['query']
        
        # Send request through sandbox
        response = await sandbox.execute_secure_request(
            url=f"{self.target_endpoint}/api/vector/query",
            method="POST",
            json_data={
                'user_id': payload.get('user_id', 'attacker'),
                'query': query,
                'tenant_id': payload.get('tenant_id', 'default'),
                'query_type': payload.get('query_type', 'search')
            },
            timeout=30
        )
        
        # Evaluate success
        success_criteria = attack_spec.success_criteria
        success = self._evaluate_success(response, success_criteria)
        
        return {
            'success': success,
            'response': response,
            'evasion_successful': response.get('status') == 'approved',
            'detected': 'block' in response.get('action', '').lower()
        }
    
    async def _execute_post_embedding_attack(self, 
                                            sandbox,
                                            attack_spec: AttackSpec) -> Dict:
        """Execute post-embedding (vector manipulation) attack."""
        payload = attack_spec.payload
        embedding = payload.get('embedding_manipulation')
        metadata = payload.get('metadata_injection', {})
        
        # Send embedding insertion request
        response = await sandbox.execute_secure_request(
            url=f"{self.target_endpoint}/api/vector/embedding/insert",
            method="POST",
            json_data={
                'embedding': embedding,
                'metadata': metadata,
                'user_id': payload.get('user_id', 'attacker'),
                'tenant_id': payload.get('tenant_id', 'default')
            },
            timeout=30
        )
        
        success_criteria = attack_spec.success_criteria
        success = self._evaluate_success(response, success_criteria)
        
        return {
            'success': success,
            'response': response,
            'embedding_accepted': response.get('status') == 'approved',
            'constraints_bypassed': 'constraint_violation' not in response
        }
    
    async def _execute_hybrid_attack(self,
                                    sandbox,
                                    attack_spec: AttackSpec) -> Dict:
        """Execute hybrid (multi-stage) attack."""
        results = []
        overall_success = True
        
        # Execute each stage
        stages = attack_spec.payload.get('stages', [])
        for stage_idx, stage in enumerate(stages):
            stage_type = stage['type']
            
            if stage_type == 'query':
                result = await self._execute_query_stage(sandbox, stage)
            elif stage_type == 'embedding':
                result = await self._execute_embedding_stage(sandbox, stage)
            else:
                logger.warning(f"Unknown stage type: {stage_type}")
                continue
            
            results.append(result)
            
            # Check if stage failed
            if not result.get('success', False):
                overall_success = False
                if not stage.get('continue_on_failure', False):
                    break
        
        return {
            'success': overall_success,
            'stages': results,
            'evasion_successful': overall_success
        }
    
    async def _execute_query_stage(self, sandbox, stage: Dict) -> Dict:
        """Execute a query stage in hybrid attack."""
        query = stage['query']
        
        response = await sandbox.execute_secure_request(
            url=f"{self.target_endpoint}/api/vector/query",
            method="POST",
            json_data={
                'query': query,
                'user_id': 'attacker',
                'tenant_id': 'default'
            },
            timeout=30
        )
        
        return {
            'type': 'query',
            'success': response.get('status') == 'approved',
            'response': response
        }
    
    async def _execute_embedding_stage(self, sandbox, stage: Dict) -> Dict:
        """Execute an embedding stage in hybrid attack."""
        embedding = stage['embedding']
        
        response = await sandbox.execute_secure_request(
            url=f"{self.target_endpoint}/api/vector/embedding/insert",
            method="POST",
            json_data={
                'embedding': embedding,
                'user_id': 'attacker',
                'tenant_id': 'default'
            },
            timeout=30
        )
        
        return {
            'type': 'embedding',
            'success': response.get('status') == 'approved',
            'response': response
        }
    
    def _evaluate_success(self, response: Dict, success_criteria: List[str]) -> bool:
        """Evaluate if attack was successful based on criteria."""
        if not success_criteria:
            # Default: success if request was approved
            return response.get('status') == 'approved'
        
        # Check each criterion
        for criterion in success_criteria:
            if 'approved' in criterion.lower():
                if response.get('status') != 'approved':
                    return False
            elif 'no violation' in criterion.lower():
                if 'violation' in str(response).lower():
                    return False
            elif 'bypass' in criterion.lower():
                if 'block' in response.get('action', '').lower():
                    return False
        
        return True
    
    async def _collect_telemetry(self, sandbox, execution_time: float) -> Dict:
        """Collect execution telemetry from sandbox."""
        try:
            stats = await sandbox.get_resource_stats()
            return {
                'sandbox_id': sandbox.sandbox_id,
                'cpu_usage': stats.get('cpu_usage'),
                'memory_usage': stats.get('memory_usage'),
                'network_bytes_sent': stats.get('network_bytes_sent'),
                'network_bytes_received': stats.get('network_bytes_received'),
                'execution_duration_ms': int(execution_time * 1000)
            }
        except Exception as e:
            logger.error(f"Failed to collect telemetry: {e}")
            return {}
    
    async def _process_results(self):
        """Process attack results and log them."""
        logger.info("Starting result processor")
        
        while self.running:
            try:
                # Get result from queue (with timeout)
                try:
                    task, result = await asyncio.wait_for(
                        self.results_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Log attack
                await self.attack_logger.log_attack(
                    attack_spec=task.attack_spec,
                    result=result,
                    telemetry=result.telemetry
                )
                
                # Update statistics
                if result.success:
                    self.stats['successful_attacks'] += 1
                    
                    if result.evasion_successful:
                        logger.warning(
                            f"Successful evasion: {task.attack_spec.attack_id}"
                        )
                        self.stats['vulnerabilities_found'] += 1
                else:
                    self.stats['failed_attacks'] += 1
                
                if result.detected_as_malicious:
                    self.stats['detected_attacks'] += 1
                else:
                    self.stats['blocked_attacks'] += 1
                
                # Record result for rate limiter
                await self.rate_limiter.record_result(
                    success=result.success,
                    detected=result.detected_as_malicious
                )
                
                # Check for vulnerability patterns
                if result.evasion_successful:
                    await self._analyze_for_vulnerability(task.attack_spec, result)
                
            except Exception as e:
                logger.error(f"Result processor error: {e}")
                await asyncio.sleep(1)
        
        logger.info("Result processor stopped")
    
    async def _analyze_for_vulnerability(self, attack_spec: AttackSpec, result: AttackResult):
        """Analyze successful attacks for vulnerability patterns."""
        logger.info(
            f"Analyzing potential vulnerability from attack: "
            f"{attack_spec.attack_id}"
        )
        
        # This would integrate with vulnerability detection logic
        # in the AttackLogger component
        pass
    
    async def _campaign_scheduler(self):
        """Background task to manage scheduled campaigns."""
        while self.running:
            try:
                now = datetime.now()
                
                for campaign_id, campaign_config in list(self.scheduled_campaigns.items()):
                    # Check if campaign should start
                    if (campaign_config.start_time and 
                        campaign_config.start_time <= now and
                        not hasattr(campaign_config, 'executed')):
                        
                        await self._execute_campaign(campaign_config)
                        campaign_config.executed = True
                    
                    # Check if campaign should end
                    if (campaign_config.end_time and 
                        campaign_config.end_time <= now):
                        
                        del self.scheduled_campaigns[campaign_id]
                        self.stats['campaigns_completed'] += 1
                        logger.info(f"Campaign {campaign_id} completed")
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Campaign scheduler error: {e}")
                await asyncio.sleep(10)
    
    async def _stats_reporter(self):
        """Background task to report statistics."""
        while self.running:
            try:
                await asyncio.sleep(300)  # Report every 5 minutes
                
                uptime = datetime.now() - self.stats['start_time']
                rate_stats = self.rate_limiter.get_stats()
                
                logger.info(
                    f"Orchestrator Stats - "
                    f"Uptime: {uptime}, "
                    f"Total Attacks: {self.stats['total_attacks']}, "
                    f"Success Rate: {self.stats['successful_attacks']/max(self.stats['total_attacks'], 1):.2%}, "
                    f"Detection Rate: {self.stats['detected_attacks']/max(self.stats['total_attacks'], 1):.2%}, "
                    f"Current Rate: {rate_stats['current_rate']}/min, "
                    f"Queue Size: {self.attack_queue.qsize()}, "
                    f"Active Attacks: {len(self.active_attacks)}"
                )
                
            except Exception as e:
                logger.error(f"Stats reporter error: {e}")
    
    async def _cleanup_task(self):
        """Background cleanup task."""
        while self.running:
            try:
                await asyncio.sleep(3600)  # Cleanup every hour
                
                # Cleanup old sandboxes
                await self.sandbox_manager.cleanup_old_sandboxes()
                
                # Cleanup old logs
                await self.attack_logger.cleanup_old_logs()
                
            except Exception as e:
                logger.error(f"Cleanup task error: {e}")
    
    def _get_priority_value(self, priority: str) -> int:
        """Convert priority string to numeric value."""
        priority_map = {
            'critical': 1,
            'high': 2,
            'medium': 3,
            'low': 4,
            'exploratory': 5
        }
        return priority_map.get(priority.lower(), 3)
    
    def get_statistics(self) -> Dict:
        """Get current execution statistics."""
        uptime = datetime.now() - self.stats['start_time']
        rate_stats = self.rate_limiter.get_stats()
        
        return {
            **self.stats,
            'uptime_seconds': uptime.total_seconds(),
            'queue_size': self.attack_queue.qsize(),
            'active_attacks': len(self.active_attacks),
            'rate_limiter_stats': rate_stats,
            'paused': self.paused,
            'running': self.running,
            'scheduled_campaigns': len(self.scheduled_campaigns)
        }
    
    def get_queue_status(self) -> Dict:
        """Get detailed queue status."""
        return {
            'queue_size': self.attack_queue.qsize(),
            'active_attacks': len(self.active_attacks),
            'scheduled_campaigns': list(self.scheduled_campaigns.keys()),
            'rate_limit': self.rate_limiter.get_stats()
        }

# Example usage
if __name__ == "__main__":
    # This would be used with actual implementations
    pass
