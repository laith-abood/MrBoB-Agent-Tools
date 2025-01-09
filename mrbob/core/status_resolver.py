"""
Advanced status resolution module implementing CQRS pattern for policy status
management.

This module provides sophisticated status resolution with:
- Concurrent status updates using actor-based processing
- Event sourcing for status history
- In-memory caching with TTL strategy
- Optimistic locking for conflict resolution
"""

from typing import Dict, Optional, Tuple, List, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import threading
from cachetools import TTLCache
import logging
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
import hashlib
import json
import zlib
from typing import NamedTuple

logger = logging.getLogger(__name__)

class VersionVector(NamedTuple):
    """Vector clock for conflict resolution."""
    node_id: str
    counter: int
    timestamp: datetime

class StatusValidationRule(NamedTuple):
    """Rule for validating status transitions."""
    from_status: str
    to_status: str
    allowed_sources: Set[str]
    min_duration: Optional[timedelta] = None
    required_metadata: Set[str] = set()

class StatusPriority(Enum):
    """Status priority hierarchy for resolution."""
    ACTIVE = 1
    PAID = 1
    INFORCE = 1
    PENDING = 2
    TERMINATED = 3
    LAPSED = 4
    UNKNOWN = 99

class StatusSource(Enum):
    """Status source classification."""
    CARRIER = 1
    AGENCY = 2
    SYSTEM = 3
    MANUAL = 4

@dataclass
class StatusTransition:
    """Immutable status transition record."""
    policy_id: str
    from_status: str
    to_status: str
    transition_date: datetime
    source: StatusSource
    metadata: Dict = field(default_factory=dict)
    transition_id: Optional[str] = None

    def __post_init__(self):
        """Generate unique transition ID."""
        if not self.transition_id:
            content = f"{self.policy_id}:{self.from_status}:{self.to_status}:{self.transition_date.isoformat()}"
            self.transition_id = hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'transition_id': self.transition_id,
            'policy_id': self.policy_id,
            'from_status': self.from_status,
            'to_status': self.to_status,
            'transition_date': self.transition_date.isoformat(),
            'source': self.source.name,
            'metadata': self.metadata
        }

@dataclass
class CompressedTransitions:
    """Compressed status transition history."""
    def __init__(self, transitions: List[StatusTransition]):
        self.data = zlib.compress(
            json.dumps([t.to_dict() for t in transitions]).encode()
        )
    
    def decompress(self) -> List[Dict]:
        """Decompress transition history."""
        return json.loads(zlib.decompress(self.data).decode())

@dataclass
class StatusContext:
    """Rich context for status resolution."""
    current_status: str
    source: StatusSource
    effective_date: datetime
    version: int = 0
    metadata: Dict = field(default_factory=dict)
    transitions: List[StatusTransition] = field(default_factory=list)

class StatusValidator:
    """Validates status transitions based on rules."""
    
    def __init__(self):
        self._rules = self._initialize_rules()
    
    def _initialize_rules(self) -> List[StatusValidationRule]:
        """Initialize status transition rules."""
        return [
            StatusValidationRule(
                from_status='PENDING',
                to_status='ACTIVE',
                allowed_sources={'CARRIER', 'AGENCY'},
                min_duration=timedelta(days=1),
                required_metadata={'effective_date', 'carrier_id'}
            ),
            StatusValidationRule(
                from_status='ACTIVE',
                to_status='TERMINATED',
                allowed_sources={'CARRIER'},
                min_duration=timedelta(days=30),
                required_metadata={'termination_reason'}
            ),
            # Add more rules as needed
        ]
    
    def validate_transition(
        self,
        from_status: str,
        to_status: str,
        source: StatusSource,
        context: StatusContext,
        metadata: Dict
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate status transition against rules.
        
        Args:
            from_status: Current status
            to_status: Proposed status
            source: Status source
            context: Current context
            metadata: Transition metadata
            
        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        for rule in self._rules:
            if (rule.from_status == from_status and rule.to_status == to_status):
                # Check source
                if source.name not in rule.allowed_sources:
                    return False, f"Invalid source {source.name} for transition"
                
                # Check duration
                if rule.min_duration:
                    duration = datetime.now() - context.effective_date
                    if duration < rule.min_duration:
                        return False, "Minimum duration not met"
                
                # Check metadata
                missing = rule.required_metadata - set(metadata.keys())
                if missing:
                    return False, f"Missing required metadata: {missing}"
                
                return True, None
        
        return False, "No matching transition rule"

class StatusResolutionStrategy:
    """Strategy pattern for status resolution logic."""

    @staticmethod
    def get_status_priority(status: str) -> int:
        """Get numerical priority for status."""
        try:
            return StatusPriority[status.upper()].value
        except KeyError:
            return StatusPriority.UNKNOWN.value

    @staticmethod
    def should_update_status(
        current: StatusContext,
        new_status: str,
        new_source: StatusSource,
        new_date: datetime
    ) -> bool:
        """
        Determine if status should be updated based on complex rules.
        
        Args:
            current: Current status context
            new_status: Proposed new status
            new_source: Source of new status
            new_date: Effective date of new status
            
        Returns:
            bool: True if status should be updated
        """
        # Source priority check
        if new_source.value < current.source.value:
            return True

        # Same source, check status priority
        if new_source == current.source:
            current_priority = StatusResolutionStrategy.get_status_priority(
                current.current_status
            )
            new_priority = StatusResolutionStrategy.get_status_priority(
                new_status
            )
            
            if new_priority < current_priority:
                return True
            
            # Same priority, use newer date
            if new_priority == current_priority:
                return new_date > current.effective_date

        return False

class ConflictResolver:
    """Resolves conflicts using version vectors."""
    
    @staticmethod
    def compare_versions(v1: VersionVector, v2: VersionVector) -> int:
        """
        Compare version vectors.
        
        Returns:
            int: -1 if v1 < v2, 0 if concurrent, 1 if v1 > v2
        """
        if v1.timestamp < v2.timestamp:
            return -1
        elif v1.timestamp > v2.timestamp:
            return 1
        
        if v1.counter < v2.counter:
            return -1
        elif v1.counter > v2.counter:
            return 1
        
        return 0

class StatusCache:
    """Thread-safe status cache with TTL and LRU eviction."""
    
    def __init__(self, maxsize: int = 10000, ttl: int = 3600):
        """
        Initialize cache with size and TTL limits.
        
        Args:
            maxsize: Maximum cache size
            ttl: Cache TTL in seconds
        """
        self.cache = TTLCache(maxsize=maxsize, ttl=ttl)
        self.lock = threading.Lock()
        
    def get(self, key: str) -> Optional[StatusContext]:
        """Thread-safe cache retrieval."""
        with self.lock:
            return self.cache.get(key)
            
    def set(self, key: str, value: StatusContext) -> None:
        """Thread-safe cache update."""
        with self.lock:
            self.cache[key] = value

class StatusResolver:
    """
    Advanced status resolver with concurrent processing and caching.
    
    Features:
    - Actor-based concurrent processing
    - Event sourcing for status history
    - Optimistic locking for conflict resolution
    - In-memory caching with TTL
    - Status transition validation
    - Compressed history storage
    """
    """
    Advanced status resolver with concurrent processing and caching.
    
    Features:
    - Actor-based concurrent processing
    - Event sourcing for status history
    - Optimistic locking for conflict resolution
    - In-memory caching with TTL
    """
    
    def __init__(
        self,
        cache_size: int = 10000,
        cache_ttl: int = 3600,
        max_workers: int = 4,
        node_id: str = 'default'
    ):
        """
        Initialize resolver with configuration.
        
        Args:
            cache_size: Maximum cache size
            cache_ttl: Cache TTL in seconds
            max_workers: Maximum worker threads
        """
        self.cache = StatusCache(cache_size, cache_ttl)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.strategy = StatusResolutionStrategy()
        self.validator = StatusValidator()
        self.conflict_resolver = ConflictResolver()
        self._status_queue = asyncio.Queue()
        self._processing = False
        self._processing_lock = threading.Lock()
        self._node_id = node_id
        self._version_counter = 0

    async def start_processing(self) -> None:
        """Start asynchronous status processing."""
        with self._processing_lock:
            if self._processing:
                return
            self._processing = True
            
        while self._processing:
            try:
                # Process status updates in batches
                batch = []
                try:
                    while len(batch) < 100:  # Max batch size
                        item = await asyncio.wait_for(
                            self._status_queue.get(),
                            timeout=0.1
                        )
                        batch.append(item)
                except asyncio.TimeoutError:
                    pass
                
                if batch:
                    # Process batch concurrently
                    await asyncio.gather(*[
                        self._process_status_update(*item)
                        for item in batch
                    ])
                    
            except Exception as e:
                logger.error(f"Error processing status batch: {str(e)}")
                continue

    async def stop_processing(self) -> None:
        """Stop asynchronous processing."""
        with self._processing_lock:
            self._processing = False

    def _get_next_version(self) -> VersionVector:
        """Get next version vector."""
        self._version_counter += 1
        return VersionVector(
            node_id=self._node_id,
            counter=self._version_counter,
            timestamp=datetime.now()
        )

    async def resolve_status(
        self,
        policy_id: str,
        new_status: str,
        source: StatusSource,
        effective_date: datetime,
        metadata: Optional[Dict] = None
    ) -> Tuple[str, Optional[StatusTransition]]:
        """
        Resolve policy status with advanced conflict resolution.
        
        Args:
            policy_id: Policy identifier
            new_status: Proposed new status
            source: Status source
            effective_date: Effective date
            metadata: Optional metadata
            
        Returns:
            Tuple[str, Optional[StatusTransition]]: Resolved status and transition
        """
        try:
            # Get current status context
            current = self.cache.get(policy_id)
            
            if not current:
                # Initialize new context
                current = StatusContext(
                    current_status=new_status,
                    source=source,
                    effective_date=effective_date,
                    metadata=metadata or {}
                )
                self.cache.set(policy_id, current)
                return new_status, None
            
            # Validate transition
            is_valid, error = self.validator.validate_transition(
                current.current_status,
                new_status,
                source,
                current,
                metadata or {}
            )
            
            if not is_valid:
                logger.warning(
                    f"Invalid transition for policy {policy_id}: {error}"
                )
                return current.current_status, None
            
            # Check if update needed
            if self.strategy.should_update_status(
                current,
                new_status,
                source,
                effective_date
            ):
                # Get next version
                next_version = self._get_next_version()
                # Create transition record with version
                transition = StatusTransition(
                    policy_id=policy_id,
                    from_status=current.current_status,
                    to_status=new_status,
                    transition_date=datetime.now(),
                    source=source,
                    metadata={
                        **(metadata or {}),
                        'version': next_version._asdict()
                    }
                )
                
                # Queue update
                await self._status_queue.put(
                    (policy_id, new_status, source, effective_date, transition)
                )
                
                return new_status, transition
            
            return current.current_status, None
            
        except Exception as e:
            logger.error(f"Error resolving status: {str(e)}")
            raise

    async def _process_status_update(
        self,
        policy_id: str,
        new_status: str,
        source: StatusSource,
        effective_date: datetime,
        transition: StatusTransition
    ) -> None:
        """
        Process status update with optimistic locking.
        
        Args:
            policy_id: Policy identifier
            new_status: New status value
            source: Status source
            effective_date: Effective date
            transition: Status transition record
        """
        try:
            # Get current context
            current = self.cache.get(policy_id)
            if not current:
                return
            
            # Optimistic locking check
            if current.version != transition.metadata.get('version', 0):
                logger.warning(
                    f"Concurrent modification detected for policy {policy_id}"
                )
                return
            
            # Check for conflicts
            current_version = VersionVector(**current.metadata.get('version', {
                'node_id': self._node_id,
                'counter': 0,
                'timestamp': datetime.min
            }))
            
            comparison = self.conflict_resolver.compare_versions(
                current_version,
                transition.metadata['version']
            )
            
            if comparison >= 0:
                logger.warning(
                    f"Concurrent modification detected for policy {policy_id}"
                )
                return current.current_status, None
            
            # Update context
            current.current_status = new_status
            current.source = source
            current.effective_date = effective_date
            current.metadata['version'] = transition.metadata['version']
            
            # Compress and store transitions
            if len(current.transitions) >= 100:
                compressed = CompressedTransitions(current.transitions)
                current.metadata['compressed_history'] = compressed.data
                current.transitions = [transition]
            else:
                current.transitions.append(transition)
            
            # Update cache
            self.cache.set(policy_id, current)
            
        except Exception as e:
            logger.error(f"Error processing status update: {str(e)}")
            raise

    def get_status_history(
        self,
        policy_id: str
    ) -> List[StatusTransition]:
        """
        Get status transition history for policy.
        
        Args:
            policy_id: Policy identifier
            
        Returns:
            List[StatusTransition]: Status transition history
        """
        try:
            current = self.cache.get(policy_id)
            return current.transitions if current else []
        except Exception as e:
            logger.error(f"Error getting status history: {str(e)}")
            return []
