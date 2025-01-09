"""
Report generation service implementing CQRS pattern with advanced 
architectural features.

Key patterns:
- Command/Query Responsibility Segregation (CQRS)
- Domain Events for state changes
- Unit of Work for transactional integrity
- Dependency Injection for loose coupling
"""

import json
import logging
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Protocol, Set, Tuple
from uuid import UUID
from cachetools import TTLCache
from tenacity import retry, stop_after_attempt, wait_exponential

# Add project root to path for absolute imports
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from mrbob.domain.report_aggregate import (
        Report,
        ReportType,
        ReportMetadata,
        ReportSection,
        ReportRepository
    )
    from mrbob.reports.application.events import EventPublisher
    from mrbob.reports.application.templates import TemplateEngine
    from mrbob.reports.application.exceptions import ReportGenerationError
except ImportError as e:
    logging.error(f"Failed to import required modules: {e}")
    raise


logger = logging.getLogger(__name__)


# Cache configuration
REPORT_CACHE_SIZE = 1000
REPORT_CACHE_TTL = 3600  # 1 hour


# Validation rules
REPORT_SIZE_LIMIT = 10 * 1024 * 1024  # 10MB
MAX_SECTIONS = 50
ALLOWED_CONTENT_TYPES = {'text/plain', 'text/html', 'application/json'}


@dataclass
class ReportEvent:
    """Event representing report state change."""
    event_id: UUID = UUID('00000000-0000-0000-0000-000000000000')
    event_type: str = ''
    report_id: UUID = None
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict = field(default_factory=dict)


@dataclass
class ReportValidationRule:
    """Rule for validating report content."""
    max_size: int
    max_sections: int
    allowed_content_types: Set[str]
    required_metadata: Set[str]


@dataclass
class GenerateReportCommand:
    """Command for initiating report generation."""
    agent_npn: str
    report_type: ReportType
    generated_by: str
    options: Dict[str, Any]


@dataclass
class AddSectionCommand:
    """Command for adding a report section."""
    report_id: UUID
    title: str
    content: Dict
    order: int
    template_key: str


@dataclass
class FinalizeReportCommand:
    """Command for finalizing report generation."""
    report_id: UUID
    output_path: str


@dataclass
class GetReportQuery:
    """Query for retrieving report by ID."""
    report_id: UUID


@dataclass
class GetAgentReportsQuery:
    """Query for retrieving agent reports."""
    agent_npn: str
    report_type: Optional[ReportType] = None
    limit: int = 10
    offset: int = 0


class UnitOfWork(Protocol):
    """Protocol defining unit of work pattern."""
    
    async def __aenter__(self) -> 'UnitOfWork':
        """Enter context with transaction."""
        ...
    
    async def __aexit__(self, exc_type, exc, tb) -> None:
        """Exit context and handle transaction."""
        ...
    
    async def commit(self) -> None:
        """Commit transaction."""
        ...
    
    async def rollback(self) -> None:
        """Rollback transaction."""
        ...


class ReportValidator:
    """Validates report content and structure."""
    
    def __init__(self):
        self.rules = ReportValidationRule(
            max_size=REPORT_SIZE_LIMIT,
            max_sections=MAX_SECTIONS,
            allowed_content_types=ALLOWED_CONTENT_TYPES,
            required_metadata={'generated_by', 'report_type'}
        )
    
    def validate_report(self, report: Report) -> Tuple[bool, Optional[str]]:
        """Validate report against rules."""
        try:
            # Check metadata
            missing = self.rules.required_metadata - set(
                report.metadata.__dict__.keys()
            )
            if missing:
                return False, f"Missing required metadata: {missing}"
            
            # Check sections
            if len(report.sections) > self.rules.max_sections:
                return False, f"Too many sections: {len(report.sections)}"
            
            # Check content types
            for section in report.sections:
                content_type = section.content.get('type', 'text/plain')
                if content_type not in self.rules.allowed_content_types:
                    return False, f"Invalid content type: {content_type}"
            
            # Check size
            size = len(json.dumps(report.to_dict()).encode())
            if size > self.rules.max_size:
                return False, f"Report too large: {size} bytes"
            
            return True, None
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"


class ReportCache:
    """Thread-safe report cache with TTL."""
    
    def __init__(self):
        self.cache = TTLCache(
            maxsize=REPORT_CACHE_SIZE,
            ttl=REPORT_CACHE_TTL
        )
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Report]:
        """Get report from cache."""
        with self.lock:
            return self.cache.get(key)
    
    def set(self, key: str, report: Report) -> None:
        """Set report in cache."""
        with self.lock:
            self.cache[key] = report
    
    def invalidate(self, key: str) -> None:
        """Invalidate cache entry."""
        with self.lock:
            self.cache.pop(key, None)


class ReportService:
    """
    Advanced report generation service implementing CQRS pattern.
    
    Features:
    - Command/Query separation
    - Asynchronous processing
    - Event sourcing
    - Caching strategy
    - Retry policies
    """
    
    def __init__(
        self,
        repository: ReportRepository,
        template_engine: TemplateEngine,
        event_publisher: EventPublisher,
        unit_of_work: UnitOfWork,
        validator: Optional[ReportValidator] = None
    ):
        """
        Initialize service with dependencies.
        
        Args:
            repository: Report repository implementation
            template_engine: Template rendering engine
            event_publisher: Event publishing service
            unit_of_work: Unit of work implementation
            validator: Optional validator implementation
        """
        self._repository = repository
        self._template_engine = template_engine
        self._event_publisher = event_publisher
        self._unit_of_work = unit_of_work
        self._validator = validator or ReportValidator()
        self._cache = ReportCache()
        self._command_handlers = {
            GenerateReportCommand: self._handle_generate_report,
            AddSectionCommand: self._handle_add_section,
            FinalizeReportCommand: self._handle_finalize_report
        }
        
        # Initialize event handlers
        self._event_handlers = {
            'ReportGenerated': self._handle_report_generated,
            'SectionAdded': self._handle_section_added,
            'ReportFinalized': self._handle_report_finalized
        }
    
    async def handle_command(self, command: Any) -> Any:
        """
        Handle command with appropriate handler.
        
        Args:
            command: Command to handle
            
        Returns:
            Any: Command result
            
        Raises:
            ReportGenerationError: If command handling fails
        """
        handler = self._command_handlers.get(type(command))
        if not handler:
            raise ReportGenerationError(
                f"Unknown command type: {type(command)}"
            )
        
        try:
            async with self._unit_of_work as uow:
                result = await handler(command)
                await uow.commit()
                return result
        except Exception as e:
            logger.error(f"Command handling failed: {str(e)}")
            raise ReportGenerationError(f"Command failed: {str(e)}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _handle_generate_report(
        self,
        command: GenerateReportCommand
    ) -> UUID:
        """
        Handle report generation command with retry logic.
        
        Args:
            command: Generation command
            
        Returns:
            UUID: Generated report ID
            
        Raises:
            ReportGenerationError: If generation fails
        """
        try:
            # Create report metadata with version
            metadata = ReportMetadata(
                generated_at=datetime.now(),
                generated_by=command.generated_by,
                report_type=command.report_type,
                version=1
            )
            
            # Initialize report with event
            report = Report(metadata=metadata)
            event = ReportEvent(
                event_type='ReportGenerated',
                report_id=report.id,
                data={'metadata': metadata.to_dict()},
                event_id=UUID('00000000-0000-0000-0000-000000000001'),
                timestamp=datetime.now()
            )
            report.events.append(event)
            
            # Validate report
            is_valid, error = self._validator.validate_report(report)
            if not is_valid:
                raise ReportGenerationError(f"Invalid report: {error}")
            
            # Save initial state
            await self._repository.save(report)
            
            # Publish event
            await self._event_publisher.publish(event)
            await self._handle_report_generated(event)
            
            return report.id
            
        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")
            raise ReportGenerationError(f"Generation failed: {str(e)}")
    
    async def _handle_add_section(
        self,
        command: AddSectionCommand
    ) -> None:
        """
        Handle section addition command.
        
        Args:
            command: Section command
            
        Raises:
            ReportGenerationError: If section addition fails
        """
        try:
            # Try cache first
            cache_key = f"report:{command.report_id}"
            report = self._cache.get(cache_key)
            
            if not report:
                # Get from repository
                report = await self._repository.get(command.report_id)
                if report:
                    self._cache.set(cache_key, report)
            if not report:
                raise ReportGenerationError(
                    f"Report not found: {command.report_id}"
                )
            
            # Create section event
            event = ReportEvent(
                event_type='SectionAdded',
                report_id=report.id,
                data={
                    'section': {
                        'title': command.title,
                        'content': command.content,
                        'order': command.order,
                        'template_key': command.template_key
                    }
                },
                event_id=UUID('00000000-0000-0000-0000-000000000002'),
                timestamp=datetime.now()
            )
            
            # Create and add section
            section = ReportSection(
                title=command.title,
                content=command.content,
                order=command.order,
                template_key=command.template_key
            )
            report.add_section(section)
            report.events.append(event)
            
            # Validate updated report
            is_valid, error = self._validator.validate_report(report)
            if not is_valid:
                raise ReportGenerationError(f"Invalid report: {error}")
            
            # Save updated state
            await self._repository.save(report)
            
            # Publish event and update cache
            await self._event_publisher.publish(event)
            await self._handle_section_added(event)
                
        except Exception as e:
            logger.error(f"Section addition failed: {str(e)}")
            raise ReportGenerationError(f"Section addition failed: {str(e)}")
    
    async def _handle_finalize_report(
        self,
        command: FinalizeReportCommand
    ) -> None:
        """
        Handle report finalization command.
        
        Args:
            command: Finalization command
            
        Raises:
            ReportGenerationError: If finalization fails
        """
        try:
            # Try cache first
            cache_key = f"report:{command.report_id}"
            report = self._cache.get(cache_key)
            
            if not report:
                # Get from repository
                report = await self._repository.get(command.report_id)
                if report:
                    self._cache.set(cache_key, report)
            if not report:
                raise ReportGenerationError(
                    f"Report not found: {command.report_id}"
                )
            
            # Create finalization event
            event = ReportEvent(
                event_type='ReportFinalized',
                report_id=report.id,
                data={'output_path': command.output_path},
                event_id=UUID('00000000-0000-0000-0000-000000000003'),
                timestamp=datetime.now()
            )
            
            # Generate report with template engine
            context = {
                'metadata': report.metadata,
                'sections': report.sections
            }
            
            await self._template_engine.render(
                template_name=report.metadata.report_type.value,
                context=context,
                output_path=command.output_path
            )
            
            # Finalize report and add event
            report.generate(command.output_path)
            report.events.append(event)
            
            # Save final state
            await self._repository.save(report)
            
            # Publish event and update cache
            await self._event_publisher.publish(event)
            await self._handle_report_finalized(event)
                
        except Exception as e:
            logger.error(f"Report finalization failed: {str(e)}")
            raise ReportGenerationError(f"Finalization failed: {str(e)}")
    
    async def _handle_report_generated(self, event: ReportEvent) -> None:
        """Handle report generated event."""
        # Cache the new report
        cache_key = f"report:{event.report_id}"
        report = await self._repository.get(event.report_id)
        if report:
            self._cache.set(cache_key, report)
    
    async def _handle_section_added(self, event: ReportEvent) -> None:
        """Handle section added event."""
        # Invalidate cache
        cache_key = f"report:{event.report_id}"
        self._cache.invalidate(cache_key)
    
    async def _handle_report_finalized(self, event: ReportEvent) -> None:
        """Handle report finalized event."""
        # Update cache with finalized report
        cache_key = f"report:{event.report_id}"
        report = await self._repository.get(event.report_id)
        if report:
            self._cache.set(cache_key, report)
    
    async def get_report(self, query: GetReportQuery) -> Optional[Report]:
        """
        Handle report retrieval query.
        
        Args:
            query: Report query
            
        Returns:
            Optional[Report]: Retrieved report
        """
        try:
            # Try cache first
            cache_key = f"report:{query.report_id}"
            report = self._cache.get(cache_key)
            
            if not report:
                # Get from repository and cache
                report = await self._repository.get(query.report_id)
                if report:
                    self._cache.set(cache_key, report)
            
            return report
        except Exception as e:
            logger.error(f"Report retrieval failed: {str(e)}")
            return None
    
    async def get_agent_reports(
        self,
        query: GetAgentReportsQuery
    ) -> List[Report]:
        """
        Handle agent reports query with pagination.
        
        Args:
            query: Agent reports query
            
        Returns:
            List[Report]: Retrieved reports
        """
        try:
            reports = await self._repository.get_by_metadata(
                report_type=query.report_type,
                generated_by=query.agent_npn
            )
            
            # Apply pagination
            start = query.offset
            end = start + query.limit
            return reports[start:end]
            
        except Exception as e:
            logger.error(f"Agent reports retrieval failed: {str(e)}")
            return []
