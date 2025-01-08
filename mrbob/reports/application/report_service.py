"""
Report generation service implementing CQRS pattern with advanced architectural features.

Key patterns:
- Command/Query Responsibility Segregation (CQRS)
- Domain Events for state changes
- Unit of Work for transactional integrity
- Dependency Injection for loose coupling
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any, Protocol
from uuid import UUID
import logging
import asyncio
from functools import wraps
import json
from tenacity import retry, stop_after_attempt, wait_exponential

from ..domain.report_aggregate import (
    Report,
    ReportType,
    ReportMetadata,
    ReportSection,
    ReportRepository,
    ReportEvent
)
from .events import EventPublisher
from .templates import TemplateEngine
from .exceptions import ReportGenerationError

logger = logging.getLogger(__name__)

# Command Definitions
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

# Query Definitions
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

# Unit of Work Protocol
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
        unit_of_work: UnitOfWork
    ):
        """
        Initialize service with dependencies.
        
        Args:
            repository: Report repository implementation
            template_engine: Template rendering engine
            event_publisher: Event publishing service
            unit_of_work: Unit of work implementation
        """
        self._repository = repository
        self._template_engine = template_engine
        self._event_publisher = event_publisher
        self._unit_of_work = unit_of_work
        self._command_handlers = {
            GenerateReportCommand: self._handle_generate_report,
            AddSectionCommand: self._handle_add_section,
            FinalizeReportCommand: self._handle_finalize_report
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
            raise ReportGenerationError(f"Unknown command type: {type(command)}")
        
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
            # Create report metadata
            metadata = ReportMetadata(
                generated_at=datetime.now(),
                generated_by=command.generated_by,
                report_type=command.report_type
            )
            
            # Initialize report
            report = Report(metadata=metadata)
            
            # Save initial state
            await self._repository.save(report)
            
            # Publish events
            for event in report.events:
                await self._event_publisher.publish(event)
            
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
            # Get report
            report = await self._repository.get(command.report_id)
            if not report:
                raise ReportGenerationError(f"Report not found: {command.report_id}")
            
            # Create and add section
            section = ReportSection(
                title=command.title,
                content=command.content,
                order=command.order,
                template_key=command.template_key
            )
            report.add_section(section)
            
            # Save updated state
            await self._repository.save(report)
            
            # Publish events
            for event in report.events:
                await self._event_publisher.publish(event)
                
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
            # Get report
            report = await self._repository.get(command.report_id)
            if not report:
                raise ReportGenerationError(f"Report not found: {command.report_id}")
            
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
            
            # Finalize report
            report.generate(command.output_path)
            
            # Save final state
            await self._repository.save(report)
            
            # Publish events
            for event in report.events:
                await self._event_publisher.publish(event)
                
        except Exception as e:
            logger.error(f"Report finalization failed: {str(e)}")
            raise ReportGenerationError(f"Finalization failed: {str(e)}")
    
    async def get_report(self, query: GetReportQuery) -> Optional[Report]:
        """
        Handle report retrieval query.
        
        Args:
            query: Report query
            
        Returns:
            Optional[Report]: Retrieved report
        """
        try:
            return await self._repository.get(query.report_id)
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