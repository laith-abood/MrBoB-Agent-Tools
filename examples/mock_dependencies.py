"""Mock implementations of dependencies for example script."""

from typing import Dict, Any, Optional
from uuid import UUID

from mrbob.reports.application.report_service import (
    ReportRepository,
    EventPublisher,
    TemplateEngine,
    UnitOfWork,
    Report
)


class MockReportRepository(ReportRepository):
    """Mock implementation of report repository."""
    
    async def save(self, report: Report) -> None:
        """Mock save operation."""
        pass
    
    async def get(self, report_id: UUID) -> Optional[Report]:
        """Mock get operation."""
        return None
    
    async def get_by_metadata(
        self,
        report_type: Optional[str] = None,
        generated_by: Optional[str] = None
    ) -> list[Report]:
        """Mock metadata query."""
        return []


class MockEventPublisher(EventPublisher):
    """Mock implementation of event publisher."""
    
    async def publish(self, event: Dict[str, Any]) -> None:
        """Mock publish operation."""
        pass


class MockTemplateEngine(TemplateEngine):
    """Mock implementation of template engine."""
    
    async def render(
        self,
        template_name: str,
        context: Dict[str, Any],
        output_path: str
    ) -> None:
        """Mock render operation."""
        pass


class MockUnitOfWork(UnitOfWork):
    """Mock implementation of unit of work."""
    
    async def __aenter__(self) -> 'MockUnitOfWork':
        """Enter context."""
        return self
    
    async def __aexit__(self, exc_type, exc, tb) -> None:
        """Exit context."""
        pass
    
    async def commit(self) -> None:
        """Mock commit operation."""
        pass
    
    async def rollback(self) -> None:
        """Mock rollback operation."""
        pass
