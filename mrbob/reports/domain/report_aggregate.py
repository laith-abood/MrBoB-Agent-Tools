"""
Report domain model implementing DDD principles with CQRS pattern.

Key architectural patterns:
- Aggregate Root pattern for report consistency
- Value Objects for immutable report components
- Domain Events for state changes
- Repository pattern for persistence
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set
from uuid import UUID, uuid4
import json
from abc import ABC, abstractmethod

# Value Objects
class ReportType(Enum):
    """Immutable report type classification."""
    AGENT_SUMMARY = "agent_summary"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    CARRIER_BREAKDOWN = "carrier_breakdown"
    COMPLIANCE_AUDIT = "compliance_audit"

@dataclass(frozen=True)
class ReportMetadata:
    """Immutable report metadata value object."""
    version: str = "1.0"
    generated_at: datetime = field(default_factory=datetime.now)
    generated_by: str = ""
    report_type: ReportType = ReportType.AGENT_SUMMARY

@dataclass(frozen=True)
class ReportSection:
    """Immutable report section value object."""
    title: str
    content: Dict
    order: int
    template_key: str

# Domain Events
class ReportEvent(ABC):
    """Base class for report domain events."""
    @abstractmethod
    def event_type(self) -> str:
        pass

@dataclass
class ReportCreatedEvent(ReportEvent):
    """Event emitted when a new report is created."""
    metadata: ReportMetadata
    report_id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def event_type(self) -> str:
        return "report.created"

@dataclass
class SectionAddedEvent(ReportEvent):
    """Event emitted when a section is added to a report."""
    section: ReportSection
    report_id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def event_type(self) -> str:
        return "report.section_added"

@dataclass
class ReportGeneratedEvent(ReportEvent):
    """Event emitted when a report is fully generated."""
    output_path: str
    report_id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def event_type(self) -> str:
        return "report.generated"

# Aggregate Root
class Report:
    """
    Report aggregate root implementing domain logic and invariants.
    
    Features:
    - Strong consistency boundaries
    - Event-sourced state changes
    - Business rule enforcement
    """
    
    def __init__(
        self,
        metadata: ReportMetadata,
        report_id: Optional[UUID] = None
    ):
        """Initialize report aggregate."""
        self._id = report_id or uuid4()
        self._metadata = metadata
        self._sections: List[ReportSection] = []
        self._events: List[ReportEvent] = []
        self._status = "draft"
        self._validation_errors: Set[str] = set()
        
        # Record creation event
        self._record_event(
            ReportCreatedEvent(
                report_id=self._id,
                metadata=metadata
            )
        )
    
    @property
    def id(self) -> UUID:
        """Get report identifier."""
        return self._id
    
    @property
    def metadata(self) -> ReportMetadata:
        """Get report metadata."""
        return self._metadata
    
    @property
    def sections(self) -> List[ReportSection]:
        """Get ordered report sections."""
        return sorted(self._sections, key=lambda s: s.order)
    
    @property
    def events(self) -> List[ReportEvent]:
        """Get recorded domain events."""
        return self._events.copy()
    
    @property
    def status(self) -> str:
        """Get report status."""
        return self._status
    
    @property
    def validation_errors(self) -> Set[str]:
        """Get validation errors."""
        return self._validation_errors.copy()
    
    def add_section(self, section: ReportSection) -> None:
        """
        Add section to report with invariant checks.
        
        Args:
            section: Report section to add
            
        Raises:
            ValueError: If section is invalid
        """
        # Validate section
        if not self._validate_section(section):
            raise ValueError(
                f"Invalid section: {', '.join(self._validation_errors)}"
            )
        
        # Check for duplicate order
        if any(s.order == section.order for s in self._sections):
            raise ValueError(f"Duplicate section order: {section.order}")
        
        # Add section and record event
        self._sections.append(section)
        self._record_event(
            SectionAddedEvent(
                report_id=self._id,
                section=section
            )
        )
    
    def generate(self, output_path: str) -> None:
        """
        Generate final report with validation.
        
        Args:
            output_path: Path for generated report
            
        Raises:
            ValueError: If report is invalid
        """
        # Validate report state
        if not self._validate_report():
            raise ValueError(
                f"Invalid report: {', '.join(self._validation_errors)}"
            )
        
        # Update status and record event
        self._status = "generated"
        self._record_event(
            ReportGeneratedEvent(
                report_id=self._id,
                output_path=output_path
            )
        )
    
    def to_dict(self) -> Dict:
        """Convert report to dictionary representation."""
        return {
            "id": str(self._id),
            "metadata": {
                "generated_at": self._metadata.generated_at.isoformat(),
                "generated_by": self._metadata.generated_by,
                "report_type": self._metadata.report_type.value,
                "version": self._metadata.version
            },
            "sections": [
                {
                    "title": s.title,
                    "content": s.content,
                    "order": s.order,
                    "template_key": s.template_key
                }
                for s in self.sections
            ],
            "status": self._status
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Report":
        """Reconstruct report from dictionary."""
        metadata = ReportMetadata(
            generated_at=datetime.fromisoformat(
                data["metadata"]["generated_at"]
            ),
            generated_by=data["metadata"]["generated_by"],
            report_type=ReportType(data["metadata"]["report_type"]),
            version=data["metadata"]["version"]
        )
        
        report = cls(
            metadata=metadata,
            report_id=UUID(data["id"])
        )
        
        for section_data in data["sections"]:
            report.add_section(
                ReportSection(
                    title=section_data["title"],
                    content=section_data["content"],
                    order=section_data["order"],
                    template_key=section_data["template_key"]
                )
            )
        
        report._status = data["status"]
        return report
    
    def _record_event(self, event: ReportEvent) -> None:
        """Record domain event."""
        self._events.append(event)
    
    def _validate_section(self, section: ReportSection) -> bool:
        """
        Validate section against business rules.
        
        Args:
            section: Section to validate
            
        Returns:
            bool: True if valid
        """
        self._validation_errors.clear()
        
        if not section.title:
            self._validation_errors.add("Section title required")
        
        if not section.content:
            self._validation_errors.add("Section content required")
        
        if section.order < 0:
            self._validation_errors.add("Invalid section order")
        
        if not section.template_key:
            self._validation_errors.add("Template key required")
        
        return len(self._validation_errors) == 0
    
    def _validate_report(self) -> bool:
        """
        Validate complete report state.
        
        Returns:
            bool: True if valid
        """
        self._validation_errors.clear()
        
        if not self._sections:
            self._validation_errors.add("Report must have at least one section")
        
        if len(set(s.order for s in self._sections)) != len(self._sections):
            self._validation_errors.add("Duplicate section orders")
        
        return len(self._validation_errors) == 0

# Repository Interface
class ReportRepository(ABC):
    """Abstract repository for report persistence."""
    
    @abstractmethod
    async def save(self, report: Report) -> None:
        """Save report to persistence store."""
        pass
    
    @abstractmethod
    async def get(self, report_id: UUID) -> Optional[Report]:
        """Retrieve report by ID."""
        pass
    
    @abstractmethod
    async def get_by_metadata(
        self,
        report_type: ReportType,
        generated_by: str
    ) -> List[Report]:
        """Retrieve reports by metadata criteria."""
        pass
