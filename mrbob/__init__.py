"""MrBoB Agent Tools package initialization."""

from mrbob.core.exceptions import (
    PolicyProcessingError,
    ValidationError,
    StatusResolutionError
)
from mrbob.reports.application.exceptions import (
    ReportGenerationError,
    TemplateError
)
from mrbob.core.policy_processor import PolicyProcessor
from mrbob.reports.application.report_service import (
    ReportService,
    GenerateReportCommand,
    AddSectionCommand,
    FinalizeReportCommand,
    ReportType
)
from mrbob.analytics.performance_analyzer import PerformanceAnalyzer

__version__ = "0.1.0"

__all__ = [
    # Core
    'PolicyProcessor',
    'PolicyProcessingError',
    'ValidationError',
    'StatusResolutionError',
    
    # Reports
    'ReportService',
    'GenerateReportCommand',
    'AddSectionCommand',
    'FinalizeReportCommand',
    'ReportType',
    'ReportGenerationError',
    'TemplateError',
    
    # Analytics
    'PerformanceAnalyzer'
]
