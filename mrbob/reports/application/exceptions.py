"""Exception classes for reports module."""


class ReportGenerationError(Exception):
    """Raised when report generation fails."""
    pass


class TemplateError(Exception):
    """Raised when template processing fails."""
    pass


class ValidationError(Exception):
    """Raised when report validation fails."""
    pass
