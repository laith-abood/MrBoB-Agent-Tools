"""Exception classes for core module."""


class PolicyProcessingError(Exception):
    """Raised when policy processing fails."""
    pass


class ValidationError(Exception):
    """Raised when data validation fails."""
    pass


class StatusResolutionError(Exception):
    """Raised when status resolution fails."""
    pass
