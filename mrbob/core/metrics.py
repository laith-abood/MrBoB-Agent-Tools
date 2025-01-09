"""
Metrics tracking module for policy processing.

This module provides metrics collection and reporting functionality including:
- Processing counts
- Error tracking
- Performance metrics
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any
import time


@dataclass
class ProcessingMetrics:
    """Tracks metrics for policy processing operations."""
    
    processed_count: int = 0
    error_count: int = 0
    validation_errors: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    
    def __post_init__(self):
        """Initialize timing on creation."""
        self.start_time = time.time()
    
    def complete(self):
        """Mark processing as complete and record end time."""
        self.end_time = time.time()
    
    @property
    def processing_time(self) -> float:
        """Calculate total processing time in seconds."""
        if self.end_time == 0.0:
            return time.time() - self.start_time
        return self.end_time - self.start_time
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.processed_count == 0:
            return 0.0
        return (
            (self.processed_count - self.error_count) / 
            self.processed_count * 100
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        base_dict = asdict(self)
        # Add computed metrics
        base_dict.update({
            'processing_time_seconds': round(self.processing_time, 2),
            'success_rate_percent': round(self.success_rate, 2)
        })
        return base_dict
