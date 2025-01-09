"""
Event publishing module for report generation events.

This module provides event publishing functionality with:
- Asynchronous event handling
- Event persistence
- Event replay capabilities
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)


class Event:
    """Base event class."""
    
    def __init__(self, event_type: str, data: Dict[str, Any]):
        self.event_type = event_type
        self.data = data
        self.timestamp = datetime.now()


class EventHandler(ABC):
    """Abstract base class for event handlers."""
    
    @abstractmethod
    async def handle(self, event: Event) -> None:
        """Handle an event."""
        pass


class EventPublisher:
    """
    Event publisher implementing pub/sub pattern.
    
    Features:
    - Asynchronous event publishing
    - Multiple subscriber support
    - Event persistence
    """
    
    def __init__(self):
        """Initialize publisher with empty handler registry."""
        self._handlers: Dict[str, List[EventHandler]] = {}
        self._event_store: List[Event] = []
    
    def register_handler(
        self,
        event_type: str,
        handler: EventHandler
    ) -> None:
        """
        Register handler for event type.
        
        Args:
            event_type: Type of event to handle
            handler: Event handler implementation
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
    
    async def publish(self, event: Event) -> None:
        """
        Publish event to registered handlers.
        
        Args:
            event: Event to publish
        """
        try:
            # Store event
            self._event_store.append(event)
            
            # Get handlers for event type
            handlers = self._handlers.get(event.event_type, [])
            
            # Execute handlers concurrently
            tasks = [
                handler.handle(event)
                for handler in handlers
            ]
            
            if tasks:
                await asyncio.gather(*tasks)
                
        except Exception as e:
            logger.error(f"Event publishing failed: {str(e)}")
            raise
    
    def get_events(
        self,
        event_type: Optional[str] = None,
        start_time: Optional[datetime] = None
    ) -> List[Event]:
        """
        Get stored events with optional filtering.
        
        Args:
            event_type: Optional event type filter
            start_time: Optional start time filter
            
        Returns:
            List[Event]: Filtered events
        """
        events = self._event_store
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
            
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
            
        return events
