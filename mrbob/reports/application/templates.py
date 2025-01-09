"""
Template engine module for report generation.

This module provides template rendering functionality with:
- Multiple template format support
- Context-based rendering
- Output format flexibility
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class TemplateEngine:
    """
    Template engine for report generation.
    
    Features:
    - Template caching
    - Multiple output formats
    - Context validation
    """
    
    def __init__(self, template_dir: Optional[str] = None):
        """
        Initialize template engine.
        
        Args:
            template_dir: Optional template directory path
        """
        self._template_dir = template_dir or "templates"
        self._template_cache: Dict[str, str] = {}
    
    async def render(
        self,
        template_name: str,
        context: Dict[str, Any],
        output_path: str
    ) -> None:
        """
        Render template with context to output path.
        
        Args:
            template_name: Name of template to render
            context: Template context data
            output_path: Path to write rendered output
            
        Raises:
            ValueError: If template or context is invalid
        """
        try:
            # Validate context
            if not self._validate_context(context):
                raise ValueError("Invalid template context")
            
            # Get template content
            template = await self._get_template(template_name)
            
            # Render template with context
            rendered = self._render_template(template, context)
            
            # Write output
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                f.write(rendered)
                
        except Exception as e:
            logger.error(f"Template rendering failed: {str(e)}")
            raise
    
    async def _get_template(self, template_name: str) -> str:
        """
        Get template content with caching.
        
        Args:
            template_name: Template name/path
            
        Returns:
            str: Template content
            
        Raises:
            ValueError: If template not found
        """
        # Check cache first
        if template_name in self._template_cache:
            return self._template_cache[template_name]
        
        # Load template
        template_path = Path(self._template_dir) / f"{template_name}.html"
        if not template_path.exists():
            raise ValueError(f"Template not found: {template_name}")
        
        with open(template_path) as f:
            template = f.read()
        
        # Cache template
        self._template_cache[template_name] = template
        
        return template
    
    def _render_template(
        self,
        template: str,
        context: Dict[str, Any]
    ) -> str:
        """
        Render template with context.
        
        Args:
            template: Template content
            context: Template context
            
        Returns:
            str: Rendered content
        """
        # For now, just return JSON representation
        # In practice, you would use a proper template engine
        return json.dumps(context, indent=2)
    
    def _validate_context(self, context: Dict[str, Any]) -> bool:
        """
        Validate template context.
        
        Args:
            context: Context to validate
            
        Returns:
            bool: True if valid
        """
        required_keys = {'metadata', 'sections'}
        return all(key in context for key in required_keys)
