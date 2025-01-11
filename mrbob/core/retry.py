"""
Retry logic module for processing chunks with exponential backoff.

This module provides retry functionality for processing chunks including:
- Exponential backoff
- Retry configuration
- Compatibility with PolicyProcessor class
"""

import asyncio
import logging
from typing import Optional, List, Dict, Any
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class RetryConfig:
    """Configuration for retry logic."""
    def __init__(
        self,
        max_attempts: int = 5,
        initial_delay: int = 1,
        max_delay: int = 60,
        backoff_factor: int = 2
    ):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
async def process_chunk_with_retry(
    processor: Any,
    chunk: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    chunk_index: int = 0
) -> List[Dict[str, Any]]:
    """
    Process a chunk of policy data with retry logic.
    
    Args:
        processor: PolicyProcessor instance
        chunk: DataFrame chunk
        required_columns: Optional required columns
        chunk_index: Index of the chunk
        
    Returns:
        List[Dict[str, Any]]: Processed chunk data
    """
    try:
        # Process chunk using PolicyProcessor's _process_chunk method
        result = await processor._process_chunk(chunk, required_columns)
        logger.info(f"Successfully processed chunk {chunk_index}")
        return result
    except Exception as e:
        logger.error(f"Error processing chunk {chunk_index}: {str(e)}")
        raise
