"""
Core policy processing module with optimized data handling and validation.

This module implements the core policy processing functionality with:
- Concurrent data processing
- Memory-efficient chunk processing
- Robust error handling
- Performance monitoring
"""

from typing import Dict, List, Optional, Any
import hashlib
import time
import pandas as pd
from datetime import datetime
import logging
from dataclasses import dataclass
import asyncio
from pathlib import Path
import json
from cachetools import TTLCache

# Local imports
from mrbob.core.validation import DataValidator
from mrbob.core.status_resolver import StatusResolver, StatusSource
from mrbob.core.metrics import ProcessingMetrics
from mrbob.core.exceptions import PolicyProcessingError
from mrbob.core.retry import process_chunk_with_retry

logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Configuration for policy processing."""
    chunk_size: int = 1000
    retry_attempts: int = 3
    cache_enabled: bool = True
    validation_level: str = "strict"


class PolicyProcessor:
    """
    Optimized policy data processor with concurrent processing capabilities.
    
    Features:
    - Chunk-based processing for memory efficiency
    - Concurrent processing with asyncio
    - Robust error handling and retry logic
    - Performance monitoring and metrics
    """
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        """
        Initialize processor with configuration.
        
        Args:
            config: Processing configuration parameters
        """
        self.config = config or ProcessingConfig()
        self.validator = DataValidator()
        self.status_resolver = StatusResolver()
        self.metrics = ProcessingMetrics()
        self._validation_cache = TTLCache(maxsize=10000, ttl=3600)  # 1 hour TTL
        self._retry_config = {
            'max_attempts': 5,
            'initial_delay': 1,
            'max_delay': 60,
            'backoff_factor': 2
        }
    
    async def process_policy_data(
        self,
        file_path: str,
        required_columns: Optional[List[str]] = None,
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process policy data from CSV file with optimized handling.
        
        Args:
            file_path: Path to CSV file
            required_columns: Optional list of required columns
            batch_size: Optional chunk size override
            
        Returns:
            Dict[str, Any]: Processed policy data with metrics
            
        Raises:
            PolicyProcessingError: If processing fails
        """
        try:
            # Validate file
            path = Path(file_path)
            if not path.exists():
                raise PolicyProcessingError(f"File not found: {file_path}")
            
            # Stream data in memory-efficient chunks
            chunks = pd.read_csv(
                file_path,
                chunksize=batch_size or self.config.chunk_size
            )
            
            processed_data = []
            
            # Process chunks with progress tracking
            for chunk_index, chunk in enumerate(chunks, 1):
                try:
                    result = await process_chunk_with_retry(
                        self,
                        chunk,
                        required_columns,
                        chunk_index
                    )
                    processed_data.extend(result)
                    logger.info(f"Processed chunk {chunk_index}")
                except Exception as e:
                    error_context = {
                        'chunk_index': chunk_index,
                        'error_type': type(e).__name__,
                        'error_details': str(e)
                    }
                    logger.error(
                        f"Chunk processing failed: {json.dumps(error_context)}"
                    )
                    self.metrics.error_count += 1
            
            # Aggregate results
            aggregated_data = self._aggregate_results(processed_data)
            
            return {
                'data': aggregated_data,
                'metrics': self.metrics.to_dict()
            }
            
        except Exception as e:
            logger.error(f"Policy processing failed: {str(e)}")
            raise PolicyProcessingError(f"Processing failed: {str(e)}")
    
    def _get_chunk_cache_key(self, chunk: pd.DataFrame) -> str:
        """
        Generate cache key for chunk validation results.
        
        Args:
            chunk: DataFrame chunk
            
        Returns:
            str: Cache key
        """
        content = pd.util.hash_pandas_object(chunk).values.tobytes()
        return f"chunk_validation_{hashlib.md5(content).hexdigest()}"
    
    async def _process_chunk(
        self,
        chunk: pd.DataFrame,
        required_columns: Optional[List[str]] = None
    ) -> List<Dict[str, Any]]:
        """
        Process a chunk of policy data with retry logic.
        
        Args:
            chunk: DataFrame chunk
            required_columns: Optional required columns
            
        Returns:
            List[Dict[str, Any]]: Processed chunk data
        """
        try:
            # Check validation cache
            cache_key = self._get_chunk_cache_key(chunk)
            validation_result = self._validation_cache.get(cache_key)
            
            if validation_result is None:
                # Validate chunk data if not in cache
                validation_result = self.validator.validate_chunk(
                    chunk,
                    required_columns
                )
                self._validation_cache[cache_key] = validation_result
            
            if not validation_result.is_valid:
                logger.warning(
                    f"Chunk validation failed: {validation_result.errors}"
                )
                self.metrics.validation_errors += len(validation_result.errors)
                return []
            
            # Process policies in parallel for better performance
            tasks = []
            for _, row in chunk.iterrows():
                tasks.append(self._process_policy(row))
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            processed_policies = []
            for result in results:
                if isinstance(result, Exception):
                    self.metrics.error_count += 1
                    logger.error(f"Policy processing error: {str(result)}")
                    continue
                if result:
                    processed_policies.append(result)
                    self.metrics.processed_count += 1
            
            return processed_policies
            
        except Exception as e:
            logger.error(f"Chunk processing error: {str(e)}")
            raise
    
    async def _process_policy(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """
        Process individual policy data with status resolution.
        
        Args:
            row: Policy data row
            
        Returns:
            Optional[Dict[str, Any]]: Processed policy data
        """
        try:
            # Extract policy data
            policy_number = str(row.get('Policy Number', ''))
            if not policy_number:
                return None
            
            # Resolve status
            current_status = str(row.get('Status', ''))
            effective_date = pd.to_datetime(row.get('Effective Date'))
            
            resolved_status, transition = await self.status_resolver.resolve_status(
                policy_number,
                current_status,
                StatusSource.SYSTEM,
                effective_date
            )
            
            # Create policy record
            policy_data = {
                'policy_number': policy_number,
                'status': resolved_status,
                'carrier': str(row.get('Carrier', '')),
                'agent_npn': str(row.get('Agent NPN', '')),
                'effective_date': effective_date.isoformat(),
                'processed_date': datetime.now().isoformat()
            }
            
            if transition:
                policy_data['status_transition'] = transition.to_dict()
            
            return policy_data
            
        except Exception as e:
            logger.error(f"Error processing policy: {str(e)}")
            return None
    
    def _aggregate_results(
        self,
        processed_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate processed policy data with metrics.
        
        Args:
            processed_data: List of processed policies
            
        Returns:
            Dict[str, Any]: Aggregated results
        """
        try:
            # Convert to DataFrame for efficient aggregation
            df = pd.DataFrame(processed_data)
            
            aggregated = {
                'total_policies': len(df),
                'status_breakdown': df['status'].value_counts().to_dict(),
                'carrier_breakdown': df['carrier'].value_counts().to_dict(),
                'agent_breakdown': df['agent_npn'].value_counts().to_dict(),
                'metrics': {
                    'processed_count': self.metrics.processed_count,
                    'error_count': self.metrics.error_count,
                    'validation_errors': self.metrics.validation_errors
                },
                'policies': processed_data
            }
            
            return aggregated
            
        except Exception as e:
            logger.error(f"Aggregation error: {str(e)}")
            return {
                'total_policies': 0,
                'status_breakdown': {},
                'carrier_breakdown': {},
                'agent_breakdown': {},
                'metrics': self.metrics.to_dict(),
                'policies': []
            }
