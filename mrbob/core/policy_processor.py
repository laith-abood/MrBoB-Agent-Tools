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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
from cachetools import TTLCache

# Local imports
from mrbob.core.validation import DataValidator
from mrbob.core.status_resolver import StatusResolver, StatusSource
from mrbob.core.metrics import ProcessingMetrics
from mrbob.core.exceptions import PolicyProcessingError

logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Configuration for policy processing."""
    chunk_size: int = 1000
    max_workers: int = 4
    retry_attempts: int = 3
    cache_enabled: bool = True
    validation_level: str = "strict"


class PolicyProcessor:
    """
    Optimized policy data processor with concurrent processing capabilities.
    
    Features:
    - Chunk-based processing for memory efficiency
    - Concurrent processing with ThreadPoolExecutor
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
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
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
        Process policy data from Excel file with optimized handling.
        
        Args:
            file_path: Path to Excel file
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
            chunks = pd.read_excel(
                file_path,
                chunksize=batch_size or self.config.chunk_size,
                engine='openpyxl'  # More memory efficient for large files
            )
            
            processed_data = []
            futures = []
            
            async with asyncio.Lock():
                for chunk_index, chunk in enumerate(chunks):
                    # Submit chunk for processing with retry
                    future = self.executor.submit(
                        self._process_chunk_with_retry,
                        chunk,
                        required_columns,
                        chunk_index
                    )
                    futures.append(future)
                    
                # Process chunks with progress tracking
                total_chunks = len(futures)
                for chunk_index, future in enumerate(futures, 1):
                    try:
                        result = future.result()
                        processed_data.extend(result)
                        logger.info(
                            f"Processed chunk {chunk_index}/{total_chunks}"
                        )
                    except Exception as e:
                        error_context = {
                            'chunk_index': chunk_index,
                            'total_chunks': total_chunks,
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
    
    def _process_chunk_with_retry(
        self,
        chunk: pd.DataFrame,
        required_columns: Optional[List[str]],
        chunk_index: int
    ) -> List[Dict[str, Any]]:
        """
        Process chunk with exponential backoff retry.
        
        Args:
            chunk: DataFrame chunk
            required_columns: Required columns
            chunk_index: Index of current chunk
            
        Returns:
            List[Dict[str, Any]]: Processed chunk data
        """
        attempt = 0
        delay = self._retry_config['initial_delay']
        
        while attempt < self._retry_config['max_attempts']:
            try:
                return self._process_chunk(chunk, required_columns)
            except Exception as e:
                attempt += 1
                if attempt == self._retry_config['max_attempts']:
                    raise
                
                error_context = {
                    'chunk_index': chunk_index,
                    'attempt': attempt,
                    'max_attempts': self._retry_config['max_attempts'],
                    'delay': delay,
                    'error': str(e)
                }
                logger.warning(f"Retrying chunk: {json.dumps(error_context)}")
                
                time.sleep(delay)
                delay = min(
                    delay * self._retry_config['backoff_factor'],
                    self._retry_config['max_delay']
                )
    
    def _process_chunk(
        self,
        chunk: pd.DataFrame,
        required_columns: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
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
            
            processed_policies = []
            
            # Process policies in parallel for better performance
            for _, row in chunk.iterrows():
                try:
                    future = self.executor.submit(self._process_policy, row)
                    policy_data = future.result()
                    if policy_data:
                        processed_policies.append(policy_data)
                        self.metrics.processed_count += 1
                except Exception as e:
                    error_context = {
                        'policy_number': str(row.get('Policy Number', 'Unknown')),
                        'error_type': type(e).__name__,
                        'error_details': str(e)
                    }
                    logger.error(
                        f"Policy processing error: {json.dumps(error_context)}"
                    )
                    self.metrics.error_count += 1
            
            return processed_policies
            
        except Exception as e:
            logger.error(f"Chunk processing error: {str(e)}")
            raise
    
    def _process_policy(self, row: pd.Series) -> Optional[Dict[str, Any]]:
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
            
            resolved_status, transition = self.status_resolver.resolve_status(
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
                'metrics': self.metrics.to_dict(),
                'policies': []
            }
