"""
Core policy processing module with optimized data handling and validation.

This module implements the core policy processing functionality with:
- Concurrent data processing
- Memory-efficient chunk processing
- Robust error handling
- Performance monitoring
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
from tenacity import retry, stop_after_attempt, wait_exponential

from .validation import DataValidator
from .status_resolver import StatusResolver
from .metrics import ProcessingMetrics
from .exceptions import PolicyProcessingError

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
    
    async def process_policy_data(
        self,
        file_path: str,
        required_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Process policy data from Excel file with optimized handling.
        
        Args:
            file_path: Path to Excel file
            required_columns: Optional list of required columns
            
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
            
            # Process in chunks
            chunks = pd.read_excel(
                file_path,
                chunksize=self.config.chunk_size
            )
            
            processed_data = []
            futures = []
            
            async with asyncio.Lock():
                for chunk in chunks:
                    # Submit chunk for processing
                    future = self.executor.submit(
                        self._process_chunk,
                        chunk,
                        required_columns
                    )
                    futures.append(future)
                    
                # Wait for all chunks
                for future in futures:
                    try:
                        result = future.result()
                        processed_data.extend(result)
                    except Exception as e:
                        logger.error(f"Chunk processing failed: {str(e)}")
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
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
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
            # Validate chunk data
            validation_result = self.validator.validate_chunk(
                chunk,
                required_columns
            )
            if not validation_result.is_valid:
                logger.warning(
                    f"Chunk validation failed: {validation_result.errors}"
                )
                self.metrics.validation_errors += len(validation_result.errors)
                return []
            
            processed_policies = []
            for _, row in chunk.iterrows():
                try:
                    # Process individual policy
                    policy_data = self._process_policy(row)
                    if policy_data:
                        processed_policies.append(policy_data)
                        self.metrics.processed_count += 1
                except Exception as e:
                    logger.error(f"Policy processing error: {str(e)}")
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
                current_status,
                None,
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
