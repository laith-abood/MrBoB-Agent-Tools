"""
Data validation module for policy processing.

This module provides validation functionality for policy data including:
- DataFrame chunk validation
- Required column validation
- Data type and format validation
"""

from typing import List, Optional
from dataclasses import dataclass
import pandas as pd
import logging

logger = logging.getLogger(__name__)



@dataclass
class ValidationResult:
    """Results of data validation."""
    is_valid: bool
    errors: List[str]


class DataValidator:
    """
    Validates policy data chunks and individual records.
    
    Features:
    - Column presence validation
    - Data type validation
    - Format validation for specific fields
    """
    
    def __init__(self):
        """Initialize validator with default settings."""
        self._required_fields = {
            'Policy Number': str,
            'Status': str,
            'Carrier': str,
            'Agent NPN': str,
            'Effective Date': 'datetime64[ns]'
        }
    
    def validate_chunk(
        self,
        chunk: pd.DataFrame,
        required_columns: Optional[List[str]] = None
    ) -> ValidationResult:
        """
        Validate a chunk of policy data.
        
        Args:
            chunk: DataFrame chunk to validate
            required_columns: Optional override of required columns
            
        Returns:
            ValidationResult: Validation results with any errors
        """
        errors = []
        
        try:
            # Validate required columns
            columns_to_check = (
                required_columns or self._required_fields.keys()
            )
            missing_columns = [
                col for col in columns_to_check 
                if col not in chunk.columns
            ]
            
            if missing_columns:
                errors.append(
                    f"Missing required columns: {', '.join(missing_columns)}"
                )
            
            # Validate data types if all required columns present
            if not missing_columns:
                for field, expected_type in self._required_fields.items():
                    try:
                        if expected_type == str:
                            # Convert to string, replacing NaN with empty string
                            chunk[field] = chunk[field].fillna('').astype(str)
                        elif expected_type == 'datetime64[ns]':
                            # Convert to datetime, allowing NaT for missing values
                            chunk[field] = pd.to_datetime(
                                chunk[field], errors='coerce'
                            )
                            
                            # Check for failed conversions
                            invalid_dates = chunk[field].isna().sum()
                            if invalid_dates > 0:
                                errors.append(
                                    f"Found {invalid_dates} invalid dates in "
                                    f"{field}"
                                )
                    
                    except Exception as e:
                        errors.append(
                            f"Error converting {field} to {expected_type}: {str(e)}"
                        )
            
            # Validate policy numbers are unique
            if 'Policy Number' in chunk.columns:
                duplicates = chunk['Policy Number'].duplicated()
                if duplicates.any():
                    dup_count = duplicates.sum()
                    errors.append(
                        f"Found {dup_count} duplicate policy number(s)"
                    )
            
            # Validate status values
            if 'Status' in chunk.columns:
                invalid_statuses = ~chunk['Status'].str.strip().isin([
                    'Active',
                    'Pending',
                    'Terminated',
                    'Cancelled'
                ])
                if invalid_statuses.any():
                    invalid_count = invalid_statuses.sum()
                    errors.append(
                        f"Found {invalid_count} invalid status value(s)"
                    )
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors
            )
            
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation failed: {str(e)}"]
            )
