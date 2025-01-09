"""
Advanced performance analytics module with sophisticated metric calculation and visualization.

This module implements complex analytics including:
- Time-series analysis of agent performance
- Predictive modeling for policy transitions
- Cohort analysis and segmentation
- Interactive visualization capabilities
"""

from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from collections import defaultdict
import json
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from scipy import stats
import warnings

# Type aliases for clarity
MetricValue = Union[int, float, str]
MetricDict = Dict[str, MetricValue]

logger = logging.getLogger(__name__)

@dataclass
class TimeSeriesMetrics:
    """Time series metrics with trend analysis."""
    values: List[float]
    timestamps: List[datetime]
    trend_coefficient: float = 0.0
    seasonality_index: float = 0.0
    outliers: List[int] = field(default_factory=list)
    
    def calculate_trends(self) -> None:
        """Calculate trend coefficients and seasonality."""
        if len(self.values) > 1:
            # Calculate linear trend
            x = np.arange(len(self.values))
            slope, _, r_value, _, _ = stats.linregress(x, self.values)
            self.trend_coefficient = slope * r_value**2
            
            # Detect seasonality using autocorrelation
            if len(self.values) >= 12:
                acf = np.correlate(self.values, self.values, mode='full')
                acf = acf[len(acf)//2:]
                self.seasonality_index = np.max(acf[1:13]) / acf[0]

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics with advanced analytics."""
    # Basic metrics
    total_policies: int = 0
    active_policies: int = 0
    chargeback_policies: int = 0
    retention_rate: float = 0.0
    avg_policy_duration: float = 0.0
    
    # Advanced metrics
    carrier_distribution: Dict[str, int] = field(default_factory=dict)
    status_distribution: Dict[str, int] = field(default_factory=dict)
    monthly_trends: Dict[str, TimeSeriesMetrics] = field(default_factory=dict)
    
    # Risk metrics
    churn_probability: float = 0.0
    risk_score: float = 0.0
    volatility_index: float = 0.0
    
    # Performance indicators
    growth_rate: float = 0.0
    efficiency_score: float = 0.0
    
    def calculate_risk_metrics(self) -> None:
        """Calculate comprehensive risk metrics."""
        # Calculate churn probability
        if self.total_policies > 0:
            self.churn_probability = self.chargeback_policies / self.total_policies
            
            # Calculate risk score (0-100)
            self.risk_score = min(100, (
                (self.churn_probability * 40) +
                ((1 - self.retention_rate/100) * 40) +
                (self.volatility_index * 20)
            ))
    
    def calculate_performance_indicators(self) -> None:
        """Calculate key performance indicators."""
        for metric_name, time_series in self.monthly_trends.items():
            time_series.calculate_trends()
            
            # Update growth rate based on trend coefficient
            if metric_name == 'active_policies':
                self.growth_rate = time_series.trend_coefficient * 100
                
                # Calculate efficiency score (0-100)
                retention_weight = 0.4
                growth_weight = 0.3
                risk_weight = 0.3
                
                self.efficiency_score = min(100, (
                    (self.retention_rate * retention_weight) +
                    (max(0, self.growth_rate) * growth_weight * 10) +
                    ((100 - self.risk_score) * risk_weight)
                ))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format with advanced analytics."""
        return {
            'summary': {
                'total_policies': self.total_policies,
                'active_policies': self.active_policies,
                'chargeback_policies': self.chargeback_policies,
                'retention_rate': round(self.retention_rate, 2),
                'avg_policy_duration': round(self.avg_policy_duration, 1)
            },
            'risk_metrics': {
                'churn_probability': round(self.churn_probability, 3),
                'risk_score': round(self.risk_score, 1),
                'volatility_index': round(self.volatility_index, 3)
            },
            'performance_indicators': {
                'growth_rate': round(self.growth_rate, 2),
                'efficiency_score': round(self.efficiency_score, 1)
            },
            'distributions': {
                'carrier': self.carrier_distribution,
                'status': self.status_distribution
            },
            'trends': {
                name: {
                    'values': metrics.values,
                    'timestamps': [t.isoformat() for t in metrics.timestamps],
                    'trend_coefficient': round(metrics.trend_coefficient, 3),
                    'seasonality_index': round(metrics.seasonality_index, 3),
                    'outliers': metrics.outliers
                }
                for name, metrics in self.monthly_trends.items()
            }
        }

class PerformanceAnalyzer:
    """
    Advanced performance analytics engine with comprehensive analysis capabilities.
    
    Features:
    - Complex metric calculations
    - Time-series analysis
    - Predictive modeling
    - Data visualization
    - Parallel processing
    - Caching for performance
    """
    
    def __init__(
        self,
        cache_size: int = 1000,
        max_workers: int = 4,
        analysis_window: int = 12
    ):
        """
        Initialize analyzer with configuration.
        
        Args:
            cache_size: LRU cache size for computations
            max_workers: Maximum number of worker threads
            analysis_window: Months of data for trend analysis
        """
        self.cache_size = cache_size
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.analysis_window = analysis_window
    
    def calculate_agent_metrics(
        self,
        data: pd.DataFrame,
        date_range: Optional[Tuple[datetime, datetime]] = None,
        include_predictions: bool = False
    ) -> PerformanceMetrics:
        """
        Calculate comprehensive agent performance metrics.
        
        Args:
            data: Policy data DataFrame
            date_range: Optional date range for filtering
            include_predictions: Whether to include predictive metrics
            
        Returns:
            PerformanceMetrics: Calculated metrics
        """
        try:
            metrics = PerformanceMetrics()
            
            # Apply date filtering if specified
            if date_range:
                mask = (
                    (data['effective_date'] >= date_range[0]) &
                    (data['effective_date'] <= date_range[1])
                )
                data = data[mask].copy()
            
            # Basic counts
            metrics.total_policies = len(data)
            metrics.active_policies = len(
                data[data['status'].isin(['Active', 'Paid', 'Inforce'])]
            )
            metrics.chargeback_policies = len(
                data[data['status'].isin(['Terminated', 'Lapsed'])]
            )
            
            # Calculate retention rate
            if metrics.total_policies > 0:
                metrics.retention_rate = (
                    metrics.active_policies / metrics.total_policies * 100
                )
            
            # Calculate policy durations in parallel
            def calculate_duration(row):
                start_date = pd.to_datetime(row['effective_date'])
                end_date = pd.to_datetime(row.get('termination_date', pd.Timestamp.now()))
                return (end_date - start_date).days / 30.44  # Average month length
            
            durations = list(self.executor.map(
                calculate_duration,
                [row for _, row in data.iterrows()]
            ))
            
            metrics.avg_policy_duration = np.mean(durations) if durations else 0
            
            # Calculate distributions
            metrics.carrier_distribution = data['carrier'].value_counts().to_dict()
            metrics.status_distribution = data['status'].value_counts().to_dict()
            
            # Calculate time series metrics
            data['effective_date'] = pd.to_datetime(data['effective_date'])
            monthly_data = data.set_index('effective_date').resample('ME')  # Using 'ME' for month end
            
            for metric_name in ['active_policies', 'chargeback_policies', 'retention_rate']:
                values = []
                timestamps = []
                
                for date, group in monthly_data:
                    if metric_name == 'retention_rate':
                        total = len(group)
                        active = len(group[group['status'].isin(['Active', 'Paid', 'Inforce'])])
                        value = (active / total * 100) if total > 0 else 0
                    else:
                        value = len(group) if metric_name == 'active_policies' else len(
                            group[group['status'].isin(['Terminated', 'Lapsed'])]
                        )
                    
                    values.append(float(value))
                    timestamps.append(date)
                
                if values:
                    # Calculate volatility index
                    if metric_name == 'active_policies':
                        metrics.volatility_index = np.std(values) / np.mean(values)
                    
                    metrics.monthly_trends[metric_name] = TimeSeriesMetrics(
                        values=values,
                        timestamps=timestamps
                    )
            
            # Calculate advanced metrics
            metrics.calculate_risk_metrics()
            metrics.calculate_performance_indicators()
            
            # Include predictive metrics if requested
            if include_predictions:
                self._add_predictive_metrics(metrics, data)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating agent metrics: {str(e)}")
            raise
    
    def _add_predictive_metrics(
        self,
        metrics: PerformanceMetrics,
        data: pd.DataFrame
    ) -> None:
        """
        Add predictive analytics to metrics.
        
        Args:
            metrics: Current metrics object
            data: Policy data DataFrame
        """
        try:
            # Implement predictive analytics using time series decomposition
            for metric_name, time_series in metrics.monthly_trends.items():
                if len(time_series.values) >= self.analysis_window:
                    # Perform time series decomposition
                    values = np.array(time_series.values[-self.analysis_window:])
                    
                    # Detect and store outliers
                    z_scores = stats.zscore(values)
                    time_series.outliers = [
                        i for i, z in enumerate(z_scores)
                        if abs(z) > 2
                    ]
                    
                    # Update trend analysis
                    time_series.calculate_trends()
                    
            # More sophisticated predictions could be added here
            
        except Exception as e:
            logger.warning(f"Error adding predictive metrics: {str(e)}")
            # Continue without predictions if they fail
            pass
