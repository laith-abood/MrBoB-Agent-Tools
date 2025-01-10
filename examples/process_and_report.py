"""
Example script demonstrating core functionality of MrBoB Agent Tools.

This script shows how to:
1. Process raw policy data
2. Generate performance reports
3. Analyze agent performance
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

# MrBoB imports
from mrbob import (
    PolicyProcessor,
    PolicyProcessingError,
    ReportService,
    GenerateReportCommand,
    AddSectionCommand,
    FinalizeReportCommand,
    ReportType,
    ReportGenerationError,
    PerformanceAnalyzer
)

# Local imports
import mock_dependencies as mock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def process_policy_data(file_path: str) -> Dict[str, Any]:
    """
    Process raw policy data file.
    
    Args:
        file_path: Path to input data file
        
    Returns:
        Dict[str, Any]: Processed policy data
        
    Raises:
        PolicyProcessingError: If processing fails
    """
    try:
        # Initialize processor with default configuration
        processor = PolicyProcessor()
        
        # Process data with validation
        result = await processor.process_policy_data(
            file_path=file_path,
            required_columns=[
                'Policy Number',
                'Status',
                'Carrier',
                'Agent NPN',
                'Effective Date'
            ]
        )
        
        logger.info(
            f"Processed {result['metrics']['processed_count']} policies"
        )
        return result
        
    except PolicyProcessingError as e:
        logger.error(f"Failed to process policy data: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing policies: {str(e)}")
        raise PolicyProcessingError(str(e))


async def analyze_performance(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze agent performance metrics.
    
    Args:
        data: Processed policy data
        
    Returns:
        Dict[str, Any]: Performance analysis results
    """
    try:
        # Initialize analyzer
        analyzer = PerformanceAnalyzer()
        
        # Convert policy data to DataFrame
        df = pd.DataFrame(data['data']['policies'])
        
        # Calculate comprehensive metrics
        metrics = analyzer.calculate_agent_metrics(
            data=df,
            include_predictions=True
        )
        
        logger.info(
            f"Generated metrics for {len(data['data']['policies'])} policies"
        )
        return metrics.to_dict()
        
    except Exception as e:
        logger.error(f"Failed to analyze performance: {str(e)}")
        raise


async def generate_reports(
    data: Dict[str, Any],
    output_dir: str
) -> List[str]:
    """
    Generate agent performance reports.
    
    Args:
        data: Processed and analyzed policy data
        output_dir: Directory to save reports
        
    Returns:
        List[str]: Paths to generated report files
        
    Raises:
        ReportGenerationError: If report generation fails
    """
    try:
        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize report service with mock dependencies
        report_service = ReportService(
            repository=mock.MockReportRepository(),
            template_engine=mock.MockTemplateEngine(),
            event_publisher=mock.MockEventPublisher(),
            unit_of_work=mock.MockUnitOfWork()
        )
        
        # Generate reports for each agent
        report_files = []
        for agent_npn in data['data']['agent_breakdown'].keys():
            # Create report command
            command = GenerateReportCommand(
                agent_npn=agent_npn,
                report_type=ReportType.PERFORMANCE_ANALYSIS,
                generated_by='system',
                options={
                    'include_charts': True,
                    'format': 'pdf'
                }
            )
            
            # Generate report
            report_id = await report_service.handle_command(command)
            
            # Add performance metrics section
            metrics_section = AddSectionCommand(
                report_id=report_id,
                title="Performance Metrics",
                content=data['analysis']['summary'],
                order=1,
                template_key="metrics"
            )
            await report_service.handle_command(metrics_section)
            
            # Add policy details section
            policy_section = AddSectionCommand(
                report_id=report_id,
                title="Policy Details",
                content={
                    'policies': [
                        policy for policy in data['data']['policies']
                        if policy['agent_npn'] == agent_npn
                    ]
                },
                order=2,
                template_key="policies"
            )
            await report_service.handle_command(policy_section)
            
            # Finalize and save report
            report_path = output_path / f"agent_{agent_npn}_{report_id}.pdf"
            finalize_command = FinalizeReportCommand(
                report_id=report_id,
                output_path=str(report_path)
            )
            await report_service.handle_command(finalize_command)
            
            report_files.append(str(report_path))
            
            logger.info(f"Generated report for agent {agent_npn}")
        
        return report_files
        
    except ReportGenerationError as e:
        logger.error(f"Failed to generate reports: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error generating reports: {str(e)}")
        raise ReportGenerationError(str(e))


async def main():
    """Main execution flow."""
    try:
        # Process policy data
        data = await process_policy_data("data/sample_policy_data.csv")
        
        # Analyze performance
        analysis = await analyze_performance(data)
        
        # Generate reports
        reports = await generate_reports(
            data={**data, 'analysis': analysis},
            output_dir="output/reports"
        )
        
        logger.info(f"Generated {len(reports)} report files")
        for report in reports:
            logger.info(f"Report saved to: {report}")
            
    except Exception as e:
        logger.error(f"Process failed: {str(e)}")
        raise


if __name__ == "__main__":
    # Run async main
    asyncio.run(main())
