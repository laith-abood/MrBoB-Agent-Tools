"""FastAPI server for MrBoB Agent Tools."""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import logging

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

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
from examples.mock_dependencies import (
    MockReportRepository,
    MockTemplateEngine,
    MockEventPublisher,
    MockUnitOfWork
)

app = FastAPI(title="MrBoB Agent Tools")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

# Ensure output directories exist
UPLOAD_DIR = Path("data/uploads")
REPORT_DIR = Path("output/reports")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Initialize services
processor = PolicyProcessor()
analyzer = PerformanceAnalyzer()
report_service = ReportService(
    repository=MockReportRepository(),
    template_engine=MockTemplateEngine(),
    event_publisher=MockEventPublisher(),
    unit_of_work=MockUnitOfWork()
)

@app.get("/")
async def root():
    """Serve the main frontend page."""
    return FileResponse("frontend/index.html")

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Handle file upload and process policy data."""
    """
    Handle file upload and process policy data.
    
    Args:
        file: Uploaded CSV file
        
    Returns:
        Dict[str, Any]: Processing results and metrics
    """
    if not file.filename:
        raise HTTPException(400, "No file provided")
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(400, "File must be a CSV file")
    
    try:
        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = UPLOAD_DIR / f"policies_{timestamp}.csv"
        
        try:
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
        except Exception as e:
            logger.error(f"Failed to save file: {str(e)}")
            raise HTTPException(500, "Failed to save uploaded file")
        
        # Process policy data
        try:
            result = await processor.process_policy_data(
                file_path=str(file_path),
                required_columns=[
                    'Policy Number',
                    'Status',
                    'Carrier',
                    'Agent NPN',
                    'Effective Date'
                ]
            )
        except PolicyProcessingError as e:
            logger.error(f"Policy processing failed: {str(e)}")
            raise HTTPException(400, f"Failed to process policy data: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during processing: {str(e)}")
            raise HTTPException(500, "Internal server error during processing")
        
        # Generate performance metrics
        metrics = await analyzer.calculate_agent_metrics(
            data=result['data']['policies']
        )
        
        # Generate reports for each agent
        reports = []
        for agent_npn in result['data']['agent_breakdown'].keys():
            # Create report
            command = GenerateReportCommand(
                agent_npn=agent_npn,
                report_type=ReportType.PERFORMANCE_ANALYSIS,
                generated_by='system',
                options={
                    'include_charts': True,
                    'format': 'html'
                }
            )
            report_id = await report_service.handle_command(command)
            
            # Add metrics section
            metrics_section = AddSectionCommand(
                report_id=report_id,
                title="Performance Metrics",
                content=metrics.to_dict()['summary'],
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
                        policy for policy in result['data']['policies']
                        if policy['agent_npn'] == agent_npn
                    ]
                },
                order=2,
                template_key="policies"
            )
            await report_service.handle_command(policy_section)
            
            # Finalize report
            report_path = REPORT_DIR / f"agent_{agent_npn}_{report_id}.html"
            finalize_command = FinalizeReportCommand(
                report_id=report_id,
                output_path=str(report_path)
            )
            await report_service.handle_command(finalize_command)
            
            reports.append({
                'id': str(report_id),
                'agent_npn': agent_npn,
                'path': str(report_path)
            })
        
        # Prepare response data
        response_data = {
            'totalAgents': len(result['data']['agent_breakdown']),
            'activePolicies': len([
                p for p in result['data']['policies']
                if p['status'] in ['Active', 'Paid', 'Inforce']
            ]),
            'retentionRate': metrics.to_dict()['summary']['retention_rate'],
            'avgPolicyDuration': metrics.to_dict()['summary']['avg_policy_duration'],
            'agents': [
                {
                    'id': report['agent_npn'],
                    'name': f"Agent {report['agent_npn']}",
                    'activePolicies': len([
                        p for p in result['data']['policies']
                        if p['agent_npn'] == report['agent_npn'] and
                        p['status'] in ['Active', 'Paid', 'Inforce']
                    ]),
                    'retentionRate': 95.5,  # TODO: Calculate per agent
                    'performanceScore': 92,  # TODO: Calculate per agent
                }
                for report in reports
            ],
            'reports': reports
        }
        
        return JSONResponse(response_data)
        
    except (PolicyProcessingError, ReportGenerationError) as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/reports/{agent_id}")
async def get_report(agent_id: str) -> str:
    """
    Get the URL for an agent's report.
    
    Args:
        agent_id: Agent NPN
        
    Returns:
        str: Report URL
    """
    try:
        # Find the latest report for this agent
        reports = sorted(
            REPORT_DIR.glob(f"agent_{agent_id}_*.html"),
            key=os.path.getmtime,
            reverse=True
        )
        
        if not reports:
            raise HTTPException(404, f"No report found for agent {agent_id}")
        
        # Return the relative URL
        return f"/output/reports/{reports[0].name}"
        
    except Exception as e:
        raise HTTPException(500, str(e))

# Mount output directory for report access
app.mount("/output", StaticFiles(directory="output"), name="output")

if __name__ == "__main__":
    try:
        logger.info("Starting server...")
        uvicorn.run(
            "server:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="debug"
        )
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise
