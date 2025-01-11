# MrBoB Agent Tools

A comprehensive suite of tools for insurance agents to manage and analyze policy data, generate reports, and track performance metrics.

## Features

- **Policy Processing**: Efficient processing of policy data with validation and status resolution
- **Performance Analytics**: Advanced analytics with trend analysis and predictions
- **Report Generation**: Customizable reports in multiple formats (PDF, Excel)
- **Caching Strategy**: Optimized performance with intelligent caching
- **Event Sourcing**: Robust state tracking and audit trails
- **Concurrent Processing**: Efficient handling of large datasets

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/laith-abood/MrBoB-Agent-Tools.git
cd MrBoB-Agent-Tools
```

2. Set up development environment:
```bash
# Make setup script executable
chmod +x scripts/setup_dev.sh

# Run setup script
./scripts/setup_dev.sh
```

3. Activate virtual environment:
```bash
source venv/bin/activate
```

## Project Structure

```
mrbob/
├── core/               # Core processing functionality
│   ├── policy_processor.py
│   └── status_resolver.py
├── analytics/          # Performance analytics
│   └── performance_analyzer.py
└── reports/           # Report generation
    ├── application/   # Application services
    │   └── report_service.py
    └── domain/        # Domain models
        └── report_aggregate.py
```

## Usage Example

```python
from mrbob import (
    PolicyProcessor,
    ReportService,
    PerformanceAnalyzer
)

# Process policy data
processor = PolicyProcessor()
result = await processor.process_policy_data("data/policies.csv")

# Analyze performance
analyzer = PerformanceAnalyzer()
metrics = analyzer.calculate_agent_metrics(result['data'])

# Generate reports
report_service = ReportService()
report_id = await report_service.generate_report(metrics)
```

See `examples/process_and_report.py` for a complete example.

## Key Components

### Policy Processor
- Validates and processes raw policy data
- Resolves policy statuses with conflict handling
- Supports concurrent processing with retry logic

### Performance Analyzer
- Calculates comprehensive performance metrics
- Provides trend analysis and predictions
- Supports various aggregation strategies

### Report Service
- Generates customized reports
- Supports multiple output formats
- Implements CQRS pattern for scalability

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Setting Up and Running the Server

1. Ensure you have Python 3.8 or higher installed.
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
3. Run the FastAPI server:
```bash
uvicorn server:app --reload
```
4. Open your browser and navigate to `http://127.0.0.1:8000` to access the application.

## Using Different Features of the Repository

### Processing Policy Data
To process policy data, use the `PolicyProcessor` class. Here's an example:
```python
from mrbob import PolicyProcessor

processor = PolicyProcessor()
result = await processor.process_policy_data("data/sample_policy_data.csv")
print(result)
```

### Analyzing Performance
To analyze performance metrics, use the `PerformanceAnalyzer` class. Here's an example:
```python
from mrbob import PerformanceAnalyzer

analyzer = PerformanceAnalyzer()
metrics = analyzer.calculate_agent_metrics(result['data'])
print(metrics)
```

### Generating Reports
To generate reports, use the `ReportService` class. Here's an example:
```python
from mrbob import ReportService, GenerateReportCommand, AddSectionCommand, FinalizeReportCommand, ReportType

report_service = ReportService()
command = GenerateReportCommand(
    agent_npn="12345",
    report_type=ReportType.PERFORMANCE_ANALYSIS,
    generated_by="system",
    options={"include_charts": True, "format": "pdf"}
)
report_id = await report_service.handle_command(command)

metrics_section = AddSectionCommand(
    report_id=report_id,
    title="Performance Metrics",
    content=metrics.to_dict()['summary'],
    order=1,
    template_key="metrics"
)
await report_service.handle_command(metrics_section)

policy_section = AddSectionCommand(
    report_id=report_id,
    title="Policy Details",
    content={"policies": result['data']['policies']},
    order=2,
    template_key="policies"
)
await report_service.handle_command(policy_section)

finalize_command = FinalizeReportCommand(
    report_id=report_id,
    output_path="output/reports/agent_12345_report.pdf"
)
await report_service.handle_command(finalize_command)
```
