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
