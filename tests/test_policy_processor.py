import pytest
import pandas as pd
from mrbob.core.policy_processor import PolicyProcessor
from mrbob.core.exceptions import PolicyProcessingError
from mrbob.core.validation import DataValidator
from mrbob.core.status_resolver import StatusResolver
from mrbob.core.metrics import ProcessingMetrics

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'Policy Number': ['POL001', 'POL002', 'POL003'],
        'Status': ['Active', 'Terminated', 'Active'],
        'Carrier': ['BlueCross', 'Aetna', 'Cigna'],
        'Agent NPN': ['12345', '67890', '13579'],
        'Effective Date': ['2024-01-01', '2023-12-15', '2024-01-05']
    })

@pytest.fixture
def processor():
    return PolicyProcessor()

@pytest.mark.asyncio
async def test_process_policy_data_with_validation(processor, sample_data):
    result = await processor._process_chunk(sample_data)
    assert len(result) == 3
    assert result[0]['policy_number'] == 'POL001'
    assert result[1]['status'] == 'Terminated'

@pytest.mark.asyncio
async def test_process_policy_data_with_status_resolution(processor, sample_data):
    result = await processor._process_chunk(sample_data)
    assert result[0]['status'] == 'Active'
    assert result[1]['status'] == 'Terminated'

@pytest.mark.asyncio
async def test_process_policy_data_with_metrics_tracking(processor, sample_data):
    result = await processor._process_chunk(sample_data)
    metrics = processor.metrics.to_dict()
    assert metrics['processed_count'] == 3
    assert metrics['error_count'] == 0
