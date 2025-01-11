import pytest
import pandas as pd
from mrbob.core.retry import process_chunk_with_retry
from mrbob.core.policy_processor import PolicyProcessor
from mrbob.core.exceptions import PolicyProcessingError

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
async def test_process_chunk_with_retry(processor, sample_data):
    result = await process_chunk_with_retry(processor, sample_data)
    assert len(result) == 3
    assert result[0]['policy_number'] == 'POL001'
    assert result[1]['status'] == 'Terminated'

@pytest.mark.asyncio
async def test_exponential_backoff_and_retry_configuration(processor, sample_data):
    processor._retry_config = {
        'max_attempts': 3,
        'initial_delay': 1,
        'max_delay': 5,
        'backoff_factor': 2
    }
    result = await process_chunk_with_retry(processor, sample_data)
    assert len(result) == 3
    assert result[0]['policy_number'] == 'POL001'
    assert result[1]['status'] == 'Terminated'

@pytest.mark.asyncio
async def test_compatibility_with_policy_processor_class(processor, sample_data):
    result = await process_chunk_with_retry(processor, sample_data)
    assert len(result) == 3
    assert result[0]['policy_number'] == 'POL001'
    assert result[1]['status'] == 'Terminated'
