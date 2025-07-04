import pytest

@pytest.fixture
def radar_chart_config():
    return {
        'middle_ring': 0,
        'values': [1, -1, 2, -2, 3]
    }