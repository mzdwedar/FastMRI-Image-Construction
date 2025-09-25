import pytest

data_path = "data"

@pytest.fixture
def dataset_loc():
    return data_path
