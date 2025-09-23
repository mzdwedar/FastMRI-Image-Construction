import pytest

from madewithml.data import CustomPreprocessor


data_path = "data"

@pytest.fixture
def dataset_loc():
    return data_path


@pytest.fixture
def preprocessor():
    return CustomPreprocessor()
