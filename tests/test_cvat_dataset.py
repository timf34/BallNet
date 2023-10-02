import numpy as np
import pytest

from config import BohsLaptopConfig, AFLLaptopConfig
from data.cvat_dataset import CVATBallDataset, create_dataset_from_config
from data.augmentation import NoAugmentation


bohs_config = BohsLaptopConfig()
afl_config = AFLLaptopConfig()

@pytest.fixture(scope="module")
def bohs_dataset():
    return create_dataset_from_config(bohs_config)

@pytest.fixture(scope="module")
def afl_dataset():
    return create_dataset_from_config(afl_config)

def test_initialization(bohs_dataset, afl_dataset):
    def assert_initialization(dataset):
        assert dataset is not None, "Failed to initialize CVATBallDataset"

    assert_initialization(bohs_dataset)
    assert_initialization(afl_dataset)

def test_length(bohs_dataset, afl_dataset):
    def assert_length(dataset):
        # Check if __len__ returns the correct length for the dataset
        assert len(dataset) == dataset.n_images, "Dataset length mismatch"

    assert_length(bohs_dataset)
    assert_length(afl_dataset)


def test_get_item():
    dataset_path = '/path/to/dataset'
    transform = NoAugmentation(size=300)
    dataset = CVATBallDataset(dataset_path=dataset_path, transform=transform)

    # Scenario 1: Test fetching an item
    image, boxes, labels = dataset[0]
    assert image is not None, "Image should not be None"
    assert boxes is not None, "Boxes should not be None"
    assert labels is not None, "Labels should not be None"


def test_get_annotations():
    dataset_path = '/path/to/dataset'
    transform = NoAugmentation(size=300)
    dataset = CVATBallDataset(dataset_path=dataset_path, transform=transform)

    # Scenario 1: Test getting annotations
    boxes, labels = dataset.get_annotations('camera_id', 'image_ndx')
    assert type(boxes) == np.ndarray, "Boxes should be numpy array"
    assert type(labels) == np.ndarray, "Labels should be numpy array"


def test_get_elems_with_ball():
    dataset_path = '/path/to/dataset'
    transform = NoAugmentation(size=300)
    dataset = CVATBallDataset(dataset_path=dataset_path, transform=transform)

    # Scenario 1: Test fetching elements with ball
    elems_with_ball = dataset.get_elems_with_ball()
    assert type(elems_with_ball) == list, "Should return a list"


# Running tests
# test_initialization(bohs_dataset, afl_dataset)
# test_length(bohs_dataset, afl_dataset)
# test_get_item()
# test_get_annotations()
# test_get_elems_with_ball()
