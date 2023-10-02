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


def test_get_item(bohs_dataset, afl_dataset):
    def assert_get_item(dataset):
        # Scenario: Test fetching an item
        image, boxes, labels = dataset[0]
        assert image is not None, "Image should not be None"
        assert boxes is not None, "Boxes should not be None"
        assert labels is not None, "Labels should not be None"

    assert_get_item(bohs_dataset)
    assert_get_item(afl_dataset)


def test_get_annotations(bohs_dataset, afl_dataset):
    def assert_get_annotations(dataset, data_folder, image_ndx=0):
        boxes, labels = dataset.get_annotations(data_folder, image_ndx)
        assert type(boxes) == np.ndarray, "Boxes should be numpy array"
        assert type(labels) == np.ndarray, "Labels should be numpy array"

    # NDX values here are hardcoded to contain a sample with no label, and with a label respectively

    bohs_data_folder = bohs_config.train_data_folders[0]
    bohs_image_ndxs = [0, 533]
    for bohs_image_ndx in bohs_image_ndxs:
        assert_get_annotations(bohs_dataset, bohs_data_folder, bohs_image_ndx)

    afl_data_folder = afl_config.train_data_folders[0]
    afl_image_ndx = [0, 495]
    for afl_image_ndx in afl_image_ndx:
        assert_get_annotations(afl_dataset, afl_data_folder, afl_image_ndx)


def test_get_elems_with_ball(bohs_dataset, afl_dataset):
    def assert_get_elems_with_ball(dataset):
        elems_with_ball = dataset.get_elems_with_ball()
        assert type(elems_with_ball) == list, "Should return a list"
        assert len(elems_with_ball) > 0, "Should return a non-empty list"
        assert len(elems_with_ball) == len(dataset.image_list), "Should return a list with the same length " \
                                                                "as the dataset. Note: assuming we only label the ball" \
                                                                "and no players."

    assert_get_elems_with_ball(bohs_dataset)
    assert_get_elems_with_ball(afl_dataset)


# Running tests
# test_initialization(bohs_dataset, afl_dataset)
# test_length(bohs_dataset, afl_dataset)
# test_get_item()
# test_get_annotations()
# test_get_elems_with_ball()
