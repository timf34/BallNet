import numpy as np
import pytest

from typing import List

from config import BohsLaptopConfig, AFLLaptopConfig
from data.cvat_dataset import create_dataset_from_config


bohs_config = BohsLaptopConfig()
afl_config = AFLLaptopConfig()
whole_dataset_config = BohsLaptopConfig()

bohs_config.training_mode = "train"
afl_config.training_mode = "val"
whole_dataset_config.training_mode = "test"
whole_dataset_config.whole_dataset = True


@pytest.fixture(scope="module")
def bohs_dataset():
    return create_dataset_from_config(bohs_config, training_mode="train", use_hardcoded_data_folders=False)

@pytest.fixture(scope="module")
def afl_dataset():
    return create_dataset_from_config(afl_config, training_mode="val", use_hardcoded_data_folders=False)

@pytest.fixture(scope="module")
def whole_dataset():
    return create_dataset_from_config(whole_dataset_config, training_mode="test", use_hardcoded_data_folders=False)

def test_initialization(bohs_dataset, afl_dataset, whole_dataset):
    def assert_initialization(dataset):
        assert dataset is not None, "Failed to initialize CVATBallDataset"

    assert_initialization(bohs_dataset)
    assert_initialization(afl_dataset)
    assert_initialization(whole_dataset)

def test_length(bohs_dataset, afl_dataset, whole_dataset):
    def assert_length(dataset):
        # Check if __len__ returns the correct length for the dataset
        assert len(dataset) == dataset.n_images, "Dataset length mismatch"

    assert_length(bohs_dataset)
    assert_length(afl_dataset)
    assert_length(whole_dataset)


def test_get_item(bohs_dataset, afl_dataset, whole_dataset):
    def assert_get_item(dataset):
        # Scenario: Test fetching an item
        image, boxes, labels = dataset[0]
        assert image is not None, "Image should not be None"
        assert boxes is not None, "Boxes should not be None"
        assert labels is not None, "Labels should not be None"

    assert_get_item(bohs_dataset)
    assert_get_item(afl_dataset)
    assert_get_item(whole_dataset)


def test_get_annotations(bohs_dataset, afl_dataset, whole_dataset):
    def assert_get_annotations(dataset, data_folder: str, image_ndx: List[str]):
        for idx in image_ndx:
            boxes, labels = dataset.get_annotations(data_folder, idx)
            assert type(boxes) == np.ndarray, "Boxes should be numpy array"
            assert type(labels) == np.ndarray, "Labels should be numpy array"

    # NDX values here are hardcoded to contain a sample with no label, and with a label respectively

    bohs_data_folder = bohs_config.data_folder_paths[0]
    bohs_image_ndxs = [0, 533]
    assert_get_annotations(bohs_dataset, bohs_data_folder, bohs_image_ndxs)
    assert_get_annotations(whole_dataset, bohs_data_folder, bohs_image_ndxs)

    afl_data_folder = afl_config.data_folder_paths[0]  # Note: set up for vals folder rn
    afl_image_ndxs = [0, 108]
    assert_get_annotations(afl_dataset, afl_data_folder, afl_image_ndxs)


def test_get_elems_with_ball(bohs_dataset, afl_dataset, whole_dataset):
    def assert_get_elems_with_ball(dataset):
        elems_with_ball = dataset.get_elems_with_ball()
        assert type(elems_with_ball) == list, "Should return a list"
        assert len(elems_with_ball) > 0, "Should return a non-empty list"
        assert len(elems_with_ball) == len(dataset.image_list), "Should return a list with the same length " \
                                                                "as the dataset. Note: assuming we only label the ball" \
                                                                "and no players."

    assert_get_elems_with_ball(bohs_dataset)
    assert_get_elems_with_ball(afl_dataset)
    assert_get_elems_with_ball(whole_dataset)
