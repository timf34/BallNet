"""
Tests for the dataloader module.

This module ensures:
1. Initialization of all types of dataloaders (train, val, test).
2. Correctness of the dataloaders.
3. Functionality of iterating through the dataloaders.
4. Functionality of the collate function.
"""

import pytest
import torch
from copy import deepcopy
from typing import List, Dict

from config import BohsLaptopConfig, AFLLaptopConfig
from data.cvat_dataloaders import make_data_loader

# Constants
VALID_IMAGE_SHAPES = [(3, 1080, 1920), (3, 720, 1280)]
VALID_BOX_SHAPES = [torch.Size([1, 4]), torch.Size([0, 4])]

bohs_config = BohsLaptopConfig()
afl_config = AFLLaptopConfig()
whole_dataset_config = BohsLaptopConfig()
whole_dataset_config.whole_dataset = True

configs = [bohs_config, afl_config]
modes = ["train", "val", "test"]


@pytest.mark.parametrize("config", configs)
def test_initialization(config):
    data_loaders = make_data_loader(config, modes, False)
    assert isinstance(data_loaders, Dict)
    for mode in modes:
        assert mode in data_loaders


@pytest.mark.parametrize("config", configs)
def test_iterating_dataloader(config):
    data_loaders = make_data_loader(config, modes, False)
    for images, boxes, labels in data_loaders["train"]:

        # Test images
        assert isinstance(images, torch.Tensor)
        assert len(images.shape) == 4
        assert images.shape[1:] in VALID_IMAGE_SHAPES

        # Test boxes
        assert isinstance(boxes, list)
        assert all(isinstance(box, torch.Tensor) for box in boxes)
        assert all(box.shape in VALID_BOX_SHAPES for box in boxes)

        # Test labels
        assert isinstance(labels, list)
        assert all(isinstance(label, torch.Tensor) for label in labels)
        assert all(len(label.shape) == 1 for label in labels)


# Note: with a small dataset size, this will sometimes fail, just run it again.
@pytest.mark.parametrize("config", configs)
def test_data_shuffling(config):
    data_loaders = make_data_loader(config, modes, False)

    # Get the boxes from the first batch for the first epoch
    first_epoch_boxes = next(iter(data_loaders["train"]))[1]
    previous_boxes = deepcopy(first_epoch_boxes)

    num_epochs_to_check = 3
    for _ in range(num_epochs_to_check):
        current_boxes = next(iter(data_loaders["train"]))[1]

        # Check if the boxes from the first batch are the same across epochs
        # If they are, it means the shuffling isn't working as expected
        for prev, curr in zip(previous_boxes, current_boxes):
            assert not torch.equal(prev, curr), f"Data is not being shuffled across epochs! " \
                                                f"Previous boxes: {prev}, Current boxes: {curr}"

        previous_boxes = deepcopy(current_boxes)
