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
