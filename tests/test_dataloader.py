"""
What this file needs to test:

1. Initialization of all the types of dataloaders (i.e. train, val, test)
    Note: functionality might still need to be implemented here.

2. Verify correctness of the dataloaders (i.e. are the images and labels correct, right length, right paths used, etc.)

3. Functionality of the dataloaders (i.e. can we iterate through them, etc.)

4. Functionality of the collate function (i.e. can we iterate through them, etc.)
"""
import torch
from typing import List, Dict

from config import BohsLaptopConfig, AFLLaptopConfig
from data.cvat_dataloaders import make_data_loader


bohs_config = BohsLaptopConfig()
afl_config = AFLLaptopConfig()
whole_dataset_config = BohsLaptopConfig()
whole_dataset_config.whole_dataset = True

# Lists for testing
configs = [bohs_config, afl_config, whole_dataset_config]
modes = ["train", "val", "test"]

def test_initialization():
    bohs_data_loaders = make_data_loader(bohs_config, modes, False)
    afl_data_loaders = make_data_loader(afl_config, modes, False)
    # whole_dataset_data_loaders = make_data_loader(whole_dataset_config, modes, False)

    assert isinstance(bohs_data_loaders, Dict)
    assert isinstance(afl_data_loaders, Dict)


def test_iterating_dataloder():
    bohs_data_loaders = make_data_loader(bohs_config, modes, False)

    for images, boxes, labels in bohs_data_loaders["train"]:
        assert isinstance(images, torch.Tensor), "images should be a tensor"
        assert len(images.shape) == 4, "Image should have 4 dimensions (batch size, channels, height, width)"
        assert images.shape[1:] in [
            (3, 1080, 1920),
            (3, 720, 1280)
        ], "Image should have shape (3, 1080, 1920) or (3, 720, 1280)"

        assert isinstance(boxes, list), "boxes should be a list of tensors"
        # boxes is a list of tensors of shape [n_boxes, 4]
        assert all(isinstance(box, torch.Tensor) for box in boxes), "boxes should be a list of tensors"
        print(len(boxes[0].shape), boxes[0].shape, len(boxes), boxes, len(boxes[0]), boxes[0])
        assert all((box.shape == torch.Size([1, 4]) or box.shape == torch.Size([0, 4])) for box in boxes), \
            f"each box should have shape (1, 4) or (0, 4). Found: {[box.shape for box in boxes if box.shape not in [torch.Size([1, 4]), torch.Size([0, 4])]]}"

        assert isinstance(labels, list)
        assert all(isinstance(label, torch.Tensor) for label in labels)
        assert all(len(label.shape) == 1 for label in labels)

