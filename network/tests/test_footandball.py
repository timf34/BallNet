import torch
import torch.nn as nn
import cv2
import os

from network.footandball import model_factory
from misc.python_learning.tensors import ball_feature_map2heatmap


def get_footandball_model(mode: str = 'train'):
    return model_factory('fb1', mode)


def log_prints():
    """
        Call this when I want to log the print statements to a file
    """
    import sys
    sys.stdout = open(os.path.join(sys.path[0], "log_test_footandball.log"), "w")  # using sys.path[0] gets current directory


def get_footandball_training_output() -> torch.Tensor:
    """
        Returns the output of the network when it is in training mode
        output: torch.Tensor([1, 180, 320, 2]) for input of shape (1, 3, 720, 1280)
    """
    model = get_footandball_model('train')
    random_image_tensor = torch.rand(1, 3, 720, 1280)
    return model(random_image_tensor)


def test_footandball_training_tensors():
    model = get_footandball_model()

    random_tensor = torch.rand(1, 3, 720, 1280)  # 3 channels is rgb (its an image!)

    output = model(random_tensor)
    print("Here is the output when the network is in training mode...")
    print("output shape: ", output.shape)
    print("output type: ", type(output))
    print("output dtype: ", output.dtype)
    print("output: ", output)
    print("\n")

    assert output.shape == torch.Size([1, 180, 320, 2])
    assert type(output) == torch.Tensor
    assert output.dtype == torch.float32


def test_save_normalized_feature_maps():
    """
        Note that this isn't going to be a robust test - more so for just trying it out
    """
    x = get_footandball_training_output()
    save_normalized_feature_maps(x)


def save_normalized_feature_maps(feature_maps: torch.Tensor, epoch: int = 0) -> None:
    """
        This function passes the input feature map through a softmax function, and then saves it locally
        using cv2.imwrite
    """
    sftmax = nn.Softmax(dim=1)
    normalized_feature_map = sftmax(feature_maps)
    h = ball_feature_map2heatmap(normalized_feature_map.permute(0, 3, 1, 2))

    print(h.shape)

    cv2.imwrite(f"working.png", h)


if __name__ == "__main__":
    print("oh no")
    print("oh yes")
    log_prints()
    test_save_normalized_feature_maps()