import torch
from network.utils import visualize_feature_maps


def test_visualize_ball_feature_maps():
    """
        This function visualizes the ball feature maps.
    """
    t1 = torch.rand(1, 2, 270, 480)
    t2 = torch.rand(1, 270, 480)

    visualize_feature_maps(t1)
    visualize_feature_maps(t2)


if __name__ == '__main__':
    test_visualize_ball_feature_maps()