from network.ssd_loss import SSDLoss
from data.augmentation import heatmap2image, tensor2image
from network.utils import visualize_feature_maps
import torch
import cv2

"""
    One of the main purposes of this file is to figure out why the loss doesn't go to 0 when we overfit on a single 
    image with the ball in it, and the network succesfully predicts the ball.
    
    To do this we will instead use the raw ball feature map rather than needing to load the network each time. 
    This should be much easier and straightforward to work with. 
    
    We will also need the targets
"""

feature_map_path = r'C:\Users\timf3\PycharmProjects\BohsNet\txt_testing_files\no_softmax_ball_feature_map.pt'
no_softmax_feature_map_path = r'C:\Users\timf3\PycharmProjects\BohsNet\txt_testing_files\predictions\195.pth'
gt_map_path = r'C:\Users\timf3\PycharmProjects\BohsNet\txt_testing_files\gt_map.pt'
no_softmax_image = r'C:\Users\timf3\PycharmProjects\BohsNet\images\heatmaps\Date-26-05-2022-Time-20-27\195.png'


def load_ssd_loss():
    return SSDLoss(neg_pos_ratio=3)


def png_to_tensor(no_softmax_image: str):
    """
        Given the path to a png image, load and convert it to a tensor for use in the network.
        :param
    """
    img = cv2.imread(no_softmax_image)


def visualizing_predictions_and_gt_maps_ssd_loss():
    # fmap = torch.load(no_softmax_feature_map_path)
    fmap = torch.load(feature_map_path)
    gt_map = torch.load(gt_map_path)
    loss_gt_map = [x.to('cuda') for x in gt_map]

    gt_map = gt_map.float().to('cuda')

    print(fmap)
    print(gt_map)

    print("gt map shape", gt_map.shape)  # torch.Size([1, 270, 480]) (except its also wrapped in a list
    print("fmap shape", fmap.shape)  # torch.Size([1, 2, 270, 480]) or if its no softmax -> torch.Size([1, 270, 480, 2])!
    print(f"their types, gt map {type(gt_map)} and fmap {type(fmap)}")

    print("fmap type", type(fmap), "fmap shape", fmap.shape)
    # visualize_feature_maps(fmap)
    print("gt map type", type(gt_map), "gt map shape", gt_map.shape)
    # visualize_feature_maps(gt_map)
    print("loss gt map type", type(loss_gt_map))
    # visualize_feature_maps(loss_gt_map)

    ssd_loss = load_ssd_loss()
    loss = ssd_loss(fmap, loss_gt_map)
    print(loss)

    print("lets compare the two tensors: ", torch.allclose(fmap, gt_map))

    # The maps don't match exactly but they are extrememely close and yet the loss is 2.9844 or so. See the log
    # for more visualizing_preds...


if __name__ == '__main__':
    visualizing_predictions_and_gt_maps_ssd_loss()