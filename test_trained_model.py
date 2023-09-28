import cv2
import numpy as np

import torch
import torch.backends.cudnn as cudnn

from network import footandball
from data.data_reader import make_dataloaders
from config import Config
from data.augmentation import numpy2tensor, tensor2image, heatmap2image
from scripts.run_and_eval_utils import draw_bboxes
from misc.python_learning.tensors import ball_feature_map2heatmap, inspect_ball_feature_map

IMAGES_TXT = './txt_testing_files/image_paths.txt'
WEIGHTS_TXT = './txt_testing_files/weights_path.txt'

LOAD_WORKING_WEIGHTS = False

device = "cuda" if torch.cuda.is_available() else 'cpu'


def read_txt_file(path) -> list:
    """
    Reads a txt file and returns a list of lines.
    :param path: Path to the txt file.
    :return: List of lines.
    """
    with open(path, 'r') as f:
        lines = f.read().splitlines()
    return lines


def jpeg2numpy_cv2(image_path) -> np.ndarray:
    """
    Loads a jpeg image and returns a numpy array.
    :param image_path: Path to the jpeg image.
    :return: A numpy array.
    """
    return cv2.imread(image_path)


def create_model() -> torch.nn.Module:
    """
    Creates a model and returns it.
    """
    print(device)
    model = footandball.model_factory(params.model, 'detect')
    model.print_summary(show_architecture=False)
    model = model.to(device)
    if device is 'cuda':
        cudnn.enabled = True
        cudnn.benchmark = True
    return model


def run_model(dataloader=None) -> None:
    """
        Plan:
1. Create model
2. Load weights
3. Set up a for loop to run the model over the images in the txt file
    1. I have to load the image separately first as a numpy array so that I can draw and show it after.
    2. I'll separately load this image as a torch tensor so that I can pass it to the model.

    """
    # Load the model
    model = create_model()
    # Load the weights

    if LOAD_WORKING_WEIGHTS:
        weights_path = r'C:\Users\timf3\Downloads\model_20210221_2206_final.pth'
    else:
        weights_path = read_txt_file(WEIGHTS_TXT)[0]
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    # TODO: note sure if the images_paths.txt code is working at all
    #  and this file/ fuction in general needs cleaning up
    # But lets get the code training first and then clean it up later.

    if dataloader is None:
        # Get image path list
        image_paths_list = read_txt_file(IMAGES_TXT)

        # Set up a for loop to run the model over the images in the txt file
        for image_path in image_paths_list:
            print("Image path:", image_path)
            # Load the jpeg image
            image = jpeg2numpy_cv2(image_path)
            print("Image shape:", image.shape)
            # Convert to torch tensor
            image_tensor = numpy2tensor(image).unsqueeze(dim=0).to(device)
        # Run the model
            with torch.no_grad():
                detections = model(image_tensor)[0]  # Returns a dictionary
                print(detections)
    else:
        for idx ,(images, boxes, labels) in enumerate(dataloader['train']):
            print(idx)
            print(dataloader)
            print(f"Training set: Dataset size: {len(dataloader['train'].dataset)}")
            # images = images.to(device)
            # print(images)
            print("images.shape", images.shape)
            image = tensor2image(images[0, :, :, :])
            cv2.imshow("image", image)
            cv2.waitKey(0)

            images = images.to(device)
            with torch.no_grad():
                detections = model(images)[0]
            print("detection shape in for loop,", detections["ball_feature_map"].shape)

            print("detection shape,", detections["ball_feature_map"].shape)
            # Draw the bounding boxes
            image = draw_bboxes(image, detections)

            # Saving our detections
            torch.save(detections["ball_feature_map"], "./txt_testing_files/ball_feature_map.pt")

            show_images(image, detections)


def show_images(image, detections: dict) -> None:
    """
        This function shows the image with the bounding box

        It also shows the heatmap
    """
    # To show the image:
    cv2.imshow('image', image)
    cv2.waitKey(0)

    # To show the heatmap:
    t = detections['ball_feature_map'][0][1, :, :]
    print("tensor shape:", t.shape)
    hmp = heatmap2image(t)

    cv2.imshow("heatmap", hmp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def testing_ball_feature_map():
    ball_feature_map = './txt_testing_files/ball_feature_map.pt'

    t = torch.load(ball_feature_map)
    print(t.shape)

    inspect_ball_feature_map(t)
    heatmap = ball_feature_map2heatmap(t)
    cv2.imshow("heatmap", heatmap)
    cv2.waitKey(0)


if __name__ == '__main__':
    print("Lets have a look at the model")
    params = Config()
    dataloader = make_dataloaders(params)
    run_model(dataloader)
    # testing_ball_feature_map()


# This is bad practice but I'll leave a test function down here. Should really make a separate file for this.
def test_read_txt_file() -> None:
    image_paths_list = read_txt_file(IMAGES_TXT)
    weight_paths_list = read_txt_file(WEIGHTS_TXT)
    print("Image paths: \n", image_paths_list, "\n weights paths:\n", weight_paths_list)
