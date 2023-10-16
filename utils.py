import cv2
import numpy as np
import os
import random
import torch

from typing import List

from config import BaseConfig


BALL_LABEL = 1


def create_directory(directory: str) -> None:
    """Ensure the directory exists. If not, create it."""
    if not os.path.exists(directory):
        os.mkdir(directory)
        print(f"Folder {directory} was created")
    else:
        print(f"Folder {directory} already exists")


def save_weights_to_path(model, filepath: str) -> None:
    """Save model weights to the given filepath if they don't exist already."""
    if os.path.exists(filepath):
        print(f'WARNING: Model weights already exist at {filepath}... not overwriting')
        return
    print(f"Saving model weights to: {filepath}")
    torch.save(model.state_dict(), filepath)


def get_model_filepath(base_folder: str, model_name: str, epoch: int) -> str:
    """Get the file path for the model weights."""
    model_folder = os.path.join(base_folder, model_name)
    create_directory(model_folder)
    return os.path.join(model_folder, f'{model_name}_{epoch}.pth')


def save_model_weights(model, epoch: int, config: BaseConfig) -> None:
    # TODO: saving twice to AWS. Is this necessary? Almost definitely not.
    #  self.checkpoints_folder = '/opt/ml/checkpoints' and  model_folder = os.environ['SM_MODEL_DIR']
    """
    Saves once locally to main_folder and also again to checkpoints_folder if aws is enabled.
    """
    main_filepath = get_model_filepath(config.model_folder, config.model_name, epoch)
    save_weights_to_path(model, main_filepath)

    if config.aws:
        checkpoint_filepath = get_model_filepath(config.checkpoints_folder, config.model_name, epoch)
        save_weights_to_path(model, checkpoint_filepath)


# TODO: add typing
def draw_bboxes(image, detections):
    """
    Draw bounding boxes on the image
    :param image: image to draw bounding boxes on, numpy array
    :param detections: dictionary with bounding boxes
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    for box, label, score in zip(detections['boxes'], detections['labels'], detections['scores']):
        if label == BALL_LABEL:
            x1, y1, x2, y2 = box
            x = int((x1 + x2) / 2)
            y = int((y1 + y2) / 2)
            color = (0, 0, 255)
            radius = 12
            cv2.circle(image, (int(x), int(y)), radius, color, 2)
            cv2.putText(image, '{:0.2f}'.format(score), (max(0, int(x - radius)), max(0, (y - radius - 10))), font, 1, color, 2)
    return image

def set_seed(seed=42) -> None:
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def box_to_xy(box: List[int]) -> Tuple[int, int]:
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)
