import torch
import cv2
import os

from dataclasses import dataclass
from typing import List

from config import BaseConfig
import network.footandball as footandball
import data.augmentation as augmentations
from utils import draw_bboxes


"""
Functionality:
    
This file is for running inference of a trained model 

We should be able to specify which weights to use.

We should also be able to easily input either a single image, a list of images, a folder, or a video to perform 
inference on.    
"""


@dataclass
class InferenceConfig(BaseConfig):
    image_path: str = r"C:\\Users\\timf3\\PycharmProjects\\AFL-Data\\marvel\\afl-preprocessed\\train\\unpacked_png\\marvel_1_time_04_09_04_date_20_08_2023_0\\frame_0000515.png"
    # image_path: str = r"C:\\Users\\timf3\\OneDrive - Trinity College Dublin\\Documents\\Documents\\datasets\\Datasets\\Bohs\\bohs-preprocessed\\unpacked_jpg\\jetson1_date_24_02_2023_time__19_45_01_43\\frame_0000536.jpg"
    weights_path: str = r"models/model_14_10_2023__1948/model_14_10_2023__1948_200.pth"

    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    ball_threshold: float = 0.7

    def __post_init__(self):
        assert os.path.exists(self.image_path), f'Cannot find image_path: {self.image_path}'
        assert os.path.exists(self.weights_path), f'Cannot find BohsNet model weights_path: {config.weights_path}'


class RunDetector:
    def __init__(self, model, config: InferenceConfig, image_path: str):
        model.print_summary(show_architecture=False)
        self.model = model.to(config.device)
        self.config = config
        self.load_model_weights()

        self.image_path: str = image_path
        self.image = cv2.imread(self.image_path)

    def load_model_weights(self) -> None:
        if self.config.device == 'cpu':
            print('Loading CPU weights_path...')
            state_dict = torch.load(self.config.weights_path, map_location=lambda storage, loc: storage)
        else:
            print('Loading GPU weights_path...')
            state_dict = torch.load(self.config.weights_path)

        self.model.load_state_dict(state_dict)

    def run(self) -> None:

        img_tensor = augmentations.numpy2tensor(self.image)

        with torch.no_grad():
            # Add dimension for the batch size
            img_tensor = img_tensor.unsqueeze(dim=0).to(config.device)
            detections = model(img_tensor)[0]

        print(detections)
        image = draw_bboxes(self.image, detections)

        # Show the image
        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    config = InferenceConfig()
    model = footandball.model_factory("fb1", 'detect', ball_threshold=config.ball_threshold)

    detector = RunDetector(model, config, config.image_path)
    detector.run()

    del detector
