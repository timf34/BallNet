import torch
import cv2
import os

from dataclasses import dataclass, field
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
    image_paths: List[str] = field(default_factory=lambda: [
        r"C:\\Users\\timf3\\PycharmProjects\\AFL-Data\\marvel\\afl-preprocessed\\train\\unpacked_png\\marvel_1_time_04_09_04_date_20_08_2023_0\\frame_0000515.png",
        r"C:\\Users\\timf3\\PycharmProjects\\AFL-Data\\marvel\\afl-preprocessed\\train\\unpacked_png\\marvel_1_time_04_09_04_date_20_08_2023_0\\frame_0001527.png",
        r"C:\Users\timf3\PycharmProjects\AFL-Data\marvel\afl-preprocessed\train\unpacked_png\marvel_1_time_04_09_04_date_20_08_2023_0\frame_0001528.png",
        r"C:\Users\timf3\PycharmProjects\AFL-Data\marvel\afl-preprocessed\train\unpacked_png\marvel_1_time_04_09_04_date_20_08_2023_3\frame_0001547.png",
        r"C:\Users\timf3\PycharmProjects\AFL-Data\marvel\afl-preprocessed\train\unpacked_png\marvel_1_time_04_09_04_date_20_08_2023_3\frame_0001548.png",
        r"C:\Users\timf3\PycharmProjects\AFL-Data\marvel\afl-preprocessed\train\unpacked_png\marvel_1_time_04_09_04_date_20_08_2023_3\frame_0001549.png"
    ])
    # If provided, replaces image_paths with all PNG files in the directory
    # input_image_txt_list: str = r"C:\Users\timf3\PycharmProjects\AFL-Data\marvel\afl-preprocessed\train\images_lists\marvel_1_time_04_09_04_date_20_08_2023_4.txt"
    input_image_txt_list: str = r"C:\Users\timf3\PycharmProjects\AFL-Data\marvel\afl-preprocessed\val\image_lists\marvel_3_time_04_09_06_date_20_08_2023_4.txt"


    weights_path: str = r"models/model_15_10_2023__1154/model_15_10_2023__1154_100.pth"

    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    ball_threshold: float = 0.7

    def __post_init__(self):
        if self.input_image_txt_list:
            with open(self.input_image_txt_list, 'r') as f:
                self.image_paths.clear()
                self.image_paths = [line.strip() for line in f]

        for image_path in self.image_paths:
            assert os.path.exists(image_path), f'Cannot find image_path: {image_path}'
        assert os.path.exists(self.weights_path), f'Cannot find BohsNet model weights_path: {config.weights_path}'


class RunDetector:
    def __init__(self, model, config: InferenceConfig):
        model.print_summary(show_architecture=False)
        self.model = model.to(config.device)
        self.config = config
        self.load_model_weights()

        self.image_path: List[str] = self.config.image_paths

    def load_model_weights(self) -> None:
        if self.config.device == 'cpu':
            print('Loading CPU weights_path...')
            state_dict = torch.load(self.config.weights_path, map_location=lambda storage, loc: storage)
        else:
            print('Loading GPU weights_path...')
            state_dict = torch.load(self.config.weights_path)

        self.model.load_state_dict(state_dict)

    def run_on_image(self, image_path: str) -> None:
        image = cv2.imread(image_path)
        img_tensor = augmentations.numpy2tensor(image)

        with torch.no_grad():
            # Add dimension for the batch size
            img_tensor = img_tensor.unsqueeze(dim=0).to(config.device)
            detections = model(img_tensor)[0]

        image = draw_bboxes(image, detections)

        # Show the image
        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def run(self) -> None:
        for image_path in self.image_path:
            self.run_on_image(image_path)

if __name__ == '__main__':
    config = InferenceConfig()
    model = footandball.model_factory("fb1", 'detect', ball_threshold=config.ball_threshold)

    detector = RunDetector(model, config)
    detector.run()

    del detector
