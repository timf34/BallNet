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
    folder_paths: List[str] = field(default_factory=lambda: [
        # r"C:\Users\timf3\PycharmProjects\AFL-Data\marvel\afl-preprocessed\val\unpacked_png\marvel_1_time_04_09_04_date_20_08_2023_2",
        # r"C:\Users\timf3\PycharmProjects\AFL-Data\marvel\afl-preprocessed\val\unpacked_png\marvel_3_time_04_09_06_date_20_08_2023_4",
        # r"C:\Users\timf3\PycharmProjects\AFL-Data\marvel\afl-preprocessed\val\unpacked_png\marvel_8_time_09_09_04_date_27_08_2023_1",
        # r"C:\Users\timf3\PycharmProjects\AFL-Data\marvel\afl-preprocessed\test\unpacked_png\marvel_1_time_04_09_04_date_20_08_2023_1",
        # r"C:\Users\timf3\PycharmProjects\AFL-Data\marvel\afl-preprocessed\test\unpacked_png\marvel_3_time_04_09_06_date_20_08_2023_6",
        # r"C:\Users\timf3\PycharmProjects\AFL-Data\marvel\afl-preprocessed\test\unpacked_png\marvel_8_time_09_09_04_date_27_08_2023_6",
        # r"C:\Users\timf3\PycharmProjects\AFL-Data\marvel\afl-preprocessed\val\unpacked_png\marvel_6_time_10_24_03_date_19_08_2023_7"
        r"C:\Users\timf3\PycharmProjects\AFL-Data\marvel\marvel-fov-3\20_08_2023\marvel_3_time_04_09_06_date_20_08_2023_\marvel_3_time_04_09_06_date_20_08_2023_4"
    ])

    image_paths: List[str] = field(default_factory=lambda: [
        r"C:\\Users\\timf3\\PycharmProjects\\AFL-Data\\marvel\\afl-preprocessed\\train\\unpacked_png\\marvel_1_time_04_09_04_date_20_08_2023_0\\frame_0000515.png",
        r"C:\\Users\\timf3\\PycharmProjects\\AFL-Data\\marvel\\afl-preprocessed\\train\\unpacked_png\\marvel_1_time_04_09_04_date_20_08_2023_0\\frame_0001527.png",
        r"C:\Users\timf3\PycharmProjects\AFL-Data\marvel\afl-preprocessed\train\unpacked_png\marvel_1_time_04_09_04_date_20_08_2023_0\frame_0001528.png",
        r"C:\Users\timf3\PycharmProjects\AFL-Data\marvel\afl-preprocessed\train\unpacked_png\marvel_1_time_04_09_04_date_20_08_2023_3\frame_0001547.png",
        r"C:\Users\timf3\PycharmProjects\AFL-Data\marvel\afl-preprocessed\train\unpacked_png\marvel_1_time_04_09_04_date_20_08_2023_3\frame_0001548.png",
        r"C:\Users\timf3\PycharmProjects\AFL-Data\marvel\afl-preprocessed\train\unpacked_png\marvel_1_time_04_09_04_date_20_08_2023_3\frame_0001549.png"
    ])

    video_paths: List[str] = field(default_factory=lambda: [
        # r"C:\Users\timf3\PycharmProjects\AFL-Data\marvel\marvel-fov-5\18_08_2023\marvel_5_time_10_24_03_date_19_08_2023_.avi"
    ]
                                   )
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
        self.folder_path: List[str] = self.config.folder_paths
        self.video_paths: List[str] = self.config.video_paths

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

    def run_on_image_folders(self, folder_path: str, save_video: bool = False) -> None:
        """
        This function iterates through a folder containing images, runs inference on each image,
        and saves them to a folder.
        """
        assert os.path.isdir(folder_path), f'Folder path {folder_path} does not exist'

        # Getting list of image files in the folder
        image_files = [f for f in os.listdir(folder_path) if f.endswith('.png') or f.endswith('.jpg')]
        image_files.sort()  # Ensure the images are processed in order

        # Video writer initialization
        video_writer = None
        if save_video:
            # Example configuration, adjust according to your needs
            frame_size = (1920, 1080)  # Adjust this based on your image dimensions
            video_writer = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, frame_size)

        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (1920, 1080))

            if image is None:
                print(f'Could not read image at {image_path}')
                continue

            img_tensor = augmentations.numpy2tensor(image)

            with torch.no_grad():
                # Add dimension for the batch size
                img_tensor = img_tensor.unsqueeze(dim=0).to(self.config.device)
                detections = self.model(img_tensor)[0]

            processed_image = draw_bboxes(image, detections)

            if save_video:
                video_writer.write(processed_image)
            else:
                cv2.imshow('Processed Image', processed_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        if video_writer is not None:
            video_writer.release()

        cv2.destroyAllWindows()

    def run_on_video(self, video_path: str, save_video: bool = False) -> None:
        """
        This function processes a video file, runs inference on each frame,
        and optionally saves the output as a new video file.
        """
        assert os.path.isfile(video_path), f'Video file {video_path} does not exist'

        cap = cv2.VideoCapture(video_path)

        # Video writer initialization
        video_writer = None
        if save_video:
            # Adjust frame size and FPS according to the input video
            frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            fps = cap.get(cv2.CAP_PROP_FPS)
            video_writer = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, frame_size)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            img_tensor = augmentations.numpy2tensor(frame)
            with torch.no_grad():
                img_tensor = img_tensor.unsqueeze(dim=0).to(self.config.device)
                detections = self.model(img_tensor)[0]

            processed_frame = draw_bboxes(frame, detections)

            if save_video:
                video_writer.write(processed_frame)
            else:
                cv2.imshow('Processed Frame', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()

    def run(self) -> None:
        # for image_path in self.image_path:
        #     self.run_on_image(image_path)

        # for folder_path in self.folder_path:
        #     self.run_on_image_folders(folder_path, save_video=False)

        for video_path in self.video_paths:
            self.run_on_video(video_path, save_video=False)


if __name__ == '__main__':
    config = InferenceConfig()
    model = footandball.model_factory("fb1", 'detect', ball_threshold=config.ball_threshold)

    detector = RunDetector(model, config)
    detector.run()

    del detector
