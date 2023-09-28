import torch
import cv2
import os
import xml.etree.ElementTree as ET

from typing import List

from config import BALL_LABEL
import network.footandball as footandball
import data.augmentation as augmentations
from scripts.run_and_eval_utils import draw_bboxes, create_dir, get_file_name

"""
    This file is just for running the trained model on unseen video, and visualizing the results
    
    It saves the annotaed videos to ./data/processed_annotated_videos/{weights_file_name}/{weights_file_name}_{video}
"""


class RunDetectorArgs:
    def __init__(self,
                 output_xml_folder: str = r"C:\Users\timf3\PycharmProjects\BohsNet\data\processed_bohsnet_xml_output",
                 weights_path: str = r"models/model_06_03_2023__0757_35.pth",  # Middle camera position
                 # weights_path: str = r'C:\Users\timf3\PycharmProjects\BohsNet\models\model_22_11_2022__0202_90.pth',  # Jetson3
                 # weights_path: str = r'C:\Users\timf3\PycharmProjects\BohsNet\models\model_29_11_2022__1214_58.pth',  # Jetson1
                 output_video_folder: str = r"C:\Users\timf3\PycharmProjects\BohsNet\data\processed_annotated_videos",
                 ):
        self.video_paths: List[str] = [
            # r"C:\Users\timf3\OneDrive - Trinity College Dublin\Documents\Documents\datasets\Datasets\Bohs\1_4_22\Jetson3\jetson3_1_4_2022_time__20_40_14\25.mp4",
            # r"C:\Users\timf3\OneDrive - Trinity College Dublin\Documents\Documents\datasets\Datasets\Bohs\1_4_22\Jetson1\jetson1_date_01_04_2022_time__20_40_14\25.mp4"
            r"C:\Users\timf3\OneDrive - Trinity College Dublin\Documents\Documents\datasets\Datasets\Bohs\24_2_23\jetson1_date_24_02_2023_time__19_45_01\16.mp4"
            # r"C:\Users\timf3\PycharmProjects\BohsNetEvals\data\1_5_23\jetson1\time_17_01_06_date_01_05_2023_.avi"
        ]

        self.weights_path: str = weights_path
        self.output_video_folder: str = output_video_folder
        self.output_xml_folder: str = output_xml_folder

        self.ball_threshold: float = 0.7
        self.device: str = "cuda:0"

        # Check if the output video already exists, if so, don't run on it.
        for video_name in self.video_paths:

            output_video_name = get_file_name(self.weights_path, video_name)

            # Get the name of the weights file
            weights_file_name = self.weights_path.split("\\")[-1].split(".")[0]

            # Check if the video name exists in the output video folder
            if os.path.exists(os.path.join(self.output_video_folder, weights_file_name, output_video_name)):
                print(f"Video name {output_video_name} exists")
                # Remove video name from self.video_paths
                self.video_paths.remove(video_name)
                # TODO: This is poorly designed in the scenario where we have the video, but we don't have the dataset, it won't run
                #  This also doesn't check if we have the _full_ video (it could just be an incomplete 5 second video).
                print(f"Video name {video_name} removed from self.video_paths")
            else:
                print(f"Video name {output_video_name} does not exist... we'll run inference on this video and save it")

    def print(self):
        for i in self.__dict__:
            print(f"{i}: {self.__dict__[i]}")


class XMLBohsNetDataset:
    def __init__(self):
        # These are all defined in the methods
        self.file_path: str = None
        self.root = None
        self.child = None
        self.subchild = None

    def create_xml_file(self, file_name: str, weights_path: str, output_xml_folder) -> None:
        # TODO: saving the xml file is a mess right now, clean this up later.
        xml_file_name = file_name.split("\\")[-1].split(".")[0].split(".")[0] + ".xml"
        video_name = file_name.split("\\")[-2]
        weights_file_name = weights_path.split("\\")[-1].split(".")[0]
        output_file_name = f"{weights_file_name}_{video_name}_{xml_file_name}"
        self.file_path = os.path.join(output_xml_folder, output_file_name)
        # Create file path
        with open(self.file_path, "w") as f:
            f.write("")
        root = ET.Element("annotations")
        tree = ET.ElementTree(root)
        tree.write(self.file_path)
        # Ensure that file exists before opening it
        if os.path.exists(self.file_path):
            self.open()
        else:
            print(f"File {self.file_path} does not exist")
            raise Exception

    def open(self) -> None:
        tree = ET.parse(self.file_path)
        self.root = tree.getroot()
        self.child = ET.SubElement(self.root, "frame", frame="1")

    def add_detections_to_xml(self, detections, frame_number: int) -> None:
        # Check if dets is empty
        for box, label, score in zip(detections['boxes'], detections['labels'], detections['scores']):
            if label == BALL_LABEL:
                x1, y1, x2, y2 = box
                x = int((x1 + x2) / 2)
                y = int((y1 + y2) / 2)
                # self.subchild = ET.SubElement(self.child, "frame", frame=str(frame_number))
                # self.subchild = ET.SubElement(self.child, "ball", points=f"{x},{y}")
                # Merge the above two lines into one
                # self.subchild = ET.SubElement(self.child, "ball", frame=str(frame_number), points=f"{x},{y}")
                self.subchild = ET.SubElement(self.child, "ball", frame=str(frame_number), points=f"{x},{y}",
                                              score=str(round(float(score), 4)))

    def write_to_file(self) -> None:
        tree = ET.ElementTree(self.root)
        tree.write(self.file_path)


class RunDetector:
    def __init__(self, model, args: RunDetectorArgs, video_path: str, xml_manager: XMLBohsNetDataset):
        model.print_summary(show_architecture=False)
        self.model = model.to(args.device)
        self.args = args
        self.load_model_weights()

        # Make the folder and get the output file name
        self.video_path: str = video_path
        self.output_video_name = get_file_name(args.weights_path, self.video_path)
        self.output_video_path = create_dir(args, file_name=self.output_video_name)

        self.xml_manager = xml_manager

    def load_model_weights(self) -> None:
        if self.args.device == 'cpu':
            print('Loading CPU weights_path...')
            state_dict = torch.load(args.weights_path, map_location=lambda storage, loc: storage)
            state_dict = torch.load(args.weights_path, map_location=lambda storage, loc: storage)
        else:
            print('Loading GPU weights_path...')
            state_dict = torch.load(args.weights_path)

        self.model.load_state_dict(state_dict)

    def run(self, save_video: bool = True, create_xml: bool = True) -> None:

        # Create the xml file for the video
        if create_xml:
            self.xml_manager.create_xml_file(self.video_path.replace(".mp4", ".xml"),
                                             weights_path=self.args.weights_path,
                                             output_xml_folder=self.args.output_xml_folder)

        sequence = cv2.VideoCapture(self.video_path)
        fps = sequence.get(cv2.CAP_PROP_FPS)
        (frame_width, frame_height) = (int(sequence.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                       int(sequence.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        if save_video:
            out_sequence = cv2.VideoWriter(self.output_video_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps,
                                           (frame_width, frame_height))

        print(f'Processing video: {self.video_path}')

        frame_number = 0

        while sequence.isOpened():
            ret, frame = sequence.read()
            if not ret:
                # End of video
                break

            # Convert color space from BGR to RGB, convert to tensor and normalize
            img_tensor = augmentations.numpy2tensor(frame)

            with torch.no_grad():
                # Add dimension for the batch size
                img_tensor = img_tensor.unsqueeze(dim=0).to(args.device)
                detections = model(img_tensor)[0]
                if create_xml:
                    self.xml_manager.add_detections_to_xml(detections, frame_number)

            frame = draw_bboxes(frame, detections)
            if save_video:
                out_sequence.write(frame)

            frame_number += 1

        sequence.release()
        if save_video:
            out_sequence.release()
        if create_xml:
            self.xml_manager.write_to_file()
        print(f'Finished processing video: {self.video_path}')
        print("Run detector done:)")


if __name__ == '__main__':
    print('Run FootAndBall detector on input video')

    args = RunDetectorArgs(output_xml_folder=r"C:\Users\timf3\PycharmProjects\BohsNet\data\processed_bohsnet_xml_output")
    print(f"Args: \n")
    args.print()

    xml_manager = XMLBohsNetDataset()

    assert os.path.exists(args.weights_path), f'Cannot find BohsNet model weights_path: {args.weights_path}'

    for video in args.video_paths:
        assert os.path.exists(video), f'Cannot open video: {video}'

    model = footandball.model_factory("fb1", 'detect', ball_threshold=args.ball_threshold)

    for video in args.video_paths:
        detector = RunDetector(model, args, xml_manager=xml_manager, video_path=video)

        # TODO: note there's an issue with the xml_manager in creating the file.
        detector.run(save_video=True, create_xml=False)
        # Clear detector
        del detector
