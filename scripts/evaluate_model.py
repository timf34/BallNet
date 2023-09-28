import torch
import cv2
import os

from typing import Dict, List, Tuple

import network.footandball as footandball
from scripts.run_and_eval_utils import draw_bboxes, prep_model, AverageMeter, _ball_detection_stats
from data.data_reader import make_train_val_dataloaders
from data.augmentation import tensor2image
from config import Config

torch.manual_seed(42)

"""
    This script runs evaluation on the trained models, getting an accuracy and precision score. 
    
    It saves the output video (with dets and frame no.) to the output_video_folder, if specified.
"""


class EvaluationArgs(Config):
    def __init__(self):
        super().__init__()
        self.weights_path: str = r'C:\Users\timf3\Downloads\model (1).tar\model (1)\model_22_11_2022__0202\model_22_11_2022__0202_90.pth'
        self.output_video_folder: str = r"C:\Users\timf3\PycharmProjects\BohsNet\data\processed_annotated_videos"
        self.ball_threshold: float = 0.7
        self.device: str = "cuda:0"
        self.use_augmentation: bool = False
        self.save_video: bool = True
        self.model_mode: str = 'detect'

        # Overriding defaults we care about
        self.aws: bool = False
        self.aws_testing: bool = False
        self.batch_size: int = 1
        self.whole_dataset: bool = True

    def print(self):
        for i in self.__dict__:
            print(f"{i}: {self.__dict__[i]}")


def get_ball_list_from_dets(detections) -> List[Tuple[int, int]]:
    """
    This function takes in a list of detections, of the form Tensor.size(n,4) where n is the number of detections,
    and returns a list of tuples of the form (x, y) for each detection.
    :param detections: Tensor.size(n,4)
    :return: List[Tuple[int, int]]
    """
    dets = []
    for box in detections:
        if len(box) == 4:
            x1, y1, x2, y2 = box
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2
            x, y = int(x), int(y)
            dets.append((x, y))
        else:
            print(box)
            # Check if the box is tensor of size=(0, 4))]
    return dets


def evaluate_model_(args):
    # Load the model
    model = prep_model(args)
    model.eval()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load the dataloader
    dataloaders = make_train_val_dataloaders(args)
    val_dataloader = dataloaders["val"]

    # Initialize the metrics list
    frame_stats = []

    # Initialize the output video
    if args.output_video_folder and args.save_video:
        output_video_path = os.path.join(args.output_video_folder, "evaluation_data_model_22_11_2022__0202_90.mp4")
        if not os.path.exists(args.output_video_folder):
            os.makedirs(args.output_video_folder)
        output_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 1, (1920, 1080))

    # Iterate through the validation dataloader
    for count, (images, boxes, labels) in enumerate(val_dataloader):
        images = images.to(device)

        # Boxes is nested in a double list List[List[Tensor]], so we need to get the first element... quick fix for now.
        boxes = get_ball_list_from_dets(boxes[0])

        with torch.no_grad():
            dets: Dict[str: torch.Tensor] = model(images)[0]

            # If there's a detection...
            if len(dets["boxes"]) != 0:
                ball_dets: List[Tuple[int, int]] = get_ball_list_from_dets(dets["boxes"])
            else:
                # No detections
                ball_dets = []
            frame_stats.append(_ball_detection_stats(ball_dets, boxes))

        image = tensor2image(images[0])
        image = (image * 255).astype("uint8")
        frame = draw_bboxes(image, dets)

        if args.output_video_folder and args.save_video:
            cv2.putText(frame, str(count), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Write frame number
            output_video.write(frame)

    if args.save_video:
        output_video.release()

    percent_correctly_classified_frames = sum(c for (_, _, c) in frame_stats) / len(frame_stats)

    temp = [p for (p, _, _) in frame_stats if p is not None]
    avg_precision = sum(temp)/len(temp)

    temp = [r for (_, r, _) in frame_stats if r is not None]
    avg_recall = sum(temp) / len(temp)

    print(f"Accuracy: {percent_correctly_classified_frames}")
    print(f"Precision: {avg_precision}")
    print(f"Recall: {avg_recall}")

    with open("../data/val_dataloader_eval_results/evaluation_data_model_22_11_2022__0202_90.txt", "w") as f:
        f.write(f"Weights: {args.weights_path}\n")
        f.write(f"Accuracy: {percent_correctly_classified_frames}\n")
        f.write(f"Precision: {avg_precision}\n")
        f.write(f"Recall: {avg_recall}\n")


if __name__ == '__main__':
    print("Evaluate script")

    args = EvaluationArgs()
    print("Args: \n")
    args.print()

    assert args.use_augmentation == False, "Augmentation is not supported for evaluation"

    model = footandball.model_factory(args.model, 'detect', ball_threshold=args.ball_threshold)

    # evaluate_model(camera_id, DATASET_PATH, args, gt_annotations)
    evaluate_model_(args)

