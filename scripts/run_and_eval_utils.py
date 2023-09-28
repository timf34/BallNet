import cv2
import numpy as np
import os
import torch

from typing import List, Tuple

import network.footandball as footandball
from data.augmentation import BALL_LABEL


# Note: currently not using this... its not particularly useful rn.
class AverageMeter(object):
    # From: https://github.com/rwightman/pytorch-image-models/blob/main/timm/utils/metrics.py
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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
            radius = 25
            cv2.circle(image, (int(x), int(y)), radius, color, 2)
            cv2.putText(image, '{:0.2f}'.format(score), (max(0, int(x - radius)), max(0, (y - radius - 10))), font, 1, color, 2)

    return image


def get_file_name(weights_path: str, video_path: str) -> str:
    """
    Get the file name from an image path
    :param weights_path: path to the weights
    :param video_path: path to the video
    :return: filename
    """
    weights_file_name = os.path.basename(weights_path).split('.')[0]
    video_folder = os.path.basename(os.path.dirname(video_path))
    return f"{weights_file_name}_{video_folder}_{os.path.basename(video_path)}"


def create_dir(args, file_name: str) -> str:
    """
    Create the directory and file name for the annotated video.
    :param args: config args
    :param file_name: name of the video
    :return: path for the annotated video
    """
    # Create the output video name
    weights_file_name = os.path.basename(args.weights_path).split('.')[0]

    output_folder = os.path.join(args.output_video_folder, weights_file_name)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    return os.path.join(output_folder, file_name)


def prep_model(args, print_summary: bool = True):
    model = footandball.model_factory(args.model, args.model_mode, ball_threshold=args.ball_threshold)
    if print_summary:
        model.print_summary(show_architecture=False)
    model = model.to(args.device)

    if args.device == 'cpu':
        print('Loading CPU weights_path...')
        state_dict = torch.load(args.weights_path, map_location=lambda storage, loc: storage)
    else:
        print('Loading GPU weights_path...')
        state_dict = torch.load(args.weights_path)

    model.load_state_dict(state_dict)
    return model


def _ball_detection_stats(ball_pos: List[Tuple[int, int]], gt_ball_pos: List[Tuple[int, int]], tolerance: int = 3):
    '''
    Compute ball detection stats for the single frame
    :param ball_pos: A list of detected ball positions. Multiple balls are possible.
    :param gt_ball_poss: A list of ground truth ball positions. Multiple balls are possible.
    :param tolerance: tolerance for ball centre tolerance in pixels
    :return: A tuple (precision, recall, number of correctly_classified frames)
    '''

    # True positives, false positives and false negatives
    tp = 0
    fp = 0
    fn = 0

    # Another count of true positives based on enumerating ground truth detections
    # If tp != tp1 this means that more than one ball was detected for one ground truth ball
    tp1 = 0

    # For each detected ball, check if it corresponds to a ground truth ball
    for e in ball_pos:
        # Verify if it's a true positive or a false positive
        hit = False
        for gt_e in gt_ball_pos:
            if _dist(e, gt_e) <= tolerance:
                # Matching ground truth ball position
                hit = True
                break

        if hit:
            # Ball correctly detected - true positive
            tp += 1
        else:
            # Ball incorrectly detected - false positive
            fp += 1

    # For each ground truth  ball, check if it was detected
    for gt_e in gt_ball_pos:
        # Verify if it's a false negative
        hit = False
        for e in ball_pos:
            if _dist(e, gt_e) <= tolerance:
                # Matching ground truth ball position
                hit = True
                break

        if hit:
            tp1 += 1
        else:
            # Ball not detected - false negative
            fn += 1

    precision = None
    recall = None

    if tp+fp > 0:
        precision = tp/(tp+fp)

    if tp+fn > 0:
        recall = tp/(tp+fn)

    # Frame was correctly classified if there were no false positives and false negatives
    correctly_classified = (fp == 0 and fn == 0)

    return precision, recall, correctly_classified


def _dist(x1: Tuple[float, float], x2: Tuple[float, float]):
    # Euclidean distance between two points
    return np.sqrt((float(x1[0])-float(x2[0]))**2 + (float(x1[1])-float(x2[1]))**2)
