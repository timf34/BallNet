from typing import Tuple, List, Optional

import numpy as np
import torch
import os
from collections import defaultdict

from config import BaseConfig


class SequenceAnnotations:
    '''
    Class for storing annotations for the video sequence

    This should probably be replaced by a Python dataclass
    '''

    def __init__(self):
        # ball_pos contains list of ball positions (x,y) on each frame; multiple balls per frame are possible
        self.ball_pos = defaultdict(list)

def extract_coordinates(points) -> Tuple[int, int]:
    x, y = map(float, points.attrib['points'].split(','))
    return int(x), int(y)

def _load_cvat_groundtruth(xml_file_path: str) -> dict:
    """
    This function reads and laods the xlm file into a dictionary which we will
    then pass to _create_bohs_annotations...

    In general, this will read the groundtruth xml file, to extract frame and x-y position

    :params xml_file_path: path to the xml file
    :returns: dictionary of ball positions
    The structure of this dictionary is as follows:
        {'BallPos': [[382, 382, 437, 782]
        ... where the elements of the list are as follows [start_frame, end_frame, x, y]
    """

    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    # print("root.tage", root.tag)  # >>> annotations
    # print("atribbute", root.attrib)  # {}

    gt = {"BallPos": []}

    # Handle XML structure with "frame" attributes (from videos uploaded to CVAT)
    for child in root:
        for subchild in child:
            if "frame" in subchild.attrib:
                # print("frame:", subchild.attrib['frame'], "x:", subchild.attrib['points'].split(',')[0], "y:",  \
                # subchild.attrib['points'].split(',')[1])

                frame = int(subchild.attrib['frame'])
                x, y = extract_coordinates(subchild)

                # We put frame in twice simply to match FootAndBall dataloader
                gt["BallPos"].append([frame, frame, x, y])

    # Handle XML structure with "frame" attributes (from folders of images uploaded to CVAT)
    for image in root.findall('image'):
        image_id = int(image.attrib['id'])

        for points in image.findall('points'):
            if points.attrib['label'] == 'ball':
                # Extracting x, y coordinates from points
                x, y = extract_coordinates(points)
                # Append to the dictionary
                gt["BallPos"].append([image_id, image_id, x, y])
    return gt


def _create_bohs_annotations(gt: dict, frame_shape: Tuple[int, int] = (1080, 1920)) -> SequenceAnnotations:
    """
    Converts are groundtruth from _load_bohs_annotations to a SequenceAnnotations object

    :params groundtruth: dictionary of ball positions
    :params frame_shape: shape of the frame
    :returns: SequenceAnnotations

    """
    annotations = SequenceAnnotations()

    # print("max ball pos", max(gt["BallPos"])) # This prints [3594, 3594, 1667, 633], 3594 is max frame number

    for (start_frame, end_frame, x, y) in gt['BallPos']:
        for i in range(start_frame, end_frame + 1):
            annotations.ball_pos[i].append((x, y))

    # This is to fill in that there's no ball present for all the other frames
    # I am not sure if this is necessary.
    """ 
    for i in range(max(gt["BallPos"])[0]):
        if i not in annotations.ball_pos:
            annotations.ball_pos[i] = [[]]
    """

    return annotations


def read_cvat_ground_truth(annotations_path: str, xml_file_name: str) -> SequenceAnnotations:
    """
    Reads the groundtruth xml file and returns a SequenceAnnotations object

    :params annotations_path: path to the 'annotations' folder
    :params xml_file_name: name of the xml file
    """
    xml_file_path = os.path.join(annotations_path, xml_file_name)
    gt = _load_cvat_groundtruth(xml_file_path)
    return _create_bohs_annotations(gt)


def get_train_val_lengths(dataset: torch.utils.data.Dataset, config) -> Tuple[int, int]:
    """
        Returns two ints, the lengths of the train/ val dataset for splitting it
    """
    # Get the train and val dataset lengths
    train_dataset_length = int(config.train_size * len(dataset))
    val_dataset_length = int(config.val_size * len(dataset))
    total_train_val_length = train_dataset_length + val_dataset_length

    # Adjust dataset lengths so it equals total dataset length
    if total_train_val_length != len(dataset):
        train_dataset_length += (len(dataset) - total_train_val_length)

    # Ensure the new lengths (having added any difference to the trianing set) equal the total dataset length
    assert train_dataset_length + val_dataset_length == len(dataset), "train_dataset_length + val_dataset_length " \
                                                                      "!= len(dataset)"

    assert train_dataset_length > 0, "train_dataset_length is 0"
    assert val_dataset_length > 0, "val_dataset_length is 0. Check the train val split ratio"

    return train_dataset_length, val_dataset_length

def eval_single_frame(
        ball_pos: Optional[List[Tuple[int, int]]],
        gt_ball_pos: List[Tuple[int, int]],
        tolerance: int = 3
) -> Tuple[float, float, bool]:
    """
    Compute precision, recall, and a binary flag indicating whether the frame was
    correctly classified based on detected ball positions and ground truth ball positions.

    Parameters:
    - ball_pos (List[Tuple[int, int]]): A list of detected ball positions.
      Multiple balls may be detected in a frame.
    - gt_ball_pos (List[Tuple[int, int]]): A list of ground truth ball positions.
    - tolerance (int): The maximum allowed distance in pixels between detected and
      ground truth ball positions to consider a detection correct.

    Returns:
    - precision (float): The ratio of correctly detected balls to the total number of detections.
    - recall (float): The ratio of correctly detected balls to the total number of ground truth balls.
    - correctly_classified (bool): True if the frame has no false positives or false negatives, False otherwise.
    """

    def _dist(x1, x2):
        # Euclidean distance between two points
        return np.sqrt((float(x1[0]) - float(x2[0])) ** 2 + (float(x1[1]) - float(x2[1])) ** 2)

    # Handle None or empty list for ball_pos
    if ball_pos is None or not ball_pos:
        precision = None
        recall = 0 if gt_ball_pos else 1  # If there's no ground truth either, recall should be 1
        correctly_classified = not bool(gt_ball_pos)  # Only correct if there's no ground truth
        return precision, recall, correctly_classified

    # True positives, false positives and false negatives
    tp = 0
    fp = 0
    fn = 0

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

        if not hit:
            # Ball not detected - false negative
            fn += 1

    precision = None
    recall = None

    if tp + fp > 0:
        precision = tp / (tp + fp)

    if tp + fn > 0:
        recall = tp / (tp + fn)

    # Frame was correctly classified if there were no false positives and false negatives
    correctly_classified = (fp == 0 and fn == 0)

    return precision, recall, correctly_classified


def get_folders(dir: str) -> List[str]:
    """Returns a list of the names of the folders in the given directory"""
    return [f for f in os.listdir(dir) if os.path.isdir(os.path.join(dir, f))]


if __name__ == '__main__':
    print("oh no")
    print("oh yes")

