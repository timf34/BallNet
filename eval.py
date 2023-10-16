"""
What do we need:

- CVATEval
    - Encapsulates the Eval logic
- CVATDataLoader
- BallNet

Functionality:

- Load the dataloader and specify eval/ test mode (just do "test" for now)
"""
import torch

from dataclasses import dataclass
from typing import List, Tuple, Dict

from config import BaseConfig, AFLLaptopConfig
from data.cvat_dataloaders import make_data_loader
from network.footandball import model_factory
from data.cvat_utils import eval_single_frame
from utils import set_seed

SEED: int = 42
TOLERANCE: int = 5


@dataclass
class EvalConfig(AFLLaptopConfig):
    model_mode: str = "detect"
    batch_size: int = 1
    whole_dataset: bool = True
    use_augmentations: bool = False
    model_weights: str = "models/model_15_10_2023__1154/model_15_10_2023__1154_100.pth"


def prepare_eval_model(config: EvalConfig) -> torch.nn.Module:
    model = model_factory('fb1', config.model_mode, ball_threshold=config.ball_threshold)
    model.load_state_dict(torch.load(config.model_weights))
    model.eval()
    model.to(config.device)
    return model



# def box_to_xy(box: List[int]) -> Tuple[int, int]:
#     x1, y1, x2, y2 = box
#     return int((x1 + x2) / 2), int((y1 + y2) / 2)

def box_to_xy(box: List[int]) -> Tuple[int, int]:
    if len(box) != 4:
        box = box[0]
    x1, y1, x2, y2 = box
    x = int((x1 + x2) / 2)
    y = int((y1 + y2) / 2)
    return x, y

# TODO: this is a mess rn, will clean up later. Just get something basic working. Go for a walk, then come back and fix
#  things up properly. If this is the only thing I get done today, that's not too bad.

# TODO: note that there is still some empty labels in the val dataloader!! Need to look into this!

# Note: will clean things up afterwards here
def main():
    config = EvalConfig()
    data_loader = make_data_loader(config, modes=["val"], use_hardcoded_data_folders=False)
    model = prepare_eval_model(config)

    dataloader_keys = list(data_loader.keys())

    frame_stats = []

    for keys in dataloader_keys:
        for count, (image, boxes, labels) in enumerate(data_loader[keys]):
            image = image.to(config.device)
            detections = model(image)
            print(boxes)
            print(boxes[0].tolist())
            print(detections)

            if boxes[0].numel() == 0:
                # Pass if there are no detections
                continue

            gt_ball_pos = box_to_xy(boxes[0].tolist())

            if len(detections[0]['boxes']) == 0:
                pred_ball_pos = None

            # Note: this only works for the 1st prediction right now.
            for box, label, score in zip(detections[0]['boxes'], detections[0]['labels'], detections[0]['scores']):
                pred_ball_pos = [box_to_xy(box)]

            frame_stats.append(eval_single_frame(ball_pos=pred_ball_pos, gt_ball_pos=[gt_ball_pos], tolerance=5))


    print(frame_stats)

    percent_correctly_classified_frames = sum(
        c for (_, _, c) in frame_stats
    ) / len(frame_stats)
    temp = [p for (p, _, _) in frame_stats if p is not None]
    avg_precision = sum(temp) / len(temp)

    temp = [r for (_, r, _) in frame_stats if r is not None]
    avg_recall = sum(temp) / len(temp)

    print(f"Percent correctly classified frames: {percent_correctly_classified_frames}")
    print(f"Average precision: {avg_precision}")
    print(f"Average recall: {avg_recall}")


if __name__ == '__main__':
    set_seed(SEED)
    main()
