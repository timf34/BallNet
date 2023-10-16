import torch

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

from config import BaseConfig, AFLLaptopConfig
from data.cvat_dataloaders import make_data_loader
from network.footandball import model_factory
from data.cvat_utils import eval_single_frame
from utils import set_seed, box_to_xy

SEED: int = 42
TOLERANCE: int = 5

# TODO: note that there is still some empty labels in the test dataloader!! Need to look into this!

# TODO: I should harmonize the data types here more

@dataclass
class EvalConfig(AFLLaptopConfig):
    model_mode: str = "detect"
    data_loader_mode: str = "val"
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


def get_ball_positions(
        detections: List[Dict[str, torch.Tensor]],
        boxes: List[torch.Tensor]
) -> Tuple[Optional[List[Tuple[int, int]]], Optional[List[Tuple[int, int]]]]:
    if boxes[0].numel() == 0:
        return None, None

    gt_ball_pos = box_to_xy(boxes[0].tolist()[0])

    if detections[0]['boxes'].shape[0] == 0:
        return gt_ball_pos, None

    # Get first prediction for now
    pred_ball_pos = []
    for dets in detections:
        pred_ball_pos.append(box_to_xy(dets['boxes'][0].tolist()))
    return [gt_ball_pos], pred_ball_pos


def evaluate_frames(
        dataloader: Dict[str, torch.utils.data.DataLoader],
        config: EvalConfig,
        model: torch.nn.Module
) -> List[Tuple[float, float, bool]]:
    frame_stats = []

    for image, boxes, labels in dataloader[config.data_loader_mode]:
        image = image.to(config.device)
        detections = model(image)
        gt_ball_pos, pred_ball_pos = get_ball_positions(detections, boxes)
        frame_stats.append(eval_single_frame(ball_pos=pred_ball_pos, gt_ball_pos=gt_ball_pos, tolerance=TOLERANCE))

    return frame_stats


def print_evaluation_metrics(frame_stats):
    correct_classifications = [c for (_, _, c) in frame_stats]
    precisions = [p for (p, _, _) in frame_stats if p is not None]
    recalls = [r for (_, r, _) in frame_stats if r is not None]

    percent_correctly_classified_frames = sum(correct_classifications) / len(frame_stats)
    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)

    print(f"Percent correctly classified frames: {percent_correctly_classified_frames}")
    print(f"Average precision: {avg_precision}")
    print(f"Average recall: {avg_recall}")


def main():
    config = EvalConfig()
    data_loader = make_data_loader(config, modes=[config.data_loader_mode], use_hardcoded_data_folders=False)
    model = prepare_eval_model(config)
    frame_stats = evaluate_frames(data_loader, config, model)

    assert len(frame_stats) == data_loader[config.data_loader_mode].__len__(), "Frame stats and dataloader length do not match!"

    print_evaluation_metrics(frame_stats)


if __name__ == '__main__':
    set_seed(SEED)
    main()
