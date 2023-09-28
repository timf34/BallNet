"""
What do we need:

- BohsDataset: object which allows me to specify which dataset I want to use (so I can use a specific camera) (might need slightl adjustment)
    - Ok so we basically have this right now, I can add better features later.
- BohsNet: the CNN (done)
    - Let's get this bread.
- BohsEval: class which encapsulates the evaluation process (including measuring metrics) (a good bit of this is already done)

"""
import torch

from typing import List, Tuple, Dict

from config import Config
from data.data_reader import make_dataloaders, make_eval_dataloader
from network.footandball import model_factory
from data.bohs_utils import eval_single_frame

class EvalConfig(Config):
    def __init__(self):
        super().__init__()
        self.model_mode: str = 'detect'

        # Overriding defaults we care about
        self.aws: bool = False
        self.aws_testing: bool = False
        self.batch_size: int = 1
        self.whole_dataset: bool = True

def prepare_model(config: EvalConfig):
    model = model_factory(config.model, config.model_mode, ball_threshold=config.ball_threshold)
    model.eval()
    model.to(config.device)
    return model

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

def main():
    config = EvalConfig()
    eval_dataloader = make_eval_dataloader(config)
    model = prepare_model(config)

    frame_stats = []

    for count, (images, boxes, labels) in enumerate(eval_dataloader):
        imgaes = images.to(config.device)
        boxes = get_ball_list_from_dets()

        with torch.no_grad():
            dets = model(images)[0]

            if len(dets["boxes"]) != 0:
                ball_dets: List[Tuple[int, int]] = get_ball_list_from_dets(dets["boxes"])
            else:
                # No detections
                ball_dets = []
            frame_stats.append(eval_single_frame(ball_dets, boxes, tolerance=3))



if __name__ == '__main__':
    main()
