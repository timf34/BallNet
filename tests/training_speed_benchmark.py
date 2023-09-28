import torch, time, gc

from typing import Dict, List

from config import Config
from network.ssd_loss import SSDLoss
from data.data_reader import make_train_val_dataloaders  # Note: I might want to consider training on random tensors...
from scripts.run_and_eval_utils import prep_model

start_time = None


class BenchmarkingConfig(Config):
    def __init__(self):
        super().__init__()
        self.batch_size: int = 1
        self.whole_dataset: bool = False
        self.phases: List[str] = ['train', 'val']
        self.epochs: int = 10
        self.num_workers: int = 1
        self.weights_path: str = r'C:\Users\timf3\Downloads\model (1).tar\model (1)\model_22_11_2022__0202\model_22_11_2022__0202_90.pth'
        self.whole_dataset: bool = True
        self.model_mode: str = 'train'  # Note: in 'detect' mode, model will return dets in a list, not a tensor, which won't work with SSDLoss.
        self.use_amp: bool = True

        # Overriding defaults we care about
        self.aws: bool = False
        self.aws_testing: bool = False


# From https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
def start_timer() -> None:
    print("About to start timing...")
    global start_time
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    start_time = time.time()


def end_timer_and_print(local_msg):
    torch.cuda.synchronize()
    end_time = time.time()
    print("\n" + local_msg)
    print("Total execution time = {:.3f} sec".format(end_time - start_time))
    print("Max memory used by tensors = {} bytes \n\n".format(torch.cuda.max_memory_allocated()))


def train_with_default_precision() -> None:
    # Initialize our config, model and dataloaders
    args = BenchmarkingConfig()
    model = prep_model(args, print_summary=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    dataloaders = make_train_val_dataloaders(args)
    criterion = SSDLoss(neg_pos_ratio=args.neg_pos_ratio)

    start_timer()
    for epoch in range(args.epochs):
        for count, (images, boxes, labels) in enumerate(dataloaders["train"]):
            # Prep the data
            images = images.to(args.device)
            h, w = images.shape[-2], images.shape[-1]
            gt_maps = model.groundtruth_maps(boxes, labels, (h, w))
            gt_maps = [e.to(args.device) for e in gt_maps]

            with torch.set_grad_enabled(True):
                output = model(images)
                loss = criterion(output, gt_maps)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if count > 10:
                break
    end_timer_and_print("Default precision training:")


def train_with_amp(message: str = "Mixed precision training...", cudnn_autotuner: bool = True) -> None:
    # Initialize our config, model and dataloaders
    args = BenchmarkingConfig()
    model = prep_model(args, print_summary=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    dataloaders = make_train_val_dataloaders(args)
    criterion = SSDLoss(neg_pos_ratio=args.neg_pos_ratio)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    if cudnn_autotuner:
        torch.backends.cudnn.benchmark = True

    start_timer()
    for epoch in range(args.epochs):
        for count, (images, boxes, labels) in enumerate(dataloaders["train"]):
            # Prep the data
            images = images.to(args.device)
            h, w = images.shape[-2], images.shape[-1]
            gt_maps = model.groundtruth_maps(boxes, labels, (h, w))
            gt_maps = [e.to(args.device) for e in gt_maps]

            with torch.set_grad_enabled(True):
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    output = model(images)
                    loss = criterion(output, gt_maps)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if count > 10:
                break
    end_timer_and_print(message)


def main():
    for i in range(2):
        train_with_amp(cudnn_autotuner=False)
        train_with_amp(message="Mixed precision with autotuner training...", cudnn_autotuner=True)
        train_with_default_precision()


if __name__ == "__main__":
    main()
