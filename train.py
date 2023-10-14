from tqdm import tqdm
import pickle
import os
import time
import wandb
import numpy as np

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from typing import List, Dict

from network import footandball
from data.cvat_dataloaders import make_data_loader
from network.ssd_loss import SSDLoss
from config import BaseConfig, BohsLaptopConfig, AFLLaptopConfig
from utils import save_model_weights

WANDB_MODE = 'offline'
WANDB_API_KEY = '83230c40e1c562f3ef56bf082e31911eaaad4ed9'
os.environ['WANDB_API_KEY'] = WANDB_API_KEY
CHECKPOINT_ROOT_DIR = '/opt/ml/checkpoints'

# Ball-related loss and player-related loss are mean losses (loss per one positive example)
ALPHA_C_BALL: float = 5.

torch.manual_seed(42)


def wandb_setup(model, criterion) -> None:
    wandb.init(
        project="Local Dev",
        name="Testing",
        notes="Testing",
        mode=WANDB_MODE
    )
    wandb.config.update(config)
    wandb.watch(model, criterion, log='all', log_freq=10)


def train_model(
        model: torch.nn.Module,
        optimizer: torch.optim,
        scheduler: torch.optim.lr_scheduler,
        num_epochs: int,
        dataloaders: Dict[str, DataLoader],
        device: str,
        config: BaseConfig
) -> None:
    # Loss function
    criterion = SSDLoss(neg_pos_ratio=config.neg_pos_ratio)

    # Setup Weights and Biases
    wandb_setup(model, criterion)

    phases = ['train', 'val'] if 'val' in dataloaders else ['train']
    training_stats: Dict = {'train': [], 'val': []}

    print("Training...")
    print(f"Total num epochs: {num_epochs}")
    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for phase in phases:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            batch_stats = {}
            # Iterate over data (in batches)
            for count_batches, (images, boxes, labels) in enumerate(dataloaders[phase]):

                # Load groundtruth maps
                images = images.to(device)
                h, w = images.shape[-2], images.shape[-1]
                gt_maps = model.groundtruth_maps(boxes, labels, (h, w))
                gt_maps = [e.to(device) for e in gt_maps]

                with torch.set_grad_enabled(phase == 'train'):
                    predictions = model(images)
                    loss_c_ball = criterion(predictions, gt_maps)
                    loss = ALPHA_C_BALL * loss_c_ball

                    # backward + optimize only if in training phase
                    if phase == 'train':

                        for param in model.parameters():
                            param.grad = None
                        loss.backward()
                        optimizer.step()

                        # Statistics/ wandb logging
                        batch_stats.setdefault('training_loss', []).append(loss.item())

                        if count_batches % 20 == 0:
                            wandb.log({"epoch": epoch, "training_loss": loss.item(), "loss_ball_c": loss_c_ball.item()})

                    elif phase == 'val':
                        batch_stats.setdefault('validation_loss', []).append(loss.item())
                        if count_batches % 20 == 0:
                            wandb.log({"epoch": epoch, "val_loss": loss.item(), "val_loss_ball_c": loss_c_ball.item()})

                # Average stats per batch
                avg_batch_stats: dict = {e: np.mean(batch_stats[e]) for e in batch_stats}
                training_stats[phase].append(avg_batch_stats)

            if phase == 'train':
                print(f'phase: {phase} - epoch: {epoch} avg. training_loss: {avg_batch_stats["training_loss"]}')
            else:
                print(f'phase: {phase} - epoch: {epoch} avg. val_loss: {avg_batch_stats["validation_loss"]}\n')

        if config.whole_dataset:
            if epoch % config.save_every_n_epochs == 0:
                save_model_weights(model=model, epoch=epoch, config=config)
        elif config.save_weights_when_testing and epoch % config.save_every_n_epochs == 0:
            save_model_weights(model=model, epoch=epoch, config=config)

        # Scheduler step
        scheduler.step()
        save_model_weights(model=model, epoch=epoch, config=config)

    # Save final training weights
    save_model_weights(model=model, epoch=epoch, config=config)
    print("\nTraining complete.\n")


def train(config: BaseConfig):

    # Create folder for saving model weights
    if not os.path.exists(config.model_folder):
        os.mkdir(config.model_folder)
    assert os.path.exists(config.model_folder), f' Cannot create folder to save trained model: {config.model_folder}'


    # Load dataloaders and print dataset sizes
    if config.run_validation:
        dataloaders = make_data_loader(config, ["train", "val"], use_hardcoded_data_folders=True)
    else:
        dataloaders = make_data_loader(config, ["train"], use_hardcoded_data_folders=True)
    for phase in dataloaders.keys():
        print(f"{phase} dataset size: {len(dataloaders[phase].dataset)}")


    # Create model
    model = footandball.model_factory('fb1', 'train')
    model.print_summary(show_architecture=False)

    device = "cuda" if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    model = model.to(device)
    print("device is on Cuda: ", next(model.parameters()).is_cuda)

    # Load weights if finetuning is enabled
    # if config.finetuning:
    #     print(f"Fine-tuning... Loading weights from: {config.finetuning_weights_path}")
    #     model.load_state_dict(torch.load(config.finetuning_weights_path))

    if device == 'cuda':
        cudnn.enabled = True
        cudnn.benchmark = True

    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    # Learning rate is reduced by a factor of 10 at 75% of the total number of epochs
    scheduler_milestones: List[int] = [int(config.epochs * 0.75)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, scheduler_milestones, gamma=0.1)

    train_model(model, optimizer, scheduler, config.epochs, dataloaders, device, config)


if __name__ == '__main__':
    config = BohsLaptopConfig()
    config.pretty_print()
    train(config)
