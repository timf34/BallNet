from tqdm import tqdm
import argparse
import pickle
import os
import time
import wandb
import numpy as np

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from network import footandball
from data.data_reader import make_dataloaders, make_train_val_dataloaders
from network.ssd_loss import SSDLoss
from config import Config
from misc.legacy.training_utils import debugging_ball_feature_maps

WANDB_MODE = 'online'

WANDB_API_KEY = '83230c40e1c562f3ef56bf082e31911eaaad4ed9'
os.environ['WANDB_API_KEY'] = WANDB_API_KEY
CHECKPOINT_ROOT_DIR = '/opt/ml/checkpoints'

torch.manual_seed(42)


def train_model(model, optimizer, scheduler, num_epochs, dataloaders, device, model_name, params):
    # Ball-related loss and player-related loss are mean losses (loss per one positive example)
    alpha_c_ball: float = 5.

    # Initialize scaler
    scaler = torch.cuda.amp.GradScaler(enabled=params.use_amp)

    print("device is on Cuda: ", next(model.parameters()).is_cuda)
    assert next(model.parameters()).is_cuda, "Model isn't on cuda"

    # Loss function
    criterion = SSDLoss(neg_pos_ratio=params.neg_pos_ratio)

    # I need to be much more diligent with the notes I use for WandB.
    # "BohsNet-AWS-FullTraining" - for full AWS training
    # "BohsNet-Testing" - for random testing
    wandb.init(project="BohsNet-MiddleCamera-Finetuning",
               name="Run-2",
               notes="Finetuning middle camera with more labelled data",
               mode="online")

    wandb.config.update(params)
    wandb.watch(model, criterion, log='all', log_freq=10)

    print("torch version: ", torch.__version__)

    is_validation_set = 'val' in dataloaders
    print("Validation: ", is_validation_set)
    phases = ['train', 'val'] if is_validation_set else ['train']

    # Training statistics
    training_stats = {'train': [], 'val': []}

    print('\nTraining...\n')

    # Iterate through epochs
    for epoch in tqdm(range(num_epochs)):
        # Each epoch has a training and validation phase
        for phase in phases:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            batch_stats = {'training_loss': [], 'validation_loss': [], 'loss_ball_c': []}
            count_batches = 0

            # Iterate over data (in batches)
            for images, boxes, labels in dataloaders[phase]:

                # Load groundtruth maps
                images = images.to(device)
                h, w = images.shape[-2], images.shape[-1]
                gt_maps = model.groundtruth_maps(boxes, labels, (h, w))
                gt_maps = [e.to(device) for e in gt_maps]

                with torch.set_grad_enabled(phase == 'train'):
                    with torch.autocast(device_type=params.device_type, dtype=torch.float16, enabled=params.use_amp):
                        predictions = model(images)
                        loss_c_ball = criterion(predictions, gt_maps)
                        loss = alpha_c_ball * loss_c_ball  # Not sure if this is best to put inside AMP

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        # Backpropagation
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                        for param in model.parameters():
                            param.grad = None

                        # Statistics/ wandb logging
                        batch_stats['training_loss'].append(loss.item())
                        batch_stats['loss_ball_c'].append(loss_c_ball.item())
                        if count_batches % 20 == 0:
                            wandb.log({"epoch": epoch, "training_loss": loss.item(), "loss_ball_c": loss_c_ball.item()})

                    elif phase == 'val':
                        batch_stats['validation_loss'].append(loss.item())
                        # batch_stats['loss_ball_c'].append(loss_c_ball.item())
                        if count_batches % 20 == 0:
                            wandb.log({"epoch": epoch, "val_loss": loss.item(), "val_loss_ball_c": loss_c_ball.item()})

                    # Be cautious of where we place this line with respect to logging!
                    count_batches += 1

                # Average stats per batch
                avg_batch_stats: dict = {e: np.mean(batch_stats[e]) for e in batch_stats}  # TODO: This doesn't seem to work properly for some reason... I should step through it with a smaller dataset.
                training_stats[phase].append(avg_batch_stats)

            print(f'phase: {phase} - epoch: {epoch} '
                  f'avg. training_loss: {avg_batch_stats["training_loss"]}, '
                  f'val_loss: {avg_batch_stats["validation_loss"]}, '
                  f'loss_ball_c: {avg_batch_stats["loss_ball_c"]}')

        if params.whole_dataset:
            if epoch % params.save_every_n_epochs == 0:
                save_model_weights(model=model, epoch=epoch, params=params)
        elif params.save_weights_when_testing and epoch % params.save_every_n_epochs == 0:
            save_model_weights(model=model, epoch=epoch, params=params)
        else:
            # Save the final weights
            if epoch % params.save_every_n_epochs == 0 and params.save_heatmaps is True:
                debugging_ball_feature_maps(predictions, epoch, params)
                save_model_weights(model=model, epoch=epoch, params=params)

        # Scheduler step
        scheduler.step()
        print('')
        save_model_weights(model=model, epoch=epoch, params=params)

    # Save final training weights
    save_model_weights(model=model, epoch=epoch, params=params)

    if params.save_pickle_training_stats:
        with open(f'training_stats_{model_name}.pickle', 'wb') as handle:
            pickle.dump(training_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("\nTraining complete.\n")
    return training_stats


def train(params: Config):
    if not os.path.exists(params.model_folder):
        os.mkdir(params.model_folder)

    assert os.path.exists(params.model_folder), f' Cannot create folder to save trained model: {params.model_folder}'

    if params.run_validation:
        dataloaders = make_train_val_dataloaders(params)
    else:
        # TODO: 21/11/22 note that this function has the sampler, but the above one does not!
        dataloaders = make_dataloaders(params)
    print(f"\nTraining set: Dataset size: {len(dataloaders['train'].dataset)}")
    if 'val' in dataloaders:
        print(f"Validation set: Dataset size: {len(dataloaders['val'].dataset)}\n")

    # Create model
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    print(device)
    model = footandball.model_factory(params.model, 'train')
    model.print_summary(show_architecture=False)

    if params.finetuning:
        print(f"Fine-tuning... Loading weights from: {params.finetuning_weights_path}")
        print("!!!!!!!!!!!!!! Reduce the learning rate by a factor of 10 when fine-tuning !!!!!!!!!!!!!!!")
        model.load_state_dict(torch.load(params.finetuning_weights_path))

    model = model.to(device)
    model.train()

    if device == 'cuda':
        cudnn.enabled = True
        cudnn.benchmark = True

    model_name = 'model_' + time.strftime("%d_%m_%Y_%H%M")
    print(f'Model name: {model_name}')

    optimizer = optim.Adam(model.parameters(), lr=params.lr)
    scheduler_milestones = [int(params.epochs * 0.50)]  # # finetuning: Reduce by 5 after 50% of training for finetuning; normal: Reduce learning rate by 10 after 75% of training
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, scheduler_milestones, gamma=0.5)
    train_model(model, optimizer, scheduler, params.epochs, dataloaders, device, model_name, params)


def save_model_weights(model, epoch: int, params: Config) -> None:
    """
        This function creates a folder in the params.model_folder directory with the data and time.
        Then it saves the model weights to this folder
    """
    # Going to save to both the model directory and the checkpoints directory!

    # Only save when on AWS.
    if params.aws is True:
        for folder in [params.checkpoints_folder, params.model_folder]:
            model_folder = os.path.join(folder, f'{params.model_name}')

            if not os.path.exists(model_folder):
                os.mkdir(model_folder)
            model_filepath = os.path.join(model_folder, f'{params.model_name}_{epoch}.pth')
            # Check if the model weights already exist
            if os.path.exists(model_filepath):
                print(f'WARNING: Model weights already exist at {model_filepath}... not overwriting')
                return
            else:
                print(f"Saving model weights to: {model_filepath}")
                torch.save(model.state_dict(), model_filepath)


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train BohsNet on the Bohs dataset!')

    # TODO: NOTE THAT THIS IS CURRENTLY SET UP FOR LOCAL DEV - SEE THE DEFAULTS! I NEEDED TO USE THE DEBUGGER
    parser.add_argument('--aws', default=True, type=lambda x: (str(x).lower() == 'true'), help="True or False")
    parser.add_argument('--aws_testing', default=False, type=lambda x: (str(x).lower() == 'true'), help='aws_testing')
    parser.add_argument('--run_validation', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='run_validation; True or False')
    parser.add_argument('--save_weights_when_testing', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='True or False')

    return parser


if __name__ == '__main__':
    print('Train BohsNet on the Bohs dataset! We are in train_detector.py')

    parser = init_argparse()
    args = parser.parse_args()
    print(args)

    # We only pass values that are not None, so that we can use the default values in the config dataclass
    config = Config(**{e: vars(args)[e] for e in vars(args) if vars(args)[e] is not None})
    config.print()

    train(config)
