import os
import torch

from config import BaseConfig


def create_directory(directory: str) -> None:
    """Ensure the directory exists. If not, create it."""
    if not os.path.exists(directory):
        os.mkdir(directory)


def save_weights_to_path(model, filepath: str) -> None:
    """Save model weights to the given filepath if they don't exist already."""
    if os.path.exists(filepath):
        print(f'WARNING: Model weights already exist at {filepath}... not overwriting')
        return
    print(f"Saving model weights to: {filepath}")
    torch.save(model.state_dict(), filepath)


def get_model_filepath(base_folder: str, model_name: str, epoch: int) -> str:
    """Get the file path for the model weights."""
    model_folder = os.path.join(base_folder, model_name)
    create_directory(model_folder)
    return os.path.join(model_folder, f'{model_name}_{epoch}.pth')


def save_model_weights(model, epoch: int, config: BaseConfig) -> None:
    # TODO: saving twice to AWS. Is this necessary? Almost definitely not.
    #  self.checkpoints_folder = '/opt/ml/checkpoints' and  model_folder = os.environ['SM_MODEL_DIR']
    """
    Saves once locally to main_folder and also again to checkpoints_folder if aws is enabled.
    """
    main_filepath = get_model_filepath(config.model_folder, config.model_name, epoch)
    save_weights_to_path(model, main_filepath)

    if config.aws:
        checkpoint_filepath = get_model_filepath(config.checkpoints_folder, config.model_name, epoch)
        save_weights_to_path(model, checkpoint_filepath)