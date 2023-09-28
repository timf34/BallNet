from dataclasses import dataclass, field
from typing import Tuple
import os
import time

BALL_LABEL: int = 1

def get_datetime():
    return time.strftime("%d_%m_%Y__%H%M")


@dataclass
class BaseConfig:
    """
    Base Config class
    """
    model_name = f'model_{get_datetime()}'

    # Model params
    device: str = 'cuda:0'
    device_type: bool = "cuda"  # Needed for a Torch call it seems
    lr: float = 1e-4

    # Training parms
    run_validation: bool = True
    ball_threshold: float = 0.7
    use_amp: bool = True
    finetuning: bool = False
    neg_pos_ratio: int = 3
    use_augmentation: bool = True
    save_pickle_training_stats: bool = False

    # Data params
    only_ball_frames: bool = True
    train_image_size: Tuple[int, int] = (720, 1280)
    val_image_size: Tuple[int, int] = (1080, 1920)
    train_size: float = 0.9
    val_size: float = 0.1
    ball_bbox_size: int = 20

    # Data constants
    model_folder: str = 'models'
    weights: str = r'C:\Users\timf3\PycharmProjects\BallNet2.0\models\model_20210221_2206_final.pth'
    data_path: str = r'C:\Users\timf3\OneDrive - Trinity College Dublin\Documents\Documents\datasets\Datasets\Bohs\bohs-preprocessed'

    # Debugging params
    save_every_n_epochs: int = 5
    save_weights_when_testing: bool = False

    def pretty_print(self) -> None:
        """
        Prints each attribute of the class and its value on a new line
        """
        for i in self.__dict__:
            print(f'{i}: {self.__dict__[i]}')


@dataclass
class LaptopConfig(BaseConfig):
    """Config for local development on laptop"""
    # Training params
    num_workers: int = 0
    batch_size: int = 1
    epochs: int = 10
    save_weights: bool = True

    # Data params
    whole_dataset: bool = False
    dataset_size: int = 2


@dataclass
class AWSBaseConfig(BaseConfig):
    """Base Config for training on AWS SageMaker"""
    aws_testing: bool = False  # For test runs, not full training runs

    # Paths
    checkpoints_folder = '/opt/ml/checkpoints'  # This can be access during training on AWS (opt/ml/model can't be)

    def __post_init__(self):

        try:
            self.model_folder = os.environ['SM_MODEL_DIR']
        except KeyError as e:
            raise KeyError('The environment variable SM_MODEL_DIR is not set - ensure you run the file '
                       '`train_aws_estimator` and not just `train_detector.py`! Otherwise we\'re not using'
                       'AWS and our environment variables won\'t be set.') from e
        try:
            self.bohs_path = os.environ['SM_CHANNEL_TRAINING']  # The `bohs-preprocessed` folder in S3
        except KeyError as e:
            raise KeyError('The environment variable SM_CHANNEL_TRAINING is not set - ensure you run the file '
                       '`train_aws_estimator` and not just `train_detector.py`! Otherwise we\'re not using'
                       'AWS and our environment variables won\'t be set.') from e


@dataclass
class AWSTestConfig(AWSBaseConfig):
    """Config for testing on AWS"""
    num_workers: int = 0
    batch_size: int = 1
    epochs: int = 10
    whole_dataset: bool = False
    dataset_size: int = 2


@dataclass
class AWSTrainConfig(AWSBaseConfig):
    """Full training run on AWS"""
    num_workers: int = 8
    batch_size: int = 32
    epochs: int = 20
    whole_dataset: bool = True
    dataset_size: int = -1  # Just a placeholder


@dataclass
class AWSFineTuningConfig(AWSBaseConfig):

    finetuning_weights_filename: str = "model_06_03_2023__0757_35.pth"
    # finetuneing_weights_path: str = os.path.join(bohs_path, finetuning_weights_file_name)  # "s3://bohs-preprocessed/model_22_11_2022__0202_90.pth"


if __name__ == '__main__':
    x = LaptopConfig()
    print(x)