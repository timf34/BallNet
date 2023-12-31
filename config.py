from dataclasses import dataclass
from typing import Tuple, List, Dict, Union
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
    lr: float = 3e-4

    # Training params
    run_validation: bool = True
    ball_threshold: float = 0.7
    use_amp: bool = True
    finetuning: bool = False
    neg_pos_ratio: int = 3
    use_augmentations: bool = True
    save_pickle_training_stats: bool = False
    batch_size: int = 3
    num_workers: int = 0
    epochs: int = None # Set in child class

    # Data params
    only_ball_frames: bool = True
    # TODO: I might want to consider downsizing this to (720, 1280) soon (but note this will block at a lot of ball images)
    train_image_size: Tuple[int, int] = (1080, 1920)
    val_image_size: Tuple[int, int] = (1080, 1920)
    train_size: float = 0.9
    val_size: float = 0.1
    ball_bbox_size: int = 20
    image_extension: str = '.png'
    image_name_length: int = 7
    dataset_size_per_training_data_folder: int = None

    # Data constants
    model_folder: str = 'models'
    weights: str = r'C:\Users\timf3\PycharmProjects\BallNet2.0\models\model_20210221_2206_final.pth'
    base_data_path: str = None  # Set in child class
    whole_dataset: bool = None  # Set in child class
    data_folder_paths: Union[List[str], Dict[str, List[str]], None] = None  # We use a Dict for AFLConfig and a List for BohsConfig

    # Debugging params
    save_every_n_epochs: int = 5
    save_weights_when_testing: bool = False

    # Mist
    aws: bool = False
    checkpoints_folder: str = None

    def pretty_print(self) -> None:
        """
        Prints each attribute of the class and its value on a new line
        """
        for i in self.__dict__:
            print(f'{i}: {self.__dict__[i]}')

    def __post_init__(self):
        if self.whole_dataset is None:
            raise ValueError("`whole_dataset` must be set.")
        # if self.dataset_size_per_training_data_folder is None:
        #     raise ValueError("`dataset_size_per_training_data_folder` must be set.")
        if self.base_data_path is None:
            raise ValueError("`base_data_path` must be set.")
        if self.epochs is None:
            raise ValueError("`epochs` must be set.")
        if self.aws is None:
            raise ValueError("`aws` must be set.")
        # if self.checkpoints_folder is None and self.aws is True:
        #     raise ValueError("`checkpoints_folder` must be set if using AWS.")


@dataclass
class LaptopConfig(BaseConfig):
    """Config for local development on laptop"""
    # Training params
    epochs: int = 100
    save_weights: bool = True

    # Data params
    whole_dataset: bool = True
    dataset_size_per_training_data_folder: int = 3
    use_augmentations: bool = True
    batch_size: int = 1  # Max batch size for laptop GPU

    # Misc
    save_weights_when_testing: bool = True
    device: str = 'cuda:0'  # Local nvidia GPU locally


@dataclass
class AFLLaptopConfig(LaptopConfig):
    image_extension: str = '.png'
    base_data_path: str = r'C:\Users\timf3\PycharmProjects\AFL-Data\marvel\afl-preprocessed'

    run_validation: bool = True

    # Misc
    save_weights_when_testing: bool = True

    def __post_init__(self):
        self.data_folder_paths: Dict[str, List[str]] = {
            "train": [
                # "marvel_1_time_04_09_04_date_20_08_2023_0",
                "marvel_1_time_04_09_04_date_20_08_2023_4",
            ],
            "val": [
                # "marvel_1_time_04_09_04_date_20_08_2023_2",
                "marvel_3_time_04_09_06_date_20_08_2023_4",
            ]
        }


@dataclass
class BohsLaptopConfig(LaptopConfig):
    image_extension: str = '.jpg'
    base_data_path: str = r'C:\Users\timf3\OneDrive - Trinity College Dublin\Documents\Documents\datasets\Datasets\Bohs\bohs-preprocessed'

    use_augmentations = False

    def __post_init__(self):
        self.data_folder_paths: List[str] = [
                "jetson1_date_24_02_2023_time__19_45_01_43",
                # "jetson1_date_24_02_2023_time__19_45_01_17",
        ]  # Bohs data, just for testing on laptop


@dataclass
class AWSBaseConfig(BaseConfig):
    """Base Config for training on AWS SageMaker"""
    aws: bool = True
    aws_testing: bool = False  # For test runs, not full training runs

    # Paths
    checkpoints_folder = '/opt/ml/checkpoints'  # This can be access during training on AWS (opt/ml/model can't be)

    def __post_init__(self):

        try:
            self.model_folder = os.environ['SM_MODEL_DIR']
        except KeyError as e:
            raise KeyError('The environment variable SM_MODEL_DIR is not set - ensure you run the file '
                       '`train_aws_estimator` and not just `train.py`! Otherwise we\'re not using'
                       'AWS and our environment variables won\'t be set.') from e
        try:
            self.bohs_path = os.environ['SM_CHANNEL_TRAINING']  # The `bohs-preprocessed` folder in S3
        except KeyError as e:
            raise KeyError('The environment variable SM_CHANNEL_TRAINING is not set - ensure you run the file '
                       '`train_aws_estimator` and not just `train.py`! Otherwise we\'re not using'
                       'AWS and our environment variables won\'t be set.') from e


@dataclass
class AWSTestConfig(AWSBaseConfig):
    """Config for testing on AWS"""
    base_data_path = "s3://dublin-afl-preprocessed/"
    aws_testing = True
    num_workers: int = 0
    epochs: int = 2
    whole_dataset: bool = False
    dataset_size_per_training_data_folder: int = 2


@dataclass
class AWSTrainConfig(AWSBaseConfig):
    """Full training run on AWS"""
    num_workers: int = 8
    batch_size: int = 32
    epochs: int = 20
    whole_dataset: bool = True
    dataset_size_per_training_data_folder: int = -1  # Just a placeholder


@dataclass
class AWSSagemakerNotebook(BaseConfig):

    aws: bool = True
    aws_testing: bool = False  # For test runs, not full training runs

    model_folder: str = "weights"
    checkpoints_folder = "weights"
    base_data_path: str = "dublin-afl-preprocessed"

    num_works: int = 8
    batch_size: int = 32
    epochs: int = 100
    whole_dataset: bool = True
    dataset_size_per_training_data_folder: int = -11  # Just a placeholder


@dataclass
class AWSFineTuningConfig(AWSBaseConfig):

    finetuning_weights_filename: str = "model_06_03_2023__0757_35.pth"
    # finetuneing_weights_path: str = os.path.join(bohs_path, finetuning_weights_file_name)  # "s3://bohs-preprocessed/model_22_11_2022__0202_90.pth"


if __name__ == '__main__':
    x = BohsLaptopConfig()
    x.pretty_print()
