from dataclasses import dataclass
from typing import Tuple, List
import os
import time

BALL_LABEL = 1


def get_datetime():
    return time.strftime("%d_%m_%Y__%H%M")


@dataclass
class Config:
    """
    Configuration class for the program.
    """
    aws: bool = False
    aws_testing: bool = False

    # Training parameters
    only_ball_frames: bool = True  # TODO: Note: I think its baked in to just use ball frames.
    ball_threshold: float = 0.7
    device: str = 'cuda:0'
    # TODO: I really need some sort of different script maybe to read from the config file and then set the attributes
    #  for the config class for the different training runs I'm doing. I could have a fineuning script... an AWS script...
    #  an AWS testing script... a local testing script... a local training script... a local fine tuning script... etc.
    #  These could just be txt files, or whatever really.
    lr: float = 5e-5  # 1e-4 is the default learning rate for Adam; change to 1e-5 when fine-tuning

    run_validation: bool = True  # Whether to use the validation dataloaders or not
    use_amp: bool = True
    device_type: bool = "cuda"  # "cuda" or "cpu" (for AMP, and note it can't be "cuda:0" or "cuda:1" etc.)

    finetuning: bool = True

    # Model constants
    neg_pos_ratio: int = 3
    use_augmentation: bool = True  # This is vital for good performance I believe.
    ball_delta: int = 3  # TODO: this isn't being used in footandball yet.
    image_size: Tuple[int, int] = (720, 1280)
    image_extension: str = '.jpg'  # TODO: note that this isn't being used yet either
    model: str = 'fb1'

    train_image_size = (720, 1280)
    val_image_size = (1080, 1920    )
    train_size = 0.9
    val_size = 0.1

    out_video: str = 'test.avi'
    save_pickle_training_stats: bool = False

    # Debugging params
    save_epochs_every_n_epochs: int = 5
    save_heatmaps: bool = False
    number_of_heatmaps_to_save: int = 1
    save_weights_when_testing: bool = False


    # Augmentation constants
    ball_label: int = 1
    ball_bbox_size: int = 20

    def __post_init__(self):
        self.model_name = f'model_{get_datetime()}'

        if self.aws:
            # Paths
            try:
                # Note: this will be the 'bohs-preprocessed' folder in S3!
                self.bohs_path = os.environ['SM_CHANNEL_TRAINING']
            except KeyError as e:
                raise KeyError('The environment variable SM_CHANNEL_TRAINING is not set - ensure you run the file ' 
                               '`train_aws_estimator` and not just `train_detector.py`! Otherwise we\'re not using' 
                               'AWS and our environment variables won\'t be set.') from e

            self.model_folder = os.environ['SM_MODEL_DIR']
            self.checkpoints_folder = '/opt/ml/checkpoints'  # This can be access during training on AWS (opt/ml/model can't be)

            # TODO: ensure to add this to the 'bohs-preprocessed' folder in S3!
            self.finetuning_weights_file_name: str = "model_06_03_2023__0757_35.pth"
            self.finetuning_weights_path: str = os.path.join(self.bohs_path, self.finetuning_weights_file_name)  # "s3://bohs-preprocessed/model_22_11_2022__0202_90.pth"

            # Training parameters
            self.num_workers: int = 8
            self.batch_size: int = 32
            self.epochs: int = 20
            self.whole_dataset: bool = True
            self.dataset_size: int = -1  # This value should not be used, just a placeholder

        elif self.aws_testing:
            # This is a more lightweight (data-wise) version of the above, for testing on AWS.
            try:
                self.bohs_path = os.environ['SM_CHANNEL_TRAINING']
            except KeyError as e:
                raise KeyError('The environment variable SM_CHANNEL_TRAINING is not set - ensure you run the file '
                               ' `train_aws_estimator` and not just `train_detector.py`! Otherwise we\'re not using'
                               ' AWS and our environment variables won\'t be set.') from e

            self.model_folder = os.environ['SM_MODEL_DIR']
            self.checkpoints_folder = '/opt/ml/checkpoints'

            # Training parameters
            self.num_workers: int = 0
            self.batch_size: int = 1
            self.epochs: int = 10
            self.whole_dataset: bool = False
            self.dataset_size: int = 2

        else:
            # Paths
            self.bohs_path = r'C:\Users\timf3\OneDrive - Trinity College Dublin\Documents\Documents\datasets\Datasets\Bohs\bohs-preprocessed'
            self.weights: str = r'C:\Users\timf3\PycharmProjects\BallNet2.0\models\model_20210221_2206_final.pth'
            self.model_folder = 'models'

            self.finetuning_weights_path: str = r"C:\Users\timf3\PycharmProjects\BohsNet\models\model_22_11_2022__0202_90.pth"

            # Training parameters
            self.num_workers: int = 0
            self.batch_size: int = 1
            self.epochs: int = 10
            self.save_weights: bool = True

            self.whole_dataset: bool = False
            if self.whole_dataset:
                print("WARNING: You are training on the whole dataset!!! Don't expect fast batches!")

            self.dataset_size = 2  # Size of dataset if whole_dataset is False, not it gets n from each folder!

    def print(self):
        """
        Prints the config class and all its attributes to the console, including those that are initialized in
        the __post_init__ method.
        """
        for i in self.__dict__:
            print(f'{i}: {self.__dict__[i]}')

    def __str__(self):
        """
        Returns a string representation of the config class and all its attributes.
        """
        return str(self.__dict__)


if __name__ == '__main__':
    config = Config()
    print(config.model_name)
    print(config)
    config.print()
