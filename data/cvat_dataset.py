from typing import List, Union, Dict
from PIL import Image
import torch
import os
import numpy as np

import data.augmentation as augmentation
from config import BaseConfig
from data.cvat_utils import read_cvat_ground_truth, get_folders
from utils import create_directory

BALL_BBOX_SIZE: int = 20
BALL_LABEL: int = 1
DATASET_JUMP: int = 3  # Number of frames to jump into when creating a non-whole dataset
DEBUG: bool = True


class CVATBallDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            base_data_path: str,
            data_folder_paths: List[str],
            transform: Union[augmentation.TrainAugmentation, augmentation.NoAugmentation],
            only_ball_frames: bool = False,
            whole_dataset: bool = False,
            dataset_size_per_training_data_folder: int = 1,
            image_extension: str = '.png',
            image_name_length: int = 5,
    ):
        """
        Initializes the dataset.
        :param image_folder_path: Path to 'bohs-preprocessed' folder
        :param only_ball_frames: Whether to only use ball frames.
        :param whole_dataset: Whether to use the whole dataset.
        :param dataset_size_per_training_data_folder: The size of the dataset to use if not using whole_dataset.
        :param transform: The transform to apply to the dataset.
        """
        self.base_data_path = base_data_path
        self.only_ball_frames = only_ball_frames
        self.whole_dataset = whole_dataset
        self.dataset_size_per_training_data_folder = dataset_size_per_training_data_folder
        self.transform = transform
        self.data_folder_paths: List[str] = data_folder_paths
        self.image_name_length = image_name_length  # Number of digits in the image name
        self.image_extension: str = image_extension
        self.gt_annotations: dict = {}
        self.image_list: list = []

        # The folder paths we will be using.
        self.image_folder_path = os.path.join(self.base_data_path, f'unpacked_{image_extension.replace(".", "")}')
        self.annotations_folder_path = os.path.join(self.base_data_path, 'annotations')

        assert transform is not None, "Transform must be specified"
        assert self.data_folder_paths is not None, "No video paths passed to dataset initialization"

        self._load_annotations_and_images()
        self.n_images = len(self.image_list)
        self.ball_images_ndx = set(self.get_elems_with_ball())
        self.no_ball_images_ndx = {ndx for ndx in range(self.n_images) if ndx not in self.ball_images_ndx}

        if not self.whole_dataset:
            assert self.n_images == self.dataset_size_per_training_data_folder * len(self.data_folder_paths), \
                    f"Number of images in dataset ({self.n_images}) does not match expected number of images " \
                    f"({self.dataset_size_per_training_data_folder * len(self.data_folder_paths)})"

        print(f"Dataset base path: {self.base_data_path}")
        print(f"Total number of CVAT Images: {self.n_images}")
        print(f'CVAT: {format(len(self.ball_images_ndx))} frames with the ball')
        print(f'CVAT: {(len(self.no_ball_images_ndx))} frames without the ball')
        print(f'Whole dataset: {self.whole_dataset}')

        if DEBUG:
            self._debug_create_image_path_txt_files()

    def __len__(self):
        return self.n_images

    def __getitem__(self, ndx):
        # Returns transferred image as a normalized tensor
        image_path, data_folder, image_ndx = self.image_list[ndx]
        image = Image.open(image_path)
        boxes, labels = self.get_annotations(data_folder, image_ndx)
        image, boxes, labels = self.transform((image, boxes, labels))
        boxes = torch.tensor(boxes, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.int64)
        return image, boxes, labels

    def _debug_create_image_path_txt_files(self) -> None:
        """Function to create txt files with image paths for debugging"""
        if not self.whole_dataset:
            # Create new folder and copy image paths to it. This is for testing the overfitted model
            create_directory('./txt_testing_files')

            print("Here are the paths to the images:")
            with open('./txt_testing_files/image_paths.txt', 'w+') as f:
                for image in self.image_list:
                    print(image)
                    if 'train' in image[0] or 'bohs' in image[0]:
                        f.write(image[0] + '\n')


    def _load_annotations_and_images(self) -> None:
        for data_folder in self.data_folder_paths:
            # Read ground truth data for the sequence
            self.gt_annotations[data_folder] = read_cvat_ground_truth(
                annotations_path=self.annotations_folder_path,
                xml_file_name=f'{data_folder}.xml'
            )
            annotated_frames = self._get_annotated_frames(data_folder)
            images_path = os.path.join(self.image_folder_path, data_folder)
            self._populate_image_list(images_path, annotated_frames, data_folder)


    def _get_annotated_frames(self, data_folder: str) -> List[str]:
        """
            Note: we're only including ball images currently.
            Create a list with ids of all images with any annotation
        """
        annotated_frames = list(set(self.gt_annotations[data_folder].ball_pos))

        # Slice the dataset if not using the whole dataset: This is done per data folder.
        if not self.whole_dataset and len(annotated_frames) != self.dataset_size_per_training_data_folder:
            annotated_frames = annotated_frames[DATASET_JUMP:self.dataset_size_per_training_data_folder + DATASET_JUMP]

        return annotated_frames

    def _populate_image_list(self, images_path: str, annotated_frames: List[str], data_folder: str) -> None:
        for frame_number in annotated_frames:
            padded_frame_number = str(frame_number).zfill(self.image_name_length)
            file_path = os.path.join(images_path, f'frame_{padded_frame_number}{self.image_extension}')
            self._append_image_if_exists(file_path, data_folder, padded_frame_number)

    def _append_image_if_exists(self, file_path: str, data_folder: str, frame_number: str) -> None:
        if os.path.exists(file_path):
            self.image_list.append((file_path, data_folder, frame_number))
        else:
            print(f"Doesn't exist: {file_path}")

    def get_annotations(self, data_folder, image_ndx):
        # Prepare annotations as list of boxes (xmin, ymin, xmax, ymax) in pixel coordinates
        # and torch int64 tensor of corresponding labels
        boxes = []
        labels = []
        ball_pos = self.gt_annotations[data_folder].ball_pos[int(image_ndx)]
        if ball_pos != [[]]:
            for (x, y) in ball_pos:
                x1 = x - BALL_BBOX_SIZE // 2
                x2 = x1 + BALL_BBOX_SIZE
                y1 = y - BALL_BBOX_SIZE // 2
                y2 = y1 + BALL_BBOX_SIZE
                boxes.append((x1, y1, x2, y2))
                labels.append(BALL_LABEL)

        return np.array(boxes, dtype=float), np.array(labels, dtype=np.int64)

    def get_elems_with_ball(self) -> List:
        # Get indexes of images with ball ground truth
        ball_images_ndx = []
        for ndx, (_, data_folder, image_ndx) in enumerate(self.image_list):
            ball_pos = self.gt_annotations[data_folder].ball_pos[int(image_ndx)]
            if len(ball_pos) > 0 and ball_pos != [[]]: # With the collate function, empty lists are converted to [[]]
                ball_images_ndx.append(ndx)

        return ball_images_ndx

def get_data_folders_for_mode(training_mode: str, base_path: str) -> List[str]:
    """Returns a list of data folders for the given training mode"""
    assert training_mode in {
        'train',
        'val',
        'test',
    }, f"training_mode must be one of ['train', 'val', 'test'], but was {training_mode}"

    if 'bohs' in base_path:
        return get_folders(os.path.join(base_path, 'unpacked_jpg'))
    if training_mode == 'train':
        return get_folders(os.path.join(base_path, 'train/unpacked_png'))
    elif training_mode == 'val':
        return get_folders(os.path.join(base_path, 'val/unpacked_png'))
    elif training_mode == 'test':
        return get_folders(os.path.join(base_path, 'test/unpacked_png'))

def create_dataset_from_config(
        conf: BaseConfig,
        training_mode: str,
        use_hardcoded_data_folders: bool
) -> CVATBallDataset:

    assert training_mode in {
        'train',
        'val',
        'test',
    }, f"training_mode must be one of ['train', 'val', 'test'], but was {training_mode}"

    base_data_path: str = conf.base_data_path
    whole_dataset: bool = conf.whole_dataset
    only_ball_frames: bool = conf.only_ball_frames
    dataset_size_per_training_data_folder: int = conf.dataset_size_per_training_data_folder
    image_name_length: int = conf.image_name_length
    image_extension: str = conf.image_extension
    use_augs = conf.use_augmentations

    if use_hardcoded_data_folders:
        data_folder_paths: List[str] = conf.data_folder_paths[training_mode] if "bohs" not in base_data_path else conf.data_folder_paths
        assert data_folder_paths is not None, "data_folder_paths must be set if use_hardcoded_data_folders is True"
    else:
        data_folder_paths: List[str] = get_data_folders_for_mode(training_mode, base_data_path)
        assert data_folder_paths is not None, "data_folder_paths is None, ensure get_data_folders... is working"


    if 'bohs' not in base_data_path:
        base_data_path = os.path.join(base_data_path, training_mode)  # Bohs doesn't have train/val/test folders

    if use_augs:
        transform = augmentation.TrainAugmentation(size=conf.train_image_size)
    else:
        transform = augmentation.NoAugmentation(size=conf.val_image_size)
        print("creating CVAT Dataset with **no** augmentations (besides normalization)")

    return create_dataset(
        base_data_path=base_data_path,
        data_folder_paths=data_folder_paths,
        whole_dataset=whole_dataset,
        transform=transform,
        only_ball_frames=only_ball_frames,
        dataset_size_per_training_data_folder=dataset_size_per_training_data_folder,
        image_extension=image_extension,
        image_name_length=image_name_length,
    )


def create_dataset(
            base_data_path: str,
            data_folder_paths: List[str],
            whole_dataset: bool,
            transform,
            only_ball_frames: bool = False,
            dataset_size_per_training_data_folder: int = 2,
            image_extension: str = '.jpg',
            image_name_length: int = 7,
) -> CVATBallDataset:
    return CVATBallDataset(
        base_data_path=base_data_path,
        data_folder_paths=data_folder_paths,
        only_ball_frames=only_ball_frames,
        whole_dataset=whole_dataset,
        dataset_size_per_training_data_folder=dataset_size_per_training_data_folder,
        image_extension=image_extension,
        image_name_length=image_name_length,
        transform=transform
    )


def main():
    pass


if __name__ == '__main__':
    main()
