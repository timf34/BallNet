from typing import List, Union
from PIL import Image
import random
import torch
import os
import numpy as np

import data.augmentation as augmentation
from data.bohs_utils import read_bohs_ground_truth
from config import BaseConfig

BALL_BBOX_SIZE: int = 20
BALL_LABEL: int = 1


class CVATBallDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            base_data_path: str,
            transform: Union[augmentation.TrainAugmentation, augmentation.NoAugmentation],
            training_data_folders: List[str],
            only_ball_frames: bool = False,
            whole_dataset: bool = False,
            dataset_size: int = 1,
            image_extension: str = '.png',
            image_name_length: int = 5,
            randomize_small_batches: bool = True
    ):
        """
        Initializes the dataset.
        :param image_folder_path: Path to 'bohs-preprocessed' folder
        :param only_ball_frames: Whether to only use ball frames.
        :param whole_dataset: Whether to use the whole dataset.
        :param dataset_size: The size of the dataset to use if not using whole_dataset.
        :param transform: The transform to apply to the dataset.
        """
        self.base_data_path = base_data_path
        self.only_ball_frames = only_ball_frames
        self.whole_dataset = whole_dataset
        self.dataset_size = dataset_size
        self.transform = transform
        self.training_data_folders: List[str] = training_data_folders
        self.image_name_length = image_name_length  # Number of digits in the image name
        self.image_extension: str = image_extension
        self.gt_annotations: dict = {}
        self.image_list: list = []

        # TODO: see if we need this
        self.randomize_small_batches: bool = False

        # The folder paths we will be using.
        self.image_folder_path = os.path.join(self.base_data_path, 'unpacked_jpg')
        self.annotations_folder_path = os.path.join(self.base_data_path, 'annotations')

        assert transform is not None, "Transform must be specified"
        assert self.training_data_folders is not None, "No video paths passed to dataset initialization"

        self._load_annotations_and_image()
        self.n_images = len(self.image_list)
        self.ball_images_ndx = set(self.get_elems_with_ball())
        self.no_ball_images_ndx = set([ndx for ndx in range(self.n_images) if ndx not in self.ball_images_ndx])
        print(f"Total number of Bohs Images: {self.n_images}")
        print(f'BOHS: {format(len(self.ball_images_ndx))} frames with the ball')
        print(f'BOHS: {(len(self.no_ball_images_ndx))} frames without the ball')
        print(f'Using whole dataset (bool): {whole_dataset}')

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

    def _debug_create_image_path_txt_files(self):
        """Function to create txt files with image paths for debugging"""
        if not self.whole_dataset:
            # Create new folder and copy image paths to it. This is for testing the overfitted model
            self.create_new_folder('../txt_testing_files')

            print("Here are the paths to the images:")
            with open('../txt_testing_files/image_paths.txt', 'w') as f:
                for image in self.image_list:
                    print(image)
                    f.write(image[0] + '\n')
        elif os.path.exists('../txt_testing_files'):
            os.rmdir('../txt_testing_files')


    def _load_annotations_and_image(self):
        # TODO: refactor this function into smaller functions
        # This is just copying what is here already... probably a cleaner way to do this.
        for data_folder in self.training_data_folders:
            # Read ground truth data for the sequence
            self.gt_annotations[data_folder] = read_bohs_ground_truth(annotations_path=self.annotations_folder_path,
                                                                      xml_file_name=f'{data_folder}.xml')

            # Create a list with ids of all images with any annotation
            # TODO: also note that we are only using images that include the ball currently
            annotated_frames = list(set(self.gt_annotations[data_folder].ball_pos))

            # Note that this assumes we have just one camera I think...
            if not self.whole_dataset:
                if self.randomize_small_batches:
                    # So we don't have consecutive images which are almost identical
                    random.shuffle(annotated_frames)
                annotated_frames = annotated_frames[3:self.dataset_size + 3]  # TODO: what does this do?W

            images_path = os.path.join(self.image_folder_path, data_folder)

            for e in annotated_frames:
                e = str(e)
                e = e.zfill(self.image_name_length)
                file_path = os.path.join(images_path, f'frame_{e}{self.image_extension}')
                if os.path.exists(file_path):
                    self.image_list.append((file_path, data_folder, e))
                else:
                    print("doesn't exist", file_path)
                    print("check whether its frame_000001.png or just 000001.png")


    def get_annotations(self, data_folder, image_ndx):
        # Prepare annotations as list of boxes (xmin, ymin, xmax, ymax) in pixel coordinates
        # and torch int64 tensor of corresponding labels
        boxes = []
        labels = []
        # TODO: I needed to change ndx to ints for this dataset. This is because the ndx's come like '00001' but
        #  they're indexed as 1
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

    @staticmethod
    def create_new_folder(folder_name) -> None:
        """
            This function checks if a folder exists, and if not, creates it.
        """
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
            print(f"Folder {folder_name} was created")
        else:
            print(f"Folder {folder_name} already exists")


def create_dataset_from_config(conf: BaseConfig) -> CVATBallDataset:
    base_data_path: str = conf.base_data_path
    training_data_folders: List[str] = conf.train_data_folders
    whole_dataset: bool = conf.whole_dataset
    only_ball_frames: bool = conf.only_ball_frames
    dataset_size: int = conf.dataset_size
    image_name_length: int = conf.image_name_length
    image_extension: str = conf.image_extension
    use_augs = conf.use_augmentations

    if use_augs:
        transform = augmentation.TrainAugmentation(size=conf.train_image_size)
    else:
        transform = augmentation.NoAugmentation(size=conf.val_image_size)
        print("creating Bohs Dataset with **no** augmentations (besides normalization)")

    return create_dataset(
        base_data_path=base_data_path,
        training_data_folders=training_data_folders,
        whole_dataset=whole_dataset,
        transform=transform,
        only_ball_frames=only_ball_frames,
        dataset_size=dataset_size,
        image_extension=image_extension,
        image_name_length=image_name_length,
    )


def create_dataset(
            base_data_path: str,
            training_data_folders: List[str],
            whole_dataset: bool,
            transform,
            only_ball_frames: bool = False,
            dataset_size: int = 2,
            image_extension: str = '.jpg',
            image_name_length: int = 7,
) -> CVATBallDataset:
    return CVATBallDataset(
        base_data_path=base_data_path,
        training_data_folders=training_data_folders,
        only_ball_frames=only_ball_frames,
        whole_dataset=whole_dataset,
        dataset_size=dataset_size,
        image_extension=image_extension,
        image_name_length=image_name_length,
        transform=transform
    )
