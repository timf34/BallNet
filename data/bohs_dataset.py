from typing import List, Union, Dict
from PIL import Image
import random
import torch
import os
import numpy as np

import data.augmentation as augmentation
from data.bohs_utils import read_bohs_ground_truth
from config import Config

BALL_BBOX_SIZE = 20
BALL_LABEL = 1


class BohsDataset(torch.utils.data.Dataset):
    """
    A class for loading the bohs dataset.
    """
    def __init__(self, dataset_path: str,
                 transform: Union[augmentation.TrainAugmentation, augmentation.NoAugmentation],
                 only_ball_frames: bool = False,
                 whole_dataset: bool = False,
                 dataset_size: int = 1,
                 image_extension: str = '.jpg',
                 image_name_length: int = 7,
                 randomize_small_batches: bool = True):
        """
        Initializes the dataset.
        :param image_folder_path: Path to 'bohs-preprocessed' folder
        :param only_ball_frames: Whether to only use ball frames.
        :param whole_dataset: Whether to use the whole dataset.
        :param dataset_size: The size of the dataset to use if not using whole_dataset.
        :param transform: The transform to apply to the dataset.
        """
        print("Whole dataset: ", whole_dataset)
        self.dataset_path = dataset_path
        self.only_ball_frames = only_ball_frames
        self.whole_dataset = whole_dataset
        self.dataset_size = dataset_size
        self.transform = transform
        self.cameras: List[str] = [
            # "1_4_22_60FPS_jetson3_30s_test_clip_from_1945_vid",
            # "jetson3_1_4_2022_time__19_45_01_4",
            # "time_19_00_09_date_06_11_2022__4",
            # "jetson3_1_4_2022_time__21_25_19_2_minutes",
            # "jetson3_1_4_2022_time__20_40_14_20",
            # "jetson3_1_4_2022_time__19_45_01_5"
            # "jetson1_date_01_04_2022_time__19_45_01_4",
            # "jetson1_date_01_04_2022_time__19_45_01_5",
            # "jetson1_date_01_04_2022_time__20_40_14_20",
            # "jetson1_date_01_04_2022_time__20_40_14_25",
            # "jetson1_date_01_04_2022_time__21_25_19_1",
            # "jetson1_date_17_06_2022_time__19_45_05_27",
            # "jetson1_date_24_02_2023_time__19_45_01_22",
            # "jetson1_date_24_02_2023_time__19_45_01_23",
            # "jetson1_date_24_02_2023_time__19_45_01_24",
            # "jetson1_date_24_02_2023_time__19_45_01_26",
            # "jetson1_date_24_02_2023_time__19_45_01_29",
            # "jetson1_date_24_02_2023_time__19_45_01_39",
            # "jetson1_date_24_02_2023_time__19_45_01_37",
            "jetson1_date_24_02_2023_time__19_45_01_43",
            "jetson1_date_24_02_2023_time__19_45_01_17",
        ]
        self.image_name_length = image_name_length  # Number of digits in the image name
        self.image_extension: str = image_extension
        self.gt_annotations: dict = {}
        self.image_list: list = []

        # The folder paths we will be using.
        self.image_folder_path = os.path.join(self.dataset_path, 'unpacked_jpg')
        self.annotations_folder_path = os.path.join(self.dataset_path, 'annotations')

        assert transform is not None, "Transform must be specified"

        # This is just copying what is here already... probably a cleaner way to do this.
        for camera_id in self.cameras:
            # Read ground truth data for the sequence
            self.gt_annotations[camera_id] = read_bohs_ground_truth(annotations_path=self.annotations_folder_path,
                                                                    xml_file_name=f'{camera_id}.xml')

            # Create a list with ids of all images with any annotation
            # TODO: also note that we are only using images that include the ball currently
            annotated_frames = list(set(self.gt_annotations[camera_id].ball_pos))

            # Note that this assumes we have just one camera I think...
            if not whole_dataset:
                if randomize_small_batches:
                    # So we don't have consecutive images which are almost identical
                    random.shuffle(annotated_frames)
                annotated_frames = annotated_frames[3:self.dataset_size+3]

            images_path = os.path.join(self.image_folder_path, camera_id)

            for e in annotated_frames:
                e = str(e)
                e = e.zfill(self.image_name_length)
                file_path = os.path.join(images_path, f'frame_{e}{self.image_extension}')
                if os.path.exists(file_path):
                    self.image_list.append((file_path, camera_id, e))
                else:
                    print("doesn't exist", file_path)
                    print("check whether its frame_000001.png or just 000001.png")

        self.n_images = len(self.image_list)
        print(f"Total number of Bohs Images: {self.n_images}")
        self.ball_images_ndx = set(self.get_elems_with_ball())
        self.no_ball_images_ndx = set([ndx for ndx in range(self.n_images) if ndx not in self.ball_images_ndx])
        print(f'BOHS: {format(len(self.ball_images_ndx))} frames with the ball')
        print(f'BOHS: {(len(self.no_ball_images_ndx))} frames without the ball')
        print(f'Using whole dataset (bool): {whole_dataset}')

        if not self.whole_dataset:
            # Create new folder and copy image paths to it. This is for testing the overfitted model
            self.create_new_folder('../txt_testing_files')

            print("Here are the paths to the images:")
            with open('../txt_testing_files/image_paths.txt', 'w') as f:
                for image in self.image_list:
                    print(image)
                    f.write(image[0] + '\n')

    def __len__(self):
        return self.n_images

    def __getitem__(self, ndx):
        # Returns transferred image as a normalized tensor
        image_path, camera_id, image_ndx = self.image_list[ndx]
        image = Image.open(image_path)
        boxes, labels = self.get_annotations(camera_id, image_ndx)
        image, boxes, labels = self.transform((image, boxes, labels))
        boxes = torch.tensor(boxes, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.int64)
        return image, boxes, labels

    def get_annotations(self, camera_id, image_ndx):
        # Prepare annotations as list of boxes (xmin, ymin, xmax, ymax) in pixel coordinates
        # and torch int64 tensor of corresponding labels
        boxes = []
        labels = []
        # TODO: I needed to change ndx to ints for this dataset. This is because the ndx's come like '00001' but
        #  they're indexed as 1
        ball_pos = self.gt_annotations[camera_id].ball_pos[int(image_ndx)]
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
        for ndx, (_, camera_id, image_ndx) in enumerate(self.image_list):
            ball_pos = self.gt_annotations[camera_id].ball_pos[int(image_ndx)]
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


def create_bohs_dataset(conf: Config,
                        dataset_path: str,
                        whole_dataset: bool,
                        only_ball_frames: bool = False,
                        dataset_size: int = 2,
                        image_extension: str = '.jpg',
                        cameras: List[int] = None,
                        image_name_length: int = 7,
                        use_augs: bool = False):

    if cameras is None:
        cameras = [1]

    if use_augs:
        transform = augmentation.TrainAugmentation(size=conf.train_image_size)
    else:
        transform = augmentation.NoAugmentation(size=conf.val_image_size)
        print("creating Bohs Dataset with **no** augmentations (besides normalization)")

    return BohsDataset(dataset_path=dataset_path,
                       only_ball_frames=only_ball_frames,
                       whole_dataset=whole_dataset,
                       dataset_size=dataset_size,
                       image_extension=image_extension,
                       image_name_length=image_name_length,
                       transform=transform)
