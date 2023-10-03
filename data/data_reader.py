"""
This code builds the dataloaders for the ISSIA dataset

Some general TODOs for this file:
- Remove code for the spd dataset
"""
import random
import torch

from torch.utils.data import Sampler, DataLoader, ConcatDataset, Subset
from typing import List, Dict

from data.cvat_dataset import create_dataset_from_config, CVATBallDataset
from config import BaseConfig
from data.cvat_utils import get_train_val_datasets


def make_train_val_dataloaders(config: BaseConfig) -> dict:
    """
    This is not the cleanest implementation but we will clean it up later - doesn't matter for now

    This function creates the dataloaders for the training and validation sets.
    :param config: The configuration object.
    :return: A dictionary containing the training and validation dataloaders.
    """
    # TODO: note, right now, the validation dataset will have the same augemntations as the trianing dataset.
    #  Fix this soon. I would want to be able to specify NoAugs for val I would think.

    cvat_dataset = create_dataset_from_config(
        conf=config,
        )
    train_dataset, val_dataset = get_train_val_datasets(cvat_dataset, config=config)

    train_dataset = ConcatDataset([train_dataset])
    val_dataset = ConcatDataset([val_dataset])

    return {
        'train': DataLoader(
            train_dataset,
            # sampler=BalancedSampler(train_dataset, quick_fix_for_train_val_split=True),
            shuffle=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=True,
            collate_fn=my_collate
        ),
        'val': DataLoader(
            val_dataset,
            # sampler=BalancedSampler(val_dataset, quick_fix_for_train_val_split=True),
            shuffle=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=True,
            collate_fn=my_collate
        )
    }


def make_dataloaders(config: BaseConfig) -> Dict[str, DataLoader]:
    train_bohs_dataset = create_dataset_from_config(
        conf = config,
    )
    train_dataset = ConcatDataset([train_bohs_dataset])
    return {
        'train': DataLoader(
            train_dataset,
            # shuffle=True,  # Note: Cannot use shuffle and sampler together!
            sampler=BalancedSampler(train_dataset),
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=True,
            collate_fn=my_collate
        )
    }


def make_eval_dataloader(config: BaseConfig) -> DataLoader:
    """
    This function creates the dataloader for the evaluation script.
    """
    # TODO: ensure augs are not used for the evaluation script
    #  i.e. NoAugs
    config.use_augmentations = False
    dataset =  create_dataset_from_config(
        conf=config,
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=my_collate
    )


def my_collate(batch):
    images = torch.stack([e[0] for e in batch], dim=0)
    boxes = [e[1] for e in batch]
    labels = [e[2] for e in batch]
    # image_path = [e[3] for e in batch]
    # return images, boxes, labels, image_path
    return images, boxes, labels


def collate_fn(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    images, targets = zip(*batch)
    return torch.stack(images, 0), targets


# Note: I'll come back to this for refactoring later as we won't be using it right now, and its quite a mess.
class BalancedSampler(Sampler):
    # Sampler sampling the same number of frames with and without the ball
    # TODO: we can't us this right now as our dataset only contains the ball, but I should experiment trianing without
    # images of the ball at some point soon too.
    def __init__(self, data_source):
        super().__init__(data_source)
        self.data_source = data_source
        self.sample_ndx = []
        self.generate_samples()

    def generate_samples(self):
        # Sample generation function expects concatenation of 2 datasets: one is ISSIA CNR and the other is SPD
        # or only one ISSIA CNR dataset.
        assert len(self.data_source.datasets) <= 2
        issia_dataset_ndx = None
        spd_dataset_ndx = None
        # TODO: ok so this is another quick fix I will have to deal with later.
        if isinstance(self.data_source, Subset):
            for ndx, ds in enumerate(self.data_source.dataset.datasets):
                if isinstance(ds, CVATBallDataset) or isinstance(ds, Subset):
                    bohs_dataset_ndx = ndx
                if isinstance(ds, CVATBallDataset):
                    issia_dataset_ndx = ndx
                else:
                    spd_dataset_ndx = ndx
        else:
            for ndx, ds in enumerate(self.data_source.datasets):
                # TODO: added it here.
                if isinstance(ds, CVATBallDataset) or isinstance(ds, Subset):
                    bohs_dataset_ndx = ndx
                if isinstance(ds, CVATBallDataset):
                    issia_dataset_ndx = ndx
                else:
                    spd_dataset_ndx = ndx

        # Note that we will not be training ISSIA and Bohs together so we are making some breaking changes here that
        # will probably need to be reduced or cleaned up in the future.
        # Main change being that we are changing issia_dataset_ndx to dataset_ndx (which we can equal to
        # bohs_dataset_ndx) if we want to train on the bohs dataset instead
        if issia_dataset_ndx is not None:
            dataset_ndx = issia_dataset_ndx
        else:
            dataset_ndx = bohs_dataset_ndx

        issia_ds = self.data_source.datasets[dataset_ndx]
        # TODO: added it here (alhtough I didnt use the quick fix flag)
        if isinstance(issia_ds, Subset):
            n_ball_images = len(issia_ds.dataset.ball_images_ndx)
            # no_ball_images = 0.5 * ball_images
            n_no_ball_images = min(len(issia_ds.dataset.no_ball_images_ndx), int(0.5 * n_ball_images))
            issia_samples_ndx = list(issia_ds.dataset.ball_images_ndx) + random.sample(issia_ds.dataset.no_ball_images_ndx,
                                                                           n_no_ball_images)
        else:
            n_ball_images = len(issia_ds.ball_images_ndx)
            # no_ball_images = 0.5 * ball_images
            n_no_ball_images = min(len(issia_ds.no_ball_images_ndx), int(0.5 * n_ball_images))
            issia_samples_ndx = list(issia_ds.ball_images_ndx) + random.sample(issia_ds.no_ball_images_ndx,
                                                                               n_no_ball_images)
        if dataset_ndx > 0:
            # Add sizes of previous datasets to create cumulative indexes
            issia_samples_ndx = [e + self.data_source.cummulative_sizes[dataset_ndx - 1] for e in issia_samples_ndx]

        if spd_dataset_ndx is not None:
            spd_dataset = self.data_source.datasets[spd_dataset_ndx]
            n_spd_images = min(len(spd_dataset), int(0.5 * n_ball_images))
            spd_samples_ndx = random.sample(range(len(spd_dataset)), k=n_spd_images)
            if spd_dataset_ndx > 0:
                # Add sizes of previous datasets to create cummulative indexes
                spd_samples_ndx = [e + self.data_source.cummulative_sizes[spd_dataset_ndx - 1] for e in spd_samples_ndx]
        else:
            n_spd_images = 0
            spd_samples_ndx = []

        self.sample_ndx = issia_samples_ndx + spd_samples_ndx
        random.shuffle(self.sample_ndx)

    def __iter__(self):
        self.generate_samples()  # Re-generate samples every epoch
        for ndx in self.sample_ndx:
            yield ndx

    def __len(self):
        return len(self.sample_ndx)