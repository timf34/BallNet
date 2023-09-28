"""
    This file is just to save the image paths from datdaloader to a .txt file. Super quickly.
"""
import torch

from config import Config
from data.data_reader import make_train_val_dataloaders

torch.manual_seed(42)


class SaveDataloaderArgs(Config):
    """
        Note that we need to uncomment the lines in bohs_dataset.py and data_reader.py that return the image paths
    """
    def __init__(self):
        super().__init__()
        self.output_path: str = r"C:\Users\timf3\PycharmProjects\BohsNet\data\val_dataloader_image_paths\v1_26_11_22.txt"
        self.train: bool = True
        self.whole_dataset: bool = True

    def print(self):
        for i in self.__dict__:
            print(f"{i}: {self.__dict__[i]}")


def save_dataloader_image_paths() -> None:
    # Load the args
    args = SaveDataloaderArgs()
    args.print()

    # Load the dataloader
    dataloaders = make_train_val_dataloaders(args)
    val_dataloader = dataloaders["val"]

    with open(args.output_path, "w") as f:
        for images, boxes, labels, image_paths in val_dataloader:
            for image_path in image_paths:
                f.write(image_path + "\n")


def main():
    save_dataloader_image_paths()


if __name__ == "__main__":
    main()
