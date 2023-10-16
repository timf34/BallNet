import cv2

from config import BaseConfig, AFLLaptopConfig
from data.augmentation import tensor2image
from data.cvat_dataloaders import make_data_loader




def visualize_data_loader() -> None:
    data_loader = make_data_loader(config=AFLLaptopConfig(batch_size=1, use_augmentations=True), modes=['val'], use_hardcoded_data_folders=False)

    keys = list(data_loader.keys())

    for key in keys:
        for count, (image, boxes, labels, image_path) in enumerate(data_loader[key]):
            print(boxes)
            print(image_path)
            print(image.shape)

            # Squeeze image from [1, 3, 1080, 1920] to [3, 1080, 1920]
            image = image.squeeze(0)
            image = tensor2image(image)
            cv2.imshow('image', image)
            cv2.waitKey(0)

            if count == 10:
                break


def main():
    visualize_data_loader()


if __name__ == '__main__':
    main()
