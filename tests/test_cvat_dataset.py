import numpy as np

from config import BohsLaptopConfig
from data.cvat_dataset import CVATBallDataset, create_dataset_from_config
from data.augmentation import NoAugmentation


config = BohsLaptopConfig()

def test_initialization():
    # Scenario 1: Test that the class can be initialized
    dataset = create_dataset_from_config(config)
    assert dataset is not None, "Failed to initialize CVATBallDataset"

    # Scenario 2: Transform should not be None
    # try:
    #     CVATBallDataset(dataset_path=dataset_path, transform=None)
    # except AssertionError as e:
    #     assert str(e) == "Transform must be specified", "Improper error message for None transform"


def test_length():
    dataset_path = '/path/to/dataset'
    transform = NoAugmentation(size=300)
    dataset = CVATBallDataset(dataset_path=dataset_path, transform=transform)

    # Scenario 1: Check if __len__ returns the correct length
    assert len(dataset) == dataset.n_images, "Dataset length mismatch"


def test_get_item():
    dataset_path = '/path/to/dataset'
    transform = NoAugmentation(size=300)
    dataset = CVATBallDataset(dataset_path=dataset_path, transform=transform)

    # Scenario 1: Test fetching an item
    image, boxes, labels = dataset[0]
    assert image is not None, "Image should not be None"
    assert boxes is not None, "Boxes should not be None"
    assert labels is not None, "Labels should not be None"


def test_get_annotations():
    dataset_path = '/path/to/dataset'
    transform = NoAugmentation(size=300)
    dataset = CVATBallDataset(dataset_path=dataset_path, transform=transform)

    # Scenario 1: Test getting annotations
    boxes, labels = dataset.get_annotations('camera_id', 'image_ndx')
    assert type(boxes) == np.ndarray, "Boxes should be numpy array"
    assert type(labels) == np.ndarray, "Labels should be numpy array"


def test_get_elems_with_ball():
    dataset_path = '/path/to/dataset'
    transform = NoAugmentation(size=300)
    dataset = CVATBallDataset(dataset_path=dataset_path, transform=transform)

    # Scenario 1: Test fetching elements with ball
    elems_with_ball = dataset.get_elems_with_ball()
    assert type(elems_with_ball) == list, "Should return a list"


# Running tests
test_initialization()
# test_length()
# test_get_item()
# test_get_annotations()
# test_get_elems_with_ball()
