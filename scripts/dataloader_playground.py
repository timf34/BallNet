from data.data_reader import make_train_val_dataloaders
from evaluate_model import EvaluationArgs


def get_dataloaders():
    return make_train_val_dataloaders(EvaluationArgs())


def val_dataloader_playground(dataloader) -> None:
    """
        Iterate through the eval dataloader, and write the output to a .txt file for inspection.
    """
    with open("blank.txt", "w") as f:
        for batch in dataloader["val"]:
            f.write(str(batch))
            f.write("\n\n")


def main():
    dataloader = get_dataloaders()
    val_dataloader_playground(dataloader)


if __name__ == "__main__":
    main()
