import torch


def test_set_grad_enabled_val_split():
    phases = ['train', 'val']

    for _ in range(10):
        for phase in phases:
            with torch.set_grad_enabled(phase == 'train'):
                print(phase)


if __name__ == '__main__':
    test_set_grad_enabled_val_split()