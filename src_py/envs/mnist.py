import torch
import numpy as np
from torchvision import datasets, transforms


class MNIST:
    def __init__(self, seed=0, use_cuda=False):
        batch_size = 1
        torch.manual_seed(seed)
        device = torch.device("cuda" if use_cuda else "cpu")

        train_kwargs = {"batch_size": batch_size, "shuffle": True}
        test_kwargs = {"batch_size": batch_size, "shuffle": True}

        if use_cuda:
            cuda_kwargs = {"num_workers": 1, "pin_memory": True}
            train_kwargs.update(cuda_kwargs)
            test_kwargs.update(cuda_kwargs)

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        train_set = datasets.MNIST(
            "../constructive_data", train=True, download=True, transform=transform
        )
        test_set = datasets.MNIST(
            "../constructive_data", train=False, transform=transform
        )
        self.train_loader = torch.utils.data.DataLoader(train_set, **train_kwargs)
        self.test_loader = torch.utils.data.DataLoader(test_set, **test_kwargs)

    def sequential_iterator(self, split="train"):
        """
        Iterates over the split by returning sequences of length
        28 at a time. The labels are None except at the last sequence of
        the image.
        """
        # TODO not cuda friendly
        dataloader = self.train_loader if split == "train" else self.test_loader
        for idx_image, one_image in enumerate(dataloader):
            label = None
            for idx_sequence, sequence in enumerate(one_image[0][0][0].numpy()):
                if idx_sequence == 27:
                    label = one_image[1].item()
                yield idx_image, sequence, label
