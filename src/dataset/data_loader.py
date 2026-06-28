import torch
from torchvision import transforms
from dataset.utils import Dataset
from torch.utils.data import DataLoader, random_split

if __name__ == "__main__":

    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    data = Dataset(
        root="./data",
        train_transform=train_transform,
        test_transform=test_transform,
        val_transform=val_transform
    )

    train, test = data.dataset()

    print(train)
    print(test)
