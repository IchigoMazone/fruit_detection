from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


DEFAULT_DATASET_DIR = Path("data/fruit_quality_classification")
DEFAULT_IMAGE_SIZE = 224


def default_train_transform(image_size: int = DEFAULT_IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def default_eval_transform(image_size: int = DEFAULT_IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def get_fruit_quality_datasets(
    root: str | Path = DEFAULT_DATASET_DIR,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> tuple[datasets.ImageFolder, datasets.ImageFolder, datasets.ImageFolder]:
    root = Path(root)
    train_dataset = datasets.ImageFolder(root / "train", transform=default_train_transform(image_size))
    val_dataset = datasets.ImageFolder(root / "val", transform=default_eval_transform(image_size))
    test_dataset = datasets.ImageFolder(root / "test", transform=default_eval_transform(image_size))
    return train_dataset, val_dataset, test_dataset


def get_fruit_quality_loaders(
    root: str | Path = DEFAULT_DATASET_DIR,
    batch_size: int = 32,
    image_size: int = DEFAULT_IMAGE_SIZE,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset, val_dataset, test_dataset = get_fruit_quality_datasets(root=root, image_size=image_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
