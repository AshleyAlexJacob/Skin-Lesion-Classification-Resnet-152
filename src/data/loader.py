"""Data loading and dataset preparation for the skin lesion classification pipeline."""

from __future__ import annotations

import pathlib

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_data_loaders(
    train_dir: str | pathlib.Path,
    val_dir: str | pathlib.Path,
    batch_size: int = 32,
    img_size: tuple[int, int] = (224, 224),
    num_workers: int = 4,
    subset_fraction: float = 1.0,
) -> tuple[DataLoader, DataLoader, list[str]]:
    """Create PyTorch DataLoaders for training and validation from image directories.

    Expects the data directory to be organized with one subfolder per class,
    compatible with torchvision.datasets.ImageFolder.

    Args:
        train_dir: Path to the directory containing the training image classes.
        val_dir: Path to the directory containing the validation image classes.
        batch_size: Number of images per batch.
        img_size: Target size for resizing images (height, width).
        num_workers: Number of subprocesses to use for data loading.
        subset_fraction: Fraction of dataset to use (0.0 to 1.0) for fast testing.

    Returns:
        A tuple containing:
            - dataloader_train: DataLoader for the training set.
            - dataloader_val: DataLoader for the validation set.
            - class_names: List of class names inferred from directory structure.

    Raises:
        ValueError: If the specified data directories do not exist or are empty.
    """
    train_path = pathlib.Path(train_dir)
    val_path = pathlib.Path(val_dir)
    
    for path in [train_path, val_path]:
        if not path.exists() or not path.is_dir():
            raise ValueError(
                f"Data directory '{path}' does not exist or is not a directory."
            )

    # Define standard ImageNet-like transforms for ResNet-152
    data_transforms = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Use the built-in ImageFolder dataset
    train_dataset = datasets.ImageFolder(root=str(train_path), transform=data_transforms)
    val_dataset = datasets.ImageFolder(root=str(val_path), transform=data_transforms)
    
    class_names = train_dataset.classes

    if len(train_dataset) == 0:
        raise ValueError(f"No images found in training data directory '{train_path}'.")
    if len(val_dataset) == 0:
        raise ValueError(f"No images found in validation data directory '{val_path}'.")

    if subset_fraction < 1.0:
        train_subset_size = max(1, int(len(train_dataset) * subset_fraction))
        val_subset_size = max(1, int(len(val_dataset) * subset_fraction))

        # Take the first N elements
        train_dataset = torch.utils.data.Subset(train_dataset, range(train_subset_size))
        val_dataset = torch.utils.data.Subset(val_dataset, range(val_subset_size))

    # Create DataLoaders
    dataloader_train = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    dataloader_val = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return dataloader_train, dataloader_val, class_names


# if __name__ == "__main__":
#     dataloader_train, dataloader_test, class_names = get_data_loaders(
#         data_dir="data/processed"
#     )
#     print(class_names)
#     print(len(dataloader_train))
#     print(len(dataloader_test))
