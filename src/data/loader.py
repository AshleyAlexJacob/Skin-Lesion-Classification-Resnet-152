"""Data loading and dataset preparation for the skin lesion classification pipeline."""

from __future__ import annotations

import pathlib

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_data_loaders(
    data_dir: str | pathlib.Path,
    batch_size: int = 32,
    img_size: tuple[int, int] = (224, 224),
    num_workers: int = 4,
    seed: int = 42,
    subset_fraction: float = 1.0,
) -> tuple[DataLoader, DataLoader, list[str]]:
    """Create PyTorch DataLoaders for training and testing from an image directory.

    Expects the data directory to be organized with one subfolder per class,
    compatible with torchvision.datasets.ImageFolder.

    Args:
        data_dir: Path to the directory containing the image classes.
        batch_size: Number of images per batch.
        img_size: Target size for resizing images (height, width).
        num_workers: Number of subprocesses to use for data loading.
        seed: Random seed for reproducible 70/30 dataset splitting.
        subset_fraction: Fraction of dataset to use (0.0 to 1.0) for fast testing.

    Returns:
        A tuple containing:
            - dataloader_train: DataLoader for the training set (70%).
            - dataloader_test: DataLoader for the testing set (30%).
            - class_names: List of class names inferred from directory structure.

    Raises:
        ValueError: If the specified data directory does not exist or is empty.
    """
    data_path = pathlib.Path(data_dir)
    if not data_path.exists() or not data_path.is_dir():
        raise ValueError(
            f"Data directory '{data_path}' does not exist or is not a directory."
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
    full_dataset = datasets.ImageFolder(root=str(data_path), transform=data_transforms)
    class_names = full_dataset.classes

    if len(full_dataset) == 0:
        raise ValueError(f"No images found in data directory '{data_path}'.")

    # 70/30 Split
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    test_size = total_size - train_size

    # Establish reproducible split
    generator = torch.Generator().manual_seed(seed)
    train_dataset, test_dataset = random_split(
        full_dataset, [train_size, test_size], generator=generator
    )

    if subset_fraction < 1.0:
        train_subset_size = max(1, int(train_size * subset_fraction))
        test_subset_size = max(1, int(test_size * subset_fraction))

        # Take the first N elements from the randomized split
        train_dataset = torch.utils.data.Subset(train_dataset, range(train_subset_size))
        test_dataset = torch.utils.data.Subset(test_dataset, range(test_subset_size))

    # Create DataLoaders
    dataloader_train = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    dataloader_test = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return dataloader_train, dataloader_test, class_names


# if __name__ == "__main__":
#     dataloader_train, dataloader_test, class_names = get_data_loaders(
#         data_dir="data/processed"
#     )
#     print(class_names)
#     print(len(dataloader_train))
#     print(len(dataloader_test))
