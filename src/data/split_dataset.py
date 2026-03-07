"""Module to split dataset into train, validation, and test sets."""

from __future__ import annotations

import logging
import random
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def split_data(
    input_dir: str | Path,
    output_dir: str | Path,
    train_ratio: float = 0.63,
    val_ratio: float = 0.27,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> None:
    """Split images from class subdirectories into train, val, and test sets.

    Args:
        input_dir: Path to directory containing class subdirectories.
        output_dir: Path to output directory to save the train, val, test sets.
        train_ratio: Proportion of data for the training set.
        val_ratio: Proportion of data for the validation set.
        test_ratio: Proportion of data for the test set.
        seed: Random seed for shuffling.

    Raises:
        ValueError: If ratios do not sum to 1.0 or are negative, or if input_dir is invalid.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists() or not input_path.is_dir():
        raise ValueError(f"Input directory does not exist or is not a directory: {input_path}")

    # Set random seed for reproducibility
    random.seed(seed)

    splits = ["train", "val", "test"]
    class_dirs = [d for d in input_path.iterdir() if d.is_dir()]

    if not class_dirs:
        logger.warning(f"No class subdirectories found in {input_path}")
        return

    logger.info(f"Found {len(class_dirs)} classes to split. Splitting to {output_path}")

    # Prepare output directories
    for split in splits:
        for class_dir in class_dirs:
            (output_path / split / class_dir.name).mkdir(parents=True, exist_ok=True)

    total_moved = {"train": 0, "val": 0, "test": 0}

    for class_dir in class_dirs:
        images = [
            f for f in class_dir.iterdir() 
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        images.sort()  # Sort first for deterministic behaviour before shuffle
        random.shuffle(images)

        total_images = len(images)
        train_end = int(total_images * train_ratio)
        val_end = train_end + int(total_images * val_ratio)

        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]

        # Log class split counts
        logger.debug(
            f"Class {class_dir.name}: {len(train_images)} train, "
            f"{len(val_images)} val, {len(test_images)} test"
        )

        _copy_images(train_images, output_path / "train" / class_dir.name)
        _copy_images(val_images, output_path / "val" / class_dir.name)
        _copy_images(test_images, output_path / "test" / class_dir.name)

        total_moved["train"] += len(train_images)
        total_moved["val"] += len(val_images)
        total_moved["test"] += len(test_images)

    logger.info(f"Data split completed. Total images moved: {total_moved}")


def _copy_images(images: list[Path], dest_dir: Path) -> None:
    """Copy a list of images to a destination directory.

    Args:
        images: List of paths to the images.
        dest_dir: Destination directory path.

    Raises:
        OSError: If there is an issue during copying.
    """
    for img in images:
        try:
            shutil.copy2(img, dest_dir / img.name)
        except OSError as e:
            logger.error(f"Failed to copy {img} to {dest_dir}: {e}")
            raise
