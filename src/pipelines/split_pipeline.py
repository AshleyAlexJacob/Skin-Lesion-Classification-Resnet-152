"""Data split pipeline: split dataset into train, val, and test sets."""

from __future__ import annotations

import logging
from pathlib import Path

from src.data.split_dataset import split_data

logger = logging.getLogger(__name__)


def run() -> None:
    """Run the data split pipeline.

    Splits the processed dataset into train, val, and test splits (63/27/10 ratio).
    """
    logger.info("Starting data split pipeline.")

    # The dataset to be split according to user specification
    input_dir_path = Path("data/raw/archive/HAM10000_images/")
    output_dir_path = Path("data/processed")
    
    if input_dir_path.exists():
        logger.info(f"Splitting data from {input_dir_path} into train/val/test.")
        logger.info(f"Saving splits to {output_dir_path.absolute()}")
        split_data(
            input_dir=input_dir_path,
            output_dir=output_dir_path,
            train_ratio=0.63,
            val_ratio=0.27,
            test_ratio=0.10,
            seed=42,
        )
    else:
        logger.error(f"Input directory does not exist: {input_dir_path}")

    logger.info("Data split pipeline finished.")


def main() -> None:
    """Entry point when run as a script (e.g. python -m src.pipelines.split_pipeline)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run()


if __name__ == "__main__":
    main()
