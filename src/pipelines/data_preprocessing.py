"""Data preprocessing pipeline: organize raw HAM10000 and prepare for training."""

from __future__ import annotations

import logging

from src.data.organize_ham10000_raw import run_organize_ham10000_raw

logger = logging.getLogger(__name__)


def run() -> None:
    """Run the full data preprocessing pipeline.

    Currently runs HAM10000 raw-data organization (merge part_1/part_2 into one
    folder, then move images into class subfolders using metadata). Future steps
    (e.g. apply transforms, write to data/processed/) can be added here.
    """
    logger.info("Starting data preprocessing pipeline.")
    run_organize_ham10000_raw()
    # TODO: Apply transforms and write to data/processed/ when implemented.
    logger.info("Data preprocessing pipeline finished.")


def main() -> None:
    """Entry point when run as a script (e.g. python -m src.pipelines.data_preprocessing)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run()


if __name__ == "__main__":
    main()
