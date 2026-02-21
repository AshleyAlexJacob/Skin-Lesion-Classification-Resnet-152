"""Organize HAM10000 raw images: merge part_1/part_2 and move into class subfolders."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
METADATA_FILENAME = "HAM10000_metadata.csv"
PART_1_DIRNAME = "HAM10000_images_part_1"
PART_2_DIRNAME = "HAM10000_images_part_2"
UNIFIED_DIRNAME = "HAM10000_images"


def _unified_dir(archive_dir: Path) -> Path:
    return archive_dir / UNIFIED_DIRNAME


def _part_dirs(archive_dir: Path) -> list[Path]:
    return [archive_dir / PART_1_DIRNAME, archive_dir / PART_2_DIRNAME]


def merge_image_folders(archive_dir: Path) -> None:
    """Move all images from part_1 and part_2 into a single unified folder.

    Creates archive_dir/HAM10000_images and moves every .jpg/.jpeg/.png from
    HAM10000_images_part_1 and HAM10000_images_part_2 into it. On name clash,
    skips the duplicate and logs.

    Args:
        archive_dir: Root directory containing part_1 and part_2 subdirs.

    Raises:
        OSError: If directory creation or file move fails.
    """
    unified = _unified_dir(archive_dir)
    unified.mkdir(parents=True, exist_ok=True)
    moved = 0
    skipped = 0
    for part_dir in _part_dirs(archive_dir):
        if not part_dir.is_dir():
            logger.warning("Part directory does not exist: %s", part_dir)
            continue
        for ext in IMAGE_EXTENSIONS:
            for src in part_dir.glob(f"*{ext}"):
                if not src.is_file():
                    continue
                dst = unified / src.name
                if dst.exists():
                    logger.debug("Skipping duplicate: %s", src.name)
                    skipped += 1
                    continue
                try:
                    shutil.move(str(src), str(dst))
                    moved += 1
                except OSError as e:
                    logger.exception("Failed to move %s: %s", src, e)
                    raise
    logger.info("Merge complete: %d moved, %d skipped (duplicates).", moved, skipped)


def organize_images_by_class(archive_dir: Path) -> None:
    """Move images from unified folder into dx-based subfolders using metadata.

    Reads HAM10000_metadata.csv (image_id, dx), creates UNIFIED_DIR/dx for each
    class, and moves each image from UNIFIED_DIR into the corresponding dx
    subfolder. Tries .jpg then .png for each image_id. Logs missing files and
    count of files left in unified (not in CSV).

    Args:
        archive_dir: Root directory containing metadata CSV and HAM10000_images.

    Raises:
        FileNotFoundError: If metadata CSV is missing.
        OSError: If file operations fail.
    """
    unified = _unified_dir(archive_dir)
    metadata_path = archive_dir / METADATA_FILENAME
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    df = pd.read_csv(metadata_path)
    if "image_id" not in df.columns or "dx" not in df.columns:
        raise ValueError(
            f"Metadata must contain 'image_id' and 'dx'; got {list(df.columns)}"
        )
    # First occurrence wins for duplicate image_id
    id_to_dx = df.drop_duplicates(subset="image_id", keep="first").set_index(
        "image_id"
    )["dx"]

    moved = 0
    missing = 0
    for image_id, dx in id_to_dx.items():
        image_id = str(image_id).strip()
        dx = str(dx).strip()
        class_dir = unified / dx
        class_dir.mkdir(parents=True, exist_ok=True)
        src = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = unified / f"{image_id}{ext}"
            if candidate.is_file():
                src = candidate
                break
        if src is None:
            logger.warning("Image not found for image_id=%s", image_id)
            missing += 1
            continue
        dst = class_dir / src.name
        if dst == src:
            continue
        try:
            shutil.move(str(src), str(dst))
            moved += 1
        except OSError as e:
            logger.exception("Failed to move %s -> %s: %s", src, dst, e)
            raise

    leftover = sum(1 for _ in unified.iterdir() if _.is_file())
    logger.info(
        "Organize by class: %d moved, %d missing from disk, %d files left in root.",
        moved,
        missing,
        leftover,
    )


def run_organize_ham10000_raw(archive_dir: Path | None = None) -> None:
    """Run merge then organize: one entry point for HAM10000 raw layout.

    Step 1: Move all images from part_1 and part_2 into HAM10000_images.
    Step 2: Move each image from HAM10000_images into HAM10000_images/{dx}/ using
    HAM10000_metadata.csv.

    Args:
        archive_dir: Directory containing part_1, part_2, and metadata CSV. If
            None, uses data/raw/archive relative to project root (src/pipelines
            -> two parents up -> project root).

    Raises:
        FileNotFoundError: If metadata is missing (during organize step).
        OSError: If directory or file operations fail.
    """
    try:
        if archive_dir is None:
            # src/pipelines/organize_ham10000_raw.py -> project root = parents[2]
            project_root = Path(__file__).resolve().parents[2]
            archive_dir = project_root / "data" / "raw" / "archive"
        archive_dir = Path(archive_dir)
        if not archive_dir.is_dir():
            raise FileNotFoundError(f"Archive directory not found: {archive_dir}")

        try:
            merge_image_folders(archive_dir)
        except Exception as merge_exc:
            logger.exception("Error during merging image folders: %s", merge_exc)
            raise

        try:
            organize_images_by_class(archive_dir)
        except Exception as org_exc:
            logger.exception("Error during organizing images by class: %s", org_exc)
            raise

    except (FileNotFoundError, OSError) as e:
        logger.exception("run_organize_ham10000_raw failed: %s", e)
        raise
    except Exception as e:
        logger.exception("Unexpected error in run_organize_ham10000_raw: %s", e)
        raise
