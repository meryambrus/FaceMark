from __future__ import annotations

from pathlib import Path


SUPPORTED_IMAGE_EXTENSIONS = (".bmp", ".png", ".jpeg", ".jpg")
SUPPORTED_IMAGE_WILDCARD = "Supported image files (*.bmp;*.png;*.jpeg;*.jpg)|*.bmp;*.png;*.jpeg;*.jpg"


def isSupportedImagePath(imagePath: str | Path) -> bool:
    return Path(imagePath).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def listSupportedImagesInFolder(folderPath: str | Path) -> list[Path]:
    folder = Path(folderPath)
    return sorted(
        [
            path
            for path in folder.iterdir()
            if path.is_file() and isSupportedImagePath(path)
        ],
        key=lambda path: path.name.lower(),
    )
