import shutil
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

import gdown
from itakello_logging import ItakelloLogging

from ..interfaces.base_class import BaseClass
from ..utils.consts import DATA_PATH, DATASET_URL, IMAGES_PATH
from ..utils.create_directory import create_directory

logger = ItakelloLogging().get_logger(__name__)


@dataclass
class DownloadManager(BaseClass):
    data_path: Path
    annotations_path: Path = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.images_path = create_directory(IMAGES_PATH)
        self.annotations_path = create_directory(self.data_path / "annotations")

    def download_data(self, drive_url: str) -> None:
        compressed_file_name = self.data_path / "file.tar.gz"

        # Download the tar.gz file
        logger.info(f"Downloading data from {drive_url}")
        gdown.download(drive_url, output=str(compressed_file_name), quiet=True)

        # Extract the tar.gz file into the target directory
        logger.info(f"Extracting {compressed_file_name}")
        with tarfile.open(compressed_file_name, "r:gz") as file:
            file.extractall(path=self.data_path)

        # Move the images folder to the images_path
        source_images = self.data_path / "refcocog" / "images"
        logger.info(f"Moving images from {source_images} to {self.images_path}")
        if source_images.exists():
            for item in source_images.iterdir():
                shutil.move(str(item), str(self.images_path))
            shutil.rmtree(source_images)
        else:
            logger.warning(f"Source images directory not found: {source_images}")

        # Move the annotations folder to the annotations_path
        source_annotations = self.data_path / "refcocog" / "annotations"
        logger.info(
            f"Moving annotations from {source_annotations} to {self.annotations_path}"
        )
        if source_annotations.exists():
            for item in source_annotations.iterdir():
                shutil.move(str(item), str(self.annotations_path))
            shutil.rmtree(source_annotations)
        else:
            logger.warning(
                f"Source annotations directory not found: {source_annotations}"
            )

        # Clean up: remove the tar.gz file and the extracted folder
        logger.info("Cleaning up temporary files")
        compressed_file_name.unlink()
        shutil.rmtree(self.data_path / "refcocog", ignore_errors=True)

        logger.confirmation("Dataset downloaded and organized successfully")


if __name__ == "__main__":
    dm = DownloadManager(
        data_path=DATA_PATH,
    )
    dm.download_data(drive_url=DATASET_URL)
