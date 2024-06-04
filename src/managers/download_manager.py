import shutil
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

import gdown

DATA_PATH = Path("./data")


@dataclass
class DownloadManager:
    data_path: Path
    images_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.images_path = self.data_path / "images"

    def download_data(self, drive_url: str) -> None:
        # Define the URL and the name of the compressed file
        compressed_file_name = self.data_path / "file.tar.gz"

        # Download the tar.gz file
        gdown.download(drive_url, output=compressed_file_name, quiet=False)

        # Extract the tar.gz file into the target directory
        with tarfile.open(compressed_file_name, "r:gz") as file:
            file.extractall(path=self.data_path)

        # Move the images folder to the images_path
        shutil.move(self.data_path / "refcocog" / "images", self.images_path)

        # Clean up: remove the tar.gz file and the extracted folder
        compressed_file_name.unlink()
        shutil.rmtree(self.data_path / "refcocog")


if __name__ == "__main__":
    dm = DownloadManager(data_path=DATA_PATH)
    dm.download_data(
        drive_url="https://drive.google.com/uc?id=1xijq32XfEm6FPhUb7RsZYWHc2UuwVkiq"
    )
