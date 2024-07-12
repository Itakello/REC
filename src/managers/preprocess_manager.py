import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from itakello_logging import ItakelloLogging
from PIL import Image
from tqdm import tqdm

from ..classes.llm import LLM
from ..interfaces.base_class import BaseClass
from ..models.clip_model import ClipModel
from ..utils.consts import CLIP_MODEL, MODELS_PATH
from ..utils.create_directory import create_directory

logger = ItakelloLogging().get_logger(__name__)


@dataclass
class PreprocessManager(BaseClass):
    data_path: Path
    images_path: Path
    raw_annotations_path: Path
    llm: LLM
    clip: ClipModel
    annotations_path: Path = field(init=False)
    embeddings_path: Path = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.annotations_path = self.data_path / "annotations.csv"
        self.embeddings_path = create_directory(self.data_path / "embeddings")

    def process_pickle_to_dataframe(self) -> pd.DataFrame:
        pickle_file = self.raw_annotations_path / "refs(umd).p"
        if not pickle_file.exists():
            logger.error(f"refs(umd).p not found in {self.raw_annotations_path}")
            return pd.DataFrame()

        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f)

        rows = []
        for ref in pickle_data:
            rows.append(
                {
                    "image_id": ref[
                        "image_id"
                    ],  # Keeping these temporarily for joining with JSON data
                    "split": ref["split"],
                    "sentences": [sent["raw"] for sent in ref["sentences"]],
                    "category_id": ref["category_id"],
                    "ann_id": ref["ann_id"],
                }
            )

        df = pd.DataFrame(rows)
        logger.confirmation("DataFrame created successfully from pickle data")
        return df

    def update_dataframe_with_json_data(self, df: pd.DataFrame) -> pd.DataFrame:
        instances_file = self.raw_annotations_path / "instances.json"
        if not instances_file.exists():
            logger.error(f"instances.json not found in {self.raw_annotations_path}")
            return df

        with open(instances_file, "r") as f:
            json_data = json.load(f)

        images_dict = {img["id"]: img for img in json_data["images"]}
        annotations_dict = {ann["id"]: ann for ann in json_data["annotations"]}
        categories_dict = {cat["id"]: cat for cat in json_data["categories"]}

        df["file_name"] = df["image_id"].map(
            lambda x: images_dict.get(x, {}).get("file_name")
        )
        df["height"] = df["image_id"].map(
            lambda x: images_dict.get(x, {}).get("height")
        )
        df["width"] = df["image_id"].map(lambda x: images_dict.get(x, {}).get("width"))
        df["area"] = df["ann_id"].map(lambda x: annotations_dict.get(x, {}).get("area"))
        df["iscrowd"] = df["ann_id"].map(
            lambda x: annotations_dict.get(x, {}).get("iscrowd")
        )
        df["bbox"] = df["ann_id"].map(
            lambda x: json.dumps(annotations_dict.get(x, {}).get("bbox"))
        )
        df["supercategory"] = df["category_id"].map(
            lambda x: categories_dict.get(x, {}).get("supercategory")
        )
        df["category"] = df["category_id"].map(
            lambda x: categories_dict.get(x, {}).get("name")
        )

        # Remove the columns we don't need
        columns_to_keep = [
            "file_name",
            "split",
            "sentences",
            "height",
            "width",
            "area",
            "iscrowd",
            "bbox",
            "supercategory",
            "category",
        ]
        df = df[columns_to_keep]

        logger.confirmation(
            "DataFrame updated with JSON data and unnecessary columns removed"
        )
        return df

    def fix_bboxes(self, df: pd.DataFrame) -> pd.DataFrame:
        def fix_bbox(bbox_str) -> str:
            bbox = json.loads(bbox_str)
            x, y, w, h = bbox
            return json.dumps([x, y, x + w, y + h])

        df["bbox"] = df["bbox"].apply(fix_bbox)

        logger.confirmation("Bounding boxes formats fixed")
        return df

    def generate_comprehensive_sentence(self, df: pd.DataFrame) -> pd.DataFrame:
        total_rows = len(df)

        # Initialize tqdm progress bar
        pbar = tqdm(
            total=total_rows,
            desc="Generating comprehensive sentences",
            unit="sentence",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )

        def generate_with_progress(sentences: list[str]) -> str:
            result = self.llm.generate(sentences)
            pbar.update(1)
            return result

        df["comprehensive_sentence"] = df["sentences"].apply(generate_with_progress)
        pbar.close()
        logger.confirmation("Comprehensive sentences generated successfully")

        return df

    def encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting feature encoding process")

        total_rows = len(df)
        pbar = tqdm(total=total_rows, desc="Encoding features", unit="row")

        for idx, row in df.iterrows():
            # Load and encode image
            image_path = self.images_path / row["file_name"]
            image = Image.open(image_path).convert("RGB")
            image_features = self.clip.encode_images(image)

            # Encode original sentences
            original_sentences = row["sentences"]
            original_sentences_features = self.clip.encode_sentences(
                original_sentences, average=True
            )

            # Encode comprehensive sentence
            comprehensive_sentence = row["comprehensive_sentence"]
            comprehensive_sentence_features = self.clip.encode_sentences(
                comprehensive_sentence
            )

            # Encode combined sentences
            combined_sentences = original_sentences + [comprehensive_sentence]
            combined_sentences_features = self.clip.encode_sentences(
                combined_sentences, average=True
            )

            # Save embeddings
            embeddings = {
                "image_features": image_features.cpu().numpy(),
                "original_sentences_features": original_sentences_features.cpu().numpy(),
                "comprehensive_sentence_features": comprehensive_sentence_features.cpu().numpy(),
                "combined_sentences_features": combined_sentences_features.cpu().numpy(),
            }
            embedding_filename = f"embedding_{idx}.npz"
            embedding_path = self.embeddings_path / embedding_filename
            np.savez_compressed(embedding_path, **embeddings)

            # Store only the filename of embeddings in the DataFrame
            df.at[idx, "embeddings_filename"] = embedding_filename

            pbar.update(1)

        pbar.close()

        logger.confirmation(
            f"Feature encoding completed and saved successfully: {self.embeddings_path}"
        )
        return df

    def save_dataframe_to_csv(self, df: pd.DataFrame) -> None:
        df.to_csv(self.annotations_path, index=False)
        logger.confirmation(f"CSV file saved successfully: {self.annotations_path}")

    def process_data(self, sample_size: int | None = None) -> None:
        df = self.process_pickle_to_dataframe()

        if sample_size is not None:
            df = df.sample(n=min(sample_size, len(df)), random_state=42)
            logger.info(f"Sampled {len(df)} rows for testing purposes")

        df = self.update_dataframe_with_json_data(df)
        df = self.fix_bboxes(df)
        df = self.generate_comprehensive_sentence(df)
        df = self.encode_features(df)
        self.save_dataframe_to_csv(df)


if __name__ == "__main__":

    from ..utils.consts import DATA_PATH, LLM_MODEL, LLM_SYSTEM_PROMPT_PATH
    from .download_manager import DownloadManager

    dm = DownloadManager(data_path=DATA_PATH)
    llm = LLM(
        base_model=LLM_MODEL,
        system_prompt_path=LLM_SYSTEM_PROMPT_PATH,
    )
    clip = ClipModel(version=CLIP_MODEL, models_path=MODELS_PATH)
    pm = PreprocessManager(
        data_path=DATA_PATH,
        images_path=dm.images_path,
        raw_annotations_path=dm.annotations_path,
        llm=llm,
        clip=clip,
    )
    pm.process_data(sample_size=10)
