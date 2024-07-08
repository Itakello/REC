import ast
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from itakello_logging import ItakelloLogging
from tqdm import tqdm

from ..classes.llm import LLM
from ..interfaces.base_class import BaseClass
from ..models.clip import CLIP

logger = ItakelloLogging().get_logger(__name__)


@dataclass
class PreprocessManager(BaseClass):
    data_path: Path
    images_path: Path
    annotations_path: Path
    llm: LLM
    clip: CLIP
    preprocessed_path: Path = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.preprocessed_path = self.data_path / "annotations.csv"

    def process_pickle_to_dataframe(self) -> pd.DataFrame:
        pickle_file = self.annotations_path / "refs(umd).p"
        if not pickle_file.exists():
            logger.error(f"refs(umd).p not found in {self.annotations_path}")
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
        instances_file = self.annotations_path / "instances.json"
        if not instances_file.exists():
            logger.error(f"instances.json not found in {self.annotations_path}")
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
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Encoding features"):
            # Encode image
            image = Image.open(self.images_path / row["file_name"]).convert("RGB")
            image_feature = self.encode_single_image(image)
            df.at[idx, "image_feature"] = image_feature.tolist()

            # Encode sentences
            sentences = eval(row["sentences"])
            sentences_feature = self.clip.encode_sentences(sentences).cpu().numpy()
            df.at[idx, "sentences_feature"] = sentences_feature.tolist()

            # Encode comprehensive sentence
            comp_sentence_feature = self.encode_single_sentence(
                row["comprehensive_sentence"]
            )
            df.at[idx, "comprehensive_sentence_feature"] = (
                comp_sentence_feature.tolist()
            )

            # Encode sentences + comprehensive sentence
            combined_sentences = sentences + [row["comprehensive_sentence"]]
            combined_feature = (
                self.clip.encode_sentences(combined_sentences).cpu().numpy()
            )
            df.at[idx, "combined_feature"] = combined_feature.tolist()

        return df

    def save_dataframe_to_csv(self, df: pd.DataFrame) -> None:
        df.to_csv(self.preprocessed_path, index=False)
        logger.confirmation(f"CSV file saved successfully: {self.preprocessed_path}")

    def process_data(self, sample_size: int | None = None) -> None:
        df = self.process_pickle_to_dataframe()

        if sample_size is not None:
            df = df.sample(n=min(sample_size, len(df)), random_state=42)
            logger.info(f"Sampled {len(df)} rows for comprehensive sentence generation")

        df = self.update_dataframe_with_json_data(df)
        df = self.fix_bboxes(df)
        df = self.generate_comprehensive_sentence(df)
        df = self.encode_features(df)
        self.save_dataframe_to_csv(df)


if __name__ == "__main__":
    llm = LLM(
        base_model="llama3",
        system_prompt_path=Path("prompts/referential-expression-prompt.txt"),
    )
    clip = CLIP()
    pm = PreprocessManager(
        data_path=Path("./data"),
        images_path=Path("./data/images"),
        annotations_path=Path("./data/annotations"),
        llm=llm,
        clip=clip,
    )
    pm.process_data(sample_size=2)
