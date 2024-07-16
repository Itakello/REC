import ast
import json
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from itakello_logging import ItakelloLogging
from PIL import Image
from tqdm import tqdm

from src.utils.calculate_iou import calculate_iou

from ..classes.highlighting_modality import HighlightingModality
from ..classes.llm import LLM
from ..interfaces.base_class import BaseClass
from ..models.clip_model import ClipModel
from ..models.yolo_model import YOLOModel
from ..utils.consts import CLIP_MODEL, IMAGES_PATH, PROCESSED_ANNOTATIONS_PATH
from ..utils.create_directory import create_directory

logger = ItakelloLogging().get_logger(__name__)


@dataclass
class PreprocessManager(BaseClass):
    data_path: Path
    raw_annotations_path: Path
    llm: LLM
    clip: ClipModel
    annotations_file_name: str = "annotations.csv"
    processed_annotations_path: Path = field(init=False)
    embeddings_path: Path = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.processed_annotations_path = create_directory(PROCESSED_ANNOTATIONS_PATH)
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
                    "sentences": [
                        str(sent["raw"]).replace("'", "`") for sent in ref["sentences"]
                    ],
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

    @staticmethod
    def preprocess_comprehensive_sentences(sentence: str) -> str:
        # Split the sentence into lines
        lines = sentence.split("\n")

        # Keep only the last non-empty line
        sentence = next((line for line in reversed(lines) if line.strip()), "")

        # Remove explanatory notes and comments
        sentence = re.sub(r"\(.*?\)", "", sentence)
        sentence = re.sub(r"I .*", "", sentence)
        sentence = re.sub(r"Let me know.*", "", sentence)
        sentence = re.sub(r"Note:.*", "", sentence)
        sentence = re.sub(r"This sentence captures.*", "", sentence)

        # Remove phrases like "Here's the output:", "Output:", etc.
        sentence = re.sub(
            r"^(Here\'s|Here is|Output:|Based on|Innovative input!|Single-sentence referential expression:|Single, concise referential expression:).*?:",
            "",
            sentence,
        )

        # Remove numbers at the start of sentences
        sentence = re.sub(r"^\d+\.\s*", "", sentence)

        # Remove any trailing periods and spaces
        sentence = sentence.rstrip(". ")

        # Remove quotes and adjust trailing periods
        sentence = sentence.strip('"').rstrip(".")

        return f"{sentence.strip().capitalize()}."

    def clean_comprehensive_sentences(self, df: pd.DataFrame) -> pd.DataFrame:
        # Count initial samples
        initial_count = len(df)

        # Remove samples from train and val sets where there are more than 1 line
        train_val_mask = (df["split"].isin(["train", "val"])) & (
            df["comprehensive_sentence"].str.count("\n") > 0
        )
        removed_count = train_val_mask.sum()
        df = df[~train_val_mask]

        # Identify test samples with more than 1 line
        test_mask = (df["split"] == "test") & (
            df["comprehensive_sentence"].str.count("\n") > 0
        )

        # Store original sentences
        original_sentences = df.loc[test_mask, "comprehensive_sentence"].copy()

        # Apply cleaning steps to test sentences
        df.loc[test_mask, "comprehensive_sentence"] = original_sentences.apply(
            self.preprocess_comprehensive_sentences
        )

        logger.info(f"Initial sample count: {initial_count}")
        logger.info(f"Removed samples count: {removed_count}")
        logger.info(f"Final sample count: {len(df)}")

        return df

    def encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting feature encoding process")

        total_rows = len(df)
        pbar = tqdm(total=total_rows, desc="Encoding features", unit="row")

        for idx, row in df.iterrows():
            # Load and encode image
            image_path = IMAGES_PATH / row["file_name"]
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

    def save_dataframe_to_csv(self, df: pd.DataFrame, file_name: str) -> None:
        file_path = self.processed_annotations_path / file_name
        df.to_csv(file_path, index=False)
        logger.confirmation(f"CSV file saved successfully: {file_path}")

    def get_dataframe_from_csv(self, file_name: str) -> pd.DataFrame:
        df = pd.read_csv(self.processed_annotations_path / file_name)
        df["sentences"] = df["sentences"].apply(ast.literal_eval)
        df["bbox"] = df["bbox"].apply(json.loads)
        if "yolo_predictions" in df.columns:
            df["yolo_predictions"] = df["yolo_predictions"].apply(json.loads)
        return df

    def process_data(self, sample_size: int | None = None) -> None:
        df = self.process_pickle_to_dataframe()
        self.save_dataframe_to_csv(df, file_name="0_from_pickle.csv")

        if sample_size is not None:
            df = df.sample(n=min(sample_size, len(df)), random_state=42)
            logger.info(f"Sampled {len(df)} rows for testing purposes")

        df = self.update_dataframe_with_json_data(df)
        self.save_dataframe_to_csv(df, file_name="1_added_json.csv")

        df = self.fix_bboxes(df)
        self.save_dataframe_to_csv(df, file_name="2_fixed_bboxes.csv")

        df = self.generate_comprehensive_sentence(df)
        self.save_dataframe_to_csv(df, file_name="3_added_comprehensive_sentences.csv")

        df = self.get_dataframe_from_csv(
            file_name="3_added_comprehensive_sentences.csv"
        )

        df = self.clean_comprehensive_sentences(df)
        self.save_dataframe_to_csv(
            df, file_name="4_cleaned_comprehensive_sentences.csv"
        )

        df = self.encode_features(df)
        self.save_dataframe_to_csv(df, file_name="5_encoded_features.csv")
        self.save_dataframe_to_csv(df, file_name=self.annotations_file_name)

    def add_yolo_predictions(
        self, df: pd.DataFrame, yolo_model: YOLOModel
    ) -> pd.DataFrame:
        logger.info("Starting YOLO prediction process")

        total_rows = len(df)
        pbar = tqdm(total=total_rows, desc="Adding YOLO predictions", unit="image")

        yolo_predictions = []

        for _, row in df.iterrows():
            image_path = IMAGES_PATH / row["file_name"]
            image = Image.open(image_path).convert("RGB")

            # Get YOLO predictions
            predictions = yolo_model.get_bboxes(image)

            # Convert predictions to list of lists for JSON serialization
            predictions_list = predictions.cpu().tolist()

            yolo_predictions.append(json.dumps(predictions_list))

            pbar.update(1)

        pbar.close()

        # Add YOLO predictions to the DataFrame
        df["yolo_predictions"] = yolo_predictions
        logger.confirmation("Updated CSV file saved with YOLO predictions")
        return df

    def filter_valid_samples(
        self, df: pd.DataFrame, iou_threshold: float = 0.5
    ) -> pd.DataFrame:
        valid_indices = []
        for index, row in df.iterrows():
            if row["split"] == "test":
                valid_indices.append(index)
                continue

            yolo_predictions = row["yolo_predictions"]
            gt_bbox = row["bbox"]

            # Check if any prediction matches the ground truth
            valid = any(
                calculate_iou(pred, gt_bbox)[0] >= iou_threshold
                for pred in yolo_predictions
            )
            if valid:
                valid_indices.append(index)

        df = df.loc[valid_indices]
        logger.confirmation(f"Filtered dataset to {len(df)} valid samples")
        return df

    def add_correct_candidate_idx(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting correct candidate index determination process")

        total_rows = len(df)
        pbar = tqdm(
            total=total_rows, desc="Adding correct candidate indices", unit="sample"
        )

        correct_candidate_indices = []
        correct_candidate_ious = []

        for _, row in df.iterrows():
            gt_bbox = row["bbox"]
            yolo_predictions = row["yolo_predictions"]

            # Calculate IoU for all predictions
            ious = [calculate_iou(pred, gt_bbox)[0] for pred in yolo_predictions]

            correct_candidate_indices.append(np.argmax(ious))
            correct_candidate_ious.append(np.max(ious))

            pbar.update(1)

        pbar.close()

        # Add correct candidate indices to the DataFrame
        df["correct_candidate_idx"] = correct_candidate_indices
        df["correct_candidate_iou"] = correct_candidate_ious

        logger.confirmation("Updated CSV file saved with correct candidate indices")
        return df

    def process_data_2(self, yolo_model: YOLOModel, iou_threshold: float = 0.5) -> None:
        df = self.get_dataframe_from_csv(file_name=self.annotations_file_name)

        # Add YOLO predictions
        df = self.add_yolo_predictions(df, yolo_model)
        self.save_dataframe_to_csv(
            df,
            file_name="6_added_yolo_predictions.csv",
        )

        # Filter valid samples
        df = self.filter_valid_samples(df, iou_threshold)
        self.save_dataframe_to_csv(
            df,
            file_name="7_filtered_valid_samples.csv",
        )

        df = self.add_correct_candidate_idx(df, iou_threshold)
        self.save_dataframe_to_csv(
            df,
            file_name="8_added_correct_candidate_idx_and_ious.csv",
        )
        self.save_dataframe_to_csv(df, file_name=self.annotations_file_name)

    def add_candidates_embeddings(self, highlighting_method: str) -> None:
        df = self.get_dataframe_from_csv(file_name=self.annotations_file_name)
        logger.info(
            f"Starting highlighting encoding process using {highlighting_method} method"
        )

        total_rows = len(df)
        pbar = tqdm(
            total=total_rows, desc="Adding highlighting embeddings", unit="image"
        )

        for idx, row in df.iterrows():
            image_path = IMAGES_PATH / row["file_name"]
            image = Image.open(image_path).convert("RGB")

            yolo_predictions = row["yolo_predictions"]

            # Apply highlighting to each prediction
            candidates_highlighted_images = [
                HighlightingModality().apply_highlighting(
                    image.copy(), bbox, highlighting_method
                )
                for bbox in yolo_predictions
            ]

            # Encode highlighted images
            candidates_embeddings = self.clip.encode_images(
                candidates_highlighted_images
            )

            # Load existing embeddings
            embedding_path = self.embeddings_path / row["embeddings_filename"]
            with np.load(embedding_path) as data:
                embeddings = {key: data[key] for key in data.files}

            # Add new highlighted embeddings
            embeddings["candidates_embeddings"] = candidates_embeddings.cpu().numpy()

            # Save updated embeddings
            np.savez_compressed(embedding_path, **embeddings)

            pbar.update(1)

        pbar.close()

        logger.confirmation(
            f"Highlighting encodings ({highlighting_method}) added to embeddings files"
        )

    def order_candidates_and_update_index(self) -> None:
        df = self.get_dataframe_from_csv(file_name=self.annotations_file_name)
        logger.info("Starting to order candidates and update correct candidate index")

        total_rows = len(df)
        pbar = tqdm(total=total_rows, desc="Ordering candidates", unit="sample")

        ordered_indices = []
        ordered_correct_indices = []

        for idx, row in df.iterrows():
            embedding_path = self.embeddings_path / row["embeddings_filename"]
            with np.load(embedding_path) as data:
                candidates_embeddings = torch.from_numpy(data["candidates_embeddings"])
                combined_sentences_features = torch.from_numpy(
                    data["combined_sentences_features"]
                )

            # Calculate similarities
            similarities = self.clip.get_similarity(
                candidates_embeddings, combined_sentences_features
            )

            # Get sorted indices
            sorted_indices = torch.argsort(similarities, descending=True).tolist()

            # Update ordered indices
            ordered_indices.append(json.dumps(sorted_indices))

            # Update correct candidate index
            correct_idx = row["correct_candidate_idx"]
            ordered_correct_idx = sorted_indices.index(correct_idx)
            ordered_correct_indices.append(ordered_correct_idx)

            pbar.update(1)

        pbar.close()

        # Add new columns to the DataFrame
        df["ordered_candidate_indices"] = ordered_indices
        df["ordered_correct_candidate_idx"] = ordered_correct_indices

        self.save_dataframe_to_csv(df, file_name="10_ordered_candidates.csv")
        self.save_dataframe_to_csv(df, file_name=self.annotations_file_name)
        logger.confirmation(
            "Updated CSV file with ordered candidates and correct indices"
        )

    def filter_train_samples_with_correct_candidate(self, top_k: int = 6) -> None:
        df = self.get_dataframe_from_csv(file_name=self.annotations_file_name)
        logger.info("Starting to filter train samples with correct candidate")

        initial_count = len(df)

        # Filter out samples where ordered_correct_candidate_idx is greater than or equal to top_k
        mask = (df["split"] == "train") & (df["ordered_correct_candidate_idx"] >= top_k)
        df = df[~mask]

        final_count = len(df)
        removed_count = initial_count - final_count

        logger.info(f"Initial sample count: {initial_count}")
        logger.info(f"Removed samples count: {removed_count}")
        logger.info(f"Final sample count: {final_count}")

        self.save_dataframe_to_csv(df, file_name="11_filtered_train_samples.csv")
        self.save_dataframe_to_csv(df, file_name=self.annotations_file_name)
        logger.confirmation(
            "Filtered train samples with correct candidate within top-k"
        )


if __name__ == "__main__":

    from ..utils.consts import DATA_PATH, LLM_MODEL, LLM_SYSTEM_PROMPT_PATH
    from .download_manager import DownloadManager

    dm = DownloadManager(data_path=DATA_PATH)
    llm = LLM(
        base_model=LLM_MODEL,
        system_prompt_path=LLM_SYSTEM_PROMPT_PATH,
    )
    clip = ClipModel(version=CLIP_MODEL)
    pm = PreprocessManager(
        data_path=DATA_PATH,
        raw_annotations_path=dm.annotations_path,
        llm=llm,
        clip=clip,
    )
    pm.process_data(sample_size=10)

    #  Add YOLO predictions
    yolo_model = YOLOModel(version="yolov5mu")
    pm.add_yolo_predictions(yolo_model=yolo_model)

    # Add highlighting embeddings
    highlighting_method = "ellipse"
    top_k = 6
    pm.add_top_candidates_embeddings(
        highlighting_method=highlighting_method, top_k=top_k
    )

    logger.confirmation("All preprocessing steps completed successfully")
