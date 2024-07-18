import os

from itakello_logging import ItakelloLogging

from src.classes.llm import LLM
from src.datasets.classification_dataset import ClassificationDataset
from src.datasets.regression_dataset import RegressionDataset
from src.evaluations.highlighting_method_baseline_eval import HighlightingMethodEval
from src.evaluations.similarity_baseline_eval import SimilarityBaselineEval
from src.evaluations.yolo_baseline_eval import YOLOBaselineEval
from src.managers.download_manager import DownloadManager
from src.managers.preprocess_manager import PreprocessManager
from src.managers.trainer_manager import TrainerManager
from src.models.classification_v0_model import ClassificationV0Model
from src.models.classification_v1_model import ClassificationV1Model
from src.models.classification_v2_model import ClassificationV2Model
from src.models.clip_model import ClipModel
from src.models.regression_v1_model import RegressionV1Model
from src.models.regression_v2_model import RegressionV2Model
from src.models.yolo_model import YOLOModel
from src.utils.consts import (
    CLIP_MODEL,
    CONFIG_PATH,
    DATA_PATH,
    DATASET_URL,
    HIGHLIGHTING_METHODS,
    IOU_THRESHOLDS,
    LLM_MODEL,
    LLM_SYSTEM_PROMPT_PATH,
    MODELS_PATH,
    PROCESSED_DATA_URL,
    SENTENCES_TYPES,
    YOLO_VERSIONS,
)

ItakelloLogging(
    excluded_modules=["httpcore.http11", "httpcore.connection", "httpx"], debug=True
)

os.environ["WANDB_SILENT"] = "true"


def main() -> None:
    """dm = DownloadManager(data_path=DATA_PATH)
    dm.download_data(drive_url=DATASET_URL)
    dm.download_drive_folders(drive_url=PROCESSED_DATA_URL)

    llm = LLM(
        base_model=LLM_MODEL,
        system_prompt_path=LLM_SYSTEM_PROMPT_PATH,
    )
    clip = ClipModel(version=CLIP_MODEL)
    pm = PreprocessManager(
        data_path=DATA_PATH,
        raw_annotations_path=annotations_path,
        llm=llm,
        clip=clip,
    )
    pm.process_data()

    yolo_baseline_eval = YOLOBaselineEval(
        iou_thresholds=IOU_THRESHOLDS,
        yolo_versions=YOLO_VERSIONS,
    )
    yolo_metrics = yolo_baseline_eval.evaluate()

    # NOTE: 1 - Choose YOLO model and IOU threshold

    iou_threshold = 0.8
    yolo_model = YOLOModel(version="yolov8x")

    pm.process_data_2(yolo_model=yolo_model, iou_threshold=iou_threshold)

    similarity_baseline_eval = SimilarityBaselineEval(
        highlighting_methods=HIGHLIGHTING_METHODS, sentences_types=SENTENCES_TYPES
    )
    similarity_metrics = similarity_baseline_eval.evaluate()

    # NOTE: 2 - Chooose best sentence type

    sentences_type = "combined_sentences"

    highlighting_method_baseline_eval = HighlightingMethodEval(
        highlighting_methods=HIGHLIGHTING_METHODS, sentences_type=sentences_type
    )
    highlighting_method_baseline_eval.evaluate()

    highlighting_method = "crop"
    top_k = 6

    pm.process_data_3(highlighting_method=highlighting_method, top_k=top_k)"""
    config_path = CONFIG_PATH / "trainer_config.json"

    """trainer_cl_v0 = TrainerManager(
        model_class=ClassificationV0Model,
        config_path=config_path,
        dataset_cls=ClassificationDataset,
        dataset_limit=20000,
    )

    trainer_cl_v0.train(epochs=10, use_combinations=True)"""

    """trainer = TrainerManager(
        model_class=RegressionV1Model,
        config_path=config_path,
        dataset_cls=RegressionDataset,
        dataset_limit=20000,
        is_regression=True,
    )

    trainer.train(epochs=10, use_combinations=True)"""

    trainer_cl_v2 = TrainerManager(
        model_class=ClassificationV2Model,
        config_path=config_path,
        dataset_cls=ClassificationDataset,
        dataset_limit=20000,
    )

    trainer_cl_v2.train(epochs=10, use_combinations=True)


if __name__ == "__main__":
    main()
