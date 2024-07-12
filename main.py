import os

from itakello_logging import ItakelloLogging

from src.classes.llm import LLM
from src.evaluations.yolo_baseline_eval import YOLOBaselineEval
from src.managers.download_manager import DownloadManager
from src.managers.preprocess_manager import PreprocessManager
from src.models.clip_model import ClipModel
from src.utils.consts import (
    CLIP_MODEL,
    DATA_PATH,
    DATASET_URL,
    HIGHLIGHTING_METHODS,
    IOU_THRESHOLDS,
    LLM_MODEL,
    LLM_SYSTEM_PROMPT_PATH,
    MODELS_PATH,
    SENTENCES_TYPES,
    YOLO_VERSIONS,
)

from .src.evaluations.similarity_baseline_eval import SimilarityBaselineEval
from .src.models.yolo_model import YOLOModel

ItakelloLogging(excluded_modules=[], debug=True)

os.environ["WANDB_SILENT"] = "true"


def main() -> None:
    dm = DownloadManager(data_path=DATA_PATH)
    # dm.download_data(drive_url=DATASET_URL)

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
    # pm.process_data(sample_size=100)

    yolo_baseline_eval = YOLOBaselineEval(
        iou_thresholds=IOU_THRESHOLDS,
        yolo_versions=YOLO_VERSIONS,
    )
    yolo_metrics = yolo_baseline_eval.evaluate()

    # NOTE: 1 - Choose YOLO model and IOU threshold

    yolo_model = YOLOModel(version="yolov5mu", models_path=MODELS_PATH)
    iou_threshold = 0.5

    similarity_baseline_eval = SimilarityBaselineEval(
        highlighting_methods=HIGHLIGHTING_METHODS, sentences_types=SENTENCES_TYPES
    )
    similarity_metrics = similarity_baseline_eval.evaluate()

    # NOTE: 2 - Chooose highlighting method and sentence type

    highlighting_method = "ellipse"
    sentences_type = "combined_sentences"


if __name__ == "__main__":
    main()
