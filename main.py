import os

from itakello_logging import ItakelloLogging

from src.classes.llm import LLM
from src.evaluations.highlighting_method_baseline_eval import HighlightingMethodEval
from src.evaluations.similarity_baseline_eval import SimilarityBaselineEval
from src.evaluations.yolo_baseline_eval import YOLOBaselineEval
from src.managers.download_manager import DownloadManager
from src.managers.preprocess_manager import PreprocessManager
from src.models.clip_model import ClipModel
from src.models.yolo_model import YOLOModel
from src.utils.consts import (
    CLIP_MODEL,
    DATA_PATH,
    HIGHLIGHTING_METHODS,
    IOU_THRESHOLDS,
    LLM_MODEL,
    LLM_SYSTEM_PROMPT_PATH,
    MODELS_PATH,
    SENTENCES_TYPES,
    YOLO_VERSIONS,
)

ItakelloLogging(
    excluded_modules=["httpcore.http11", "httpcore.connection", "httpx"], debug=True
)

os.environ["WANDB_SILENT"] = "true"


def main() -> None:
    dm = DownloadManager(data_path=DATA_PATH)
    # dm.download_data(drive_url=DATASET_URL)

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
    # pm.process_data()

    """yolo_baseline_eval = YOLOBaselineEval(
    iou_thresholds=IOU_THRESHOLDS,
    yolo_versions=YOLO_VERSIONS,
    )
    yolo_metrics = yolo_baseline_eval.evaluate()"""

    # NOTE: 1 - Choose YOLO model and IOU threshold

    iou_threshold = 0.8
    yolo_model = YOLOModel(version="yolov8x")

    """pm.process_data_2(yolo_model=yolo_model, iou_threshold=iou_threshold)

    similarity_baseline_eval = SimilarityBaselineEval(
        highlighting_methods=HIGHLIGHTING_METHODS, sentences_types=SENTENCES_TYPES
    )
    similarity_metrics = similarity_baseline_eval.evaluate()"""

    # NOTE: 2 - Chooose best sentence type

    sentences_type = "combined_sentences"

    highlighting_method_baseline_eval = HighlightingMethodEval(
        highlighting_methods=HIGHLIGHTING_METHODS, sentences_type=sentences_type
    )
    highlighting_method_baseline_eval.evaluate()

    highlighting_method = "crop"
    top_k = 6

    pm.process_data_3(highlighting_method=highlighting_method, top_k=top_k)


if __name__ == "__main__":
    main()
