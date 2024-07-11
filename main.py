import os

from itakello_logging import ItakelloLogging

from src.classes.llm import LLM
from src.evaluations.yolo_eval import YOLOBaselineEval
from src.managers.download_manager import DownloadManager
from src.managers.preprocess_manager import PreprocessManager
from src.models.clip_model import ClipModel
from src.utils.consts import (
    CLIP_MODEL,
    DATA_PATH,
    DATASET_URL,
    IOU_THRESHOLDS,
    LLM_MODEL,
    LLM_SYSTEM_PROMPT_PATH,
    MODELS_PATH,
    YOLO_VERSIONS,
)

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
    metrics = yolo_baseline_eval.evaluate()

    # NOTE: 1 - Choose YOLO model and IOU threshold

    yolo_model = YOLOModel(version="yolov5mu", models_path=MODELS_PATH)
    iou_threshold = 0.5


if __name__ == "__main__":
    main()
