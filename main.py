from itakello_logging import ItakelloLogging

from src.classes.llm import LLM
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

from .src.evaluations.yolo_eval import YOLOBaselineEval

ItakelloLogging(excluded_modules=[], debug=True)


def main() -> None:
    dm = DownloadManager(data_path=DATA_PATH)
    dm.download_data(drive_url=DATASET_URL)

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
    pm.process_data(sample_size=100)

    evaluator = YOLOBaselineEval(
        iou_thresholds=IOU_THRESHOLDS,
        yolo_versions=YOLO_VERSIONS,
    )
    metrics = evaluator.evaluate()
    print(metrics)


if __name__ == "__main__":
    main()
