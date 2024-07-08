from itakello_logging import ItakelloLogging

from src.classes.llm import LLM
from src.managers.download_manager import DownloadManager

# from src.evaluators.yolo_eval import YOLOEvaluator
from src.managers.preprocess_manager import PreprocessManager
from src.models.clip_model import CLIP
from src.utils.consts import DATA_PATH, LLM_MODEL, LLM_SYSTEM_PROMPT_PATH

ItakelloLogging(excluded_modules=[], debug=True)


def main() -> None:
    dm = DownloadManager(data_path=DATA_PATH)
    # dm.download_data(drive_url=DATASET_URL)

    llm = LLM(
        base_model=LLM_MODEL,
        system_prompt_path=LLM_SYSTEM_PROMPT_PATH,
    )
    clip = CLIP()
    pm = PreprocessManager(
        data_path=DATA_PATH,
        images_path=dm.images_path,
        raw_annotations_path=dm.annotations_path,
        llm=llm,
        clip=clip,
    )
    pm.process_data(sample_size=100)

    """
    yolo_eval = YOLOEvaluator(
        eval_name="yolo_baseline",
        yolo_versions=YOLO_VERSIONS,
        iou_thresholds=IOU_THRESHOLDS,
    )
    yolo_eval.evaluate()"""


if __name__ == "__main__":
    main()
