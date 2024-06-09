from pathlib import Path

from itakello_logging import ItakelloLogging

from src.evaluators.yolo_eval import YOLOEvaluator
from src.models.clip import CLIP

from src.managers.download_manager import DownloadManager

ItakelloLogging(excluded_modules=[], debug=True)


def main() -> None:
    dm = DownloadManager(data_path=Path("./data"))
    dm.download_data(
        drive_url="https://drive.google.com/uc?id=1xijq32XfEm6FPhUb7RsZYWHc2UuwVkiq"
    )

    """clip_model = CLIP()

    yolo_eval = YOLOEvaluator(
        eval_name="yolo_baseline",
        yolo_versions=[
            "yolov5nu.pt",
            "yolov5su.pt",
            "yolov5mu.pt",
            "yolov5lu.pt",
            "yolov5xu.pt",
            "yolov8n.pt",
            "yolov8s.pt",
            "yolov8m.pt",
            "yolov8l.pt",
            "yolov8x.pt",
        ],
        iou_thresholds=[0.1, 0.3, 0.5, 0.7, 0.9],
    )
    yolo_eval.evaluate()"""


if __name__ == "__main__":
    main()
