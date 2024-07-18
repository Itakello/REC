import torch

from ..interfaces.base_class import BaseClass


class BoundingBoxScalerManager(BaseClass):
    def __init__(self, img_width, img_height, min_area=2) -> None:
        self.img_width = img_width
        self.img_height = img_height
        self.min_area = min_area

    def scale_to_range(self, bbox) -> torch.Tensor:
        # Apply sigmoid to get values in the range [0, 1]
        bbox = torch.sigmoid(bbox)

        # Scale sigmoid output to image dimensions
        bbox[0] *= self.img_width  # x
        bbox[1] *= self.img_height  # y
        bbox[2] *= self.img_height  # h
        bbox[3] *= self.img_width  # w

        # Ensure valid bounding box coordinates
        bbox[0] = torch.clamp(bbox[0], min=0, max=self.img_width)  # x
        bbox[1] = torch.clamp(bbox[1], min=0, max=self.img_height)  # y
        max_height = self.img_height - bbox[1]
        max_width = self.img_width - bbox[0]
        bbox[2] = torch.clamp(
            bbox[2], min=torch.tensor(1.0, device=bbox.device), max=max_height
        )  # h
        bbox[3] = torch.clamp(
            bbox[3], min=torch.tensor(1.0, device=bbox.device), max=max_width
        )  # w

        # Ensure minimum area and non-zero dimensions
        if bbox[2] * bbox[3] < self.min_area:
            min_side = torch.sqrt(torch.tensor(self.min_area, device=bbox.device))
            bbox[2] = torch.clamp(min_side, min=1, max=max_height)
            bbox[3] = torch.clamp(min_side, min=1, max=max_width)

        # Adjust height and width if they result in zero area
        if bbox[2] <= 0 or bbox[3] <= 0:
            bbox[2] = torch.clamp(
                max_height, min=1
            )  # Adjust height to max possible if zero
            bbox[3] = torch.clamp(
                max_width, min=1
            )  # Adjust width to max possible if zero

        # Move x, y far enough from the borders if height or width are zero
        if bbox[2] == 1 and bbox[3] == 1:  # Means bbox is invalid near borders
            if bbox[0] == self.img_width:
                bbox[0] = (
                    self.img_width - bbox[3] - 1
                )  # Move x away from the right border
            if bbox[1] == self.img_height:
                bbox[1] = (
                    self.img_height - bbox[2] - 1
                )  # Move y away from the bottom border

        return bbox

    def is_valid_bbox(self, bbox) -> bool:
        x, y, h, w = bbox
        return (
            0 <= x <= self.img_width
            and 0 <= y <= self.img_height
            and 1 <= h <= (self.img_height - y)
            and 1 <= w <= (self.img_width - x)
            and h * w >= self.min_area
        )


def print_test_result(bbox_scaler, bbox):
    scaled_bbox = bbox_scaler.scale_to_range(bbox)
    valid = bbox_scaler.is_valid_bbox(scaled_bbox)
    print(f"Original bbox: {bbox}")
    print(f"Scaled bbox: {scaled_bbox}")
    print(f"Is valid: {valid}")
    print("-" * 50)


if __name__ == "__main__":
    img_width = 640
    img_height = 480
    min_area = 1  # Set the minimum area
    bbox_scaler_manager = BoundingBoxScalerManager(img_width, img_height, min_area)

    # Test case with normal values
    bbox = torch.tensor([0.5, 0.5, 0.5, 0.5])
    print_test_result(bbox_scaler_manager, bbox)

    # Test case with values close to zero
    bbox = torch.tensor([0.01, 0.01, 0.01, 0.01])
    print_test_result(bbox_scaler_manager, bbox)

    # Test case with values close to one
    bbox = torch.tensor([0.99, 0.99, 0.99, 0.99])
    print_test_result(bbox_scaler_manager, bbox)

    # Test case with extreme values (larger than one)
    bbox = torch.tensor([1.5, 1.5, 1.5, 1.5])
    print_test_result(bbox_scaler_manager, bbox)

    # Test case with negative values
    bbox = torch.tensor([-0.5, -0.5, -0.5, -0.5])
    print_test_result(bbox_scaler_manager, bbox)

    # Test case with mixed values
    bbox = torch.tensor([0.5, -0.5, 1.5, 0.5])
    print_test_result(bbox_scaler_manager, bbox)

    # Test case with high variance
    bbox = torch.tensor([0.1, 0.9, 0.2, 0.8])
    print_test_result(bbox_scaler_manager, bbox)

    # Test case with out of range values
    bbox = torch.tensor([150.0, 234.0, -0.2, -14.0])
    print_test_result(bbox_scaler_manager, bbox)
