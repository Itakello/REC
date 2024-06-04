import wandb

# Assuming you have the following data
iou_values = [0.1, 0.2, 0.3, 0.4, 0.5]
results = {
    "yolo8x": [70, 65, 60, 55, 50],
    "yolo8m": [75, 70, 65, 60, 55],
    "yolo5u": [50, 45, 40, 35, 30],
}

for yolo_v, accuracies in results.items():
    wandb.init(project="Test", name=yolo_v)
    for index, iou in enumerate(iou_values):
        wandb.log({f"yolo_baseline/iou_{iou}": accuracies[index]})
    wandb.finish()
