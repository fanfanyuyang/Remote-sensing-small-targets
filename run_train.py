from ultralytics import YOLO
from pathlib import Path

if __name__ == '__main__':
    # 使用相对路径
    data_yaml = Path("cityscapes.yaml")  # 相对于 run_train.py 的路径
    model = YOLO("yolov8x-seg.pt")      # 模型权重也用相对路径
    model.train(
        data=data_yaml,
        imgsz=1024,
        epochs=100,
        batch=4,
        device=0
    )
