from ultralytics import YOLO
from pathlib import Path

if __name__ == "__main__":
    data_yaml = Path("coco2017.yaml")
    # ★ 加载你改动后的模型结构（用 YAML，而不是 .pt）
    model = YOLO(r"E:\ultralytics-main\ultralytics\cfg\models\v8\yolov8_ghostseg.yaml")
    model.train(
        data=data_yaml,
        imgsz=640,
        epochs=10,
        batch=2,
        workers=8,
        device=0,
        cache=True,
        amp=True,
        dropout=0.0,
        multi_scale=False,
        name="yolov8_ghostseg_coco2017",
        pretrained=False  # ★重要：不要加载官方预训练权重
    )

