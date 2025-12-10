from ultralytics import YOLO
# 直接加载YAML，无需额外参数
model = YOLO(r"E:\ultralytics-main\ultralytics\cfg\models\v8\yolov8_ghostseg.yaml")
print("✅ 模型加载成功！无任何报错")