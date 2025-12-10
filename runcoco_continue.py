from ultralytics import YOLO

# 已训练 20 epoch 的 best.pt 路径
model_path = "/tmp/pycharm_project_284/runs/segment/yolov8xseg_coco2017_fullgpu/weights/best.pt"

# 加载模型
model = YOLO(model_path)

# 继续训练
model.train(
    data="coco2017.yaml",  # 同之前
    epochs=10,              # 继续训练多少 epoch
    resume=True              # 关键参数：继续训练
)
