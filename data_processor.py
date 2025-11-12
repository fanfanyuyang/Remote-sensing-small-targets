import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# DOTAv1.5 类别列表（15类）
DOTA_CLASSES = [
    'plane', 'ship', 'storage-tank', 'baseball-diamond',
    'tennis-court', 'basketball-court', 'ground-track-field',
    'harbor', 'bridge', 'large-vehicle', 'small-vehicle',
    'helicopter', 'roundabout', 'soccer-ball-field', 'swimming-pool'
]
CLASS2ID = {cls: i for i, cls in enumerate(DOTA_CLASSES)}
ID2CLASS = {i: cls for i, cls in enumerate(DOTA_CLASSES)}
IMG_SIZE = 640  # 与模型输入尺寸匹配

# 在data_processor.py中添加一个全局的collate函数
def collate_fn(batch):
    """处理可变长度的标注（替代lambda）"""
    return batch
class DOTADataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None, is_train=True):
        self.images_dir = images_dir  # 图像目录
        self.labels_dir = labels_dir  # 标注目录
        self.transform = transform
        self.is_train = is_train
        # 1. 收集所有图像文件名（确保只包含图片文件）
        self.image_names = [
            f for f in os.listdir(images_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))  # 过滤非图片文件
        ]
        if not self.image_names:
            raise ValueError(f"图像目录 {images_dir} 中未找到图片文件！")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # ---------------------- 定义 img_name ----------------------
        # 从图像列表中获取当前图像的文件名（解决：未解析的引用 'img_name'）
        img_name = self.image_names[idx]  # 例如："P0001.png"

        # ---------------------- 读取图像 ----------------------
        img_path = os.path.join(self.images_dir, img_name)  # 图像完整路径
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"无法读取图像：{img_path}（路径是否正确？）")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
        h, w = image.shape[:2]  # 图像高度和宽度

        # ---------------------- 定义 boxes ----------------------
        # 从标注文件中解析边界框（解决：未解析的引用 'boxes'）
        boxes = []  # 存储格式：[ [xmin, ymin, xmax, ymax, 类别ID], ... ]
        # 生成标注文件路径（假设标注文件与图像同名，后缀为.txt）
        label_name = os.path.splitext(img_name)[0] + ".txt"  # 例如："P0001.txt"
        label_path = os.path.join(self.labels_dir, label_name)

        # 读取标注文件并解析
        if os.path.exists(label_path):
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue  # 跳过空行
                    # DOTA标注格式：8个坐标（x1,y1,x2,y2,x3,y3,x4,y4）+ 类别 + difficult
                    parts = line.split()
                    if len(parts) != 10:
                        continue  # 跳过格式错误的行
                    # 解析坐标和类别
                    x_coords = list(map(float, parts[:8:2]))  # x1, x2, x3, x4
                    y_coords = list(map(float, parts[1:8:2])) # y1, y2, y3, y4
                    cls_name = parts[8]  # 类别名称（如"plane"）
                    difficult = int(parts[9])  # 是否为困难样本

                    # 过滤条件（根据需求调整）
                    if self.is_train and difficult == 1:
                        continue  # 训练集跳过困难样本
                    if cls_name not in CLASS2ID:
                        continue  # 跳过未定义的类别

                    # 转换为水平边界框（xmin, ymin, xmax, ymax）
                    xmin, xmax = min(x_coords), max(x_coords)
                    ymin, ymax = min(y_coords), max(y_coords)

                    # 归一化到[0,1]（相对于图像宽高）
                    xmin /= w
                    ymin /= h
                    xmax /= w
                    ymax /= h

                    # 添加到boxes列表（包含类别ID）
                    boxes.append([xmin, ymin, xmax, ymax, CLASS2ID[cls_name]])

        # ---------------------- 后续处理（数据增强等） ----------------------
        # 拆分边界框坐标和标签（用于数据增强）
        bboxes = [box[:4] for box in boxes]  # 坐标
        labels = [box[4] for box in boxes]   # 类别ID

        # 应用数据增强
        if self.transform:
            transformed = self.transform(image=image, bboxes=bboxes, labels=labels)
            image = transformed['image']
            bboxes = transformed['bboxes']
            labels = transformed['labels']

        # 转换为Tensor
        return {
            'image': image,
            'bboxes': torch.tensor(bboxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.long),
            'image_name': img_name  # 这里使用之前定义的img_name
        }
def get_transform(is_train=True):
    """数据增强：确保输出尺寸为640×640"""
    transforms = []
    if is_train:
        transforms.extend([
            # 修正1：scale范围改为(0.8, 1.0)（在[0,1]内）
            # 修正2：用size=(IMG_SIZE, IMG_SIZE)替代height和width
            A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ])
    else:
        transforms.append(A.Resize(height=IMG_SIZE, width=IMG_SIZE))

    # 标准化和转Tensor
    transforms.extend([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet均值
        ToTensorV2()
    ])

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format='pascal_voc',  # 边界框格式
            label_fields=['labels'],  # 明确标签字段名为'labels'，与传递的参数匹配
            min_visibility=0.3
        )
    )
def get_dataloader(images_dir, labels_dir, batch_size=8, is_train=True, num_workers=4):
    dataset = DOTADataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        transform=get_transform(is_train),
        is_train=is_train
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        collate_fn=collate_fn  # 这里用定义的函数替代lambda
    )