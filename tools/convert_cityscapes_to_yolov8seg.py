"""
Cityscapes -> YOLOv8-seg polygon .txt converter
Author: ChatGPT
Usage:
    python convert_cityscapes_to_yolov8seg.py
"""

import os
import json
import cv2
import numpy as np
from tqdm import tqdm
import shutil

# =================== 配置区 ===================
CITY_ROOT = r"E:\datasets\cityscapes"      # Cityscapes 原始根目录
OUT_ROOT  = r"E:\datasets\cityscapes_yolo" # YOLOv8 输出目录

# Cityscapes 19 类
CITYSCAPES_NAMES = [
    "road","sidewalk","building","wall","fence","pole","traffic_light","traffic_sign",
    "vegetation","terrain","sky","person","rider","car","truck","bus","train","motorcycle","bicycle"
]
NUM_CLASSES = len(CITYSCAPES_NAMES)
# ==============================================

def ensure_dirs():
    for split in ["train","val"]:
        os.makedirs(os.path.join(OUT_ROOT,"images",split), exist_ok=True)
        os.makedirs(os.path.join(OUT_ROOT,"labels",split), exist_ok=True)

def polygon_to_yolo_line(cls, pts, w, h):
    coords = []
    for (x,y) in pts:
        coords.append(x / w)
        coords.append(y / h)
    if len(coords) < 6:
        return None
    vals = " ".join([f"{v:.6f}" for v in coords])
    return f"{cls} {vals}"

def mask_to_polygons_lines(mask, cls_id, w, h):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 10:
            continue
        eps = 0.001 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)
        pts = [(int(p[0][0]), int(p[0][1])) for p in approx]
        if len(pts) < 3:
            continue
        line = polygon_to_yolo_line(cls_id, pts, w, h)
        if line:
            lines.append(line)
    return lines

def process_split(split):
    img_dir = os.path.join(CITY_ROOT, "leftImg8bit", split)
    gt_dir  = os.path.join(CITY_ROOT, "gtFine", split)
    out_img_dir   = os.path.join(OUT_ROOT,"images",split)
    out_label_dir = os.path.join(OUT_ROOT,"labels",split)

    for city in os.listdir(img_dir):
        city_img_dir = os.path.join(img_dir, city)
        city_gt_dir  = os.path.join(gt_dir, city)

        if not os.path.isdir(city_img_dir) or len(os.listdir(city_img_dir))==0:
            continue

        for fname in os.listdir(city_img_dir):
            if not fname.endswith("_leftImg8bit.png"):
                continue
            src_img_path = os.path.join(city_img_dir, fname)
            dst_img_path = os.path.join(out_img_dir, fname)

            img = cv2.imread(src_img_path)
            if img is None:
                print(f"WARN: cannot read image {src_img_path}, skipped")
                continue
            cv2.imwrite(dst_img_path, img)

            base = fname.replace("_leftImg8bit.png","")
            label_ids_path = os.path.join(city_gt_dir, base+"_gtFine_labelIds.png")
            labels_out_path = os.path.join(out_label_dir, fname.replace(".png",".txt"))

            lines = []
            if os.path.exists(label_ids_path):
                mask_img = cv2.imread(label_ids_path, cv2.IMREAD_UNCHANGED)
                h,w = mask_img.shape[:2]
                for cls_id in range(NUM_CLASSES):
                    mask = (mask_img==cls_id).astype(np.uint8)*255
                    lines_cls = mask_to_polygons_lines(mask, cls_id, w, h)
                    lines.extend(lines_cls)
            else:
                print(f"WARN: label not found for {fname}")

            with open(labels_out_path,"w",encoding="utf-8") as fo:
                if lines:
                    fo.write("\n".join(lines))
                else:
                    fo.write("")  # 空标签允许

def main():
    ensure_dirs()
    for split in ["train","val"]:
        print(f"Processing {split} ...")
        process_split(split)
    print("Done! YOLOv8-seg dataset saved to:", OUT_ROOT)

if __name__ == "__main__":
    main()
