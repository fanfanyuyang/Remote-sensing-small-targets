import os
import json
from tqdm import tqdm
from PIL import Image

def convert_coco_to_yolo_seg(coco_json, images_dir, labels_dir):
    os.makedirs(labels_dir, exist_ok=True)

    with open(coco_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_id_to_filename = {img["id"]: img for img in data["images"]}

    print("Converting:", coco_json)

    for ann in tqdm(data["annotations"]):
        img_id = ann["image_id"]
        img_info = img_id_to_filename[img_id]

        img_w = img_info["width"]
        img_h = img_info["height"]
        file_name = img_info["file_name"]

        label_path = os.path.join(labels_dir, file_name.replace('.jpg', '.txt'))

        seg = ann.get("segmentation", [])
        if not seg or type(seg) != list:
            continue

        cls = ann["category_id"] - 1  # COCO 类别从 1 开始，YOLO 从 0 开始

        for polygon in seg:
            if len(polygon) < 6:
                continue

            xy = []
            for i in range(0, len(polygon), 2):
                x = polygon[i] / img_w
                y = polygon[i + 1] / img_h
                xy.append(f"{x:.6f} {y:.6f}")

            line = f"{cls} " + " ".join(xy) + "\n"
            with open(label_path, "a") as f:
                f.write(line)

    print("Done:", labels_dir)


if __name__ == "__main__":
    root = r"E:\coco2017"

    convert_coco_to_yolo_seg(
        coco_json=fr"{root}\annotations\instances_train2017.json",
        images_dir=fr"{root}\train2017",
        labels_dir=fr"{root}\labels\train2017"
    )

    convert_coco_to_yolo_seg(
        coco_json=fr"{root}\annotations\instances_val2017.json",
        images_dir=fr"{root}\val2017",
        labels_dir=fr"{root}\labels\val2017"
    )
