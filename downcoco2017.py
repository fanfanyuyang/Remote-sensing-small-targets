import os
import urllib.request
import zipfile

# 目标下载目录
base_dir = r"E:\coco2017"
os.makedirs(base_dir, exist_ok=True)

# COCO 2017 数据集下载链接
urls = {
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
}


# 下载函数
def download_file(url, save_path):
    if os.path.exists(save_path):
        print(f"{save_path} 已存在，跳过下载")
        return
    print(f"开始下载 {url} ...")
    urllib.request.urlretrieve(url, save_path)
    print(f"下载完成: {save_path}")


# 解压函数
def unzip_file(zip_path, extract_to):
    print(f"解压 {zip_path} 到 {extract_to} ...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"解压完成: {zip_path}")


# 下载并解压
for name, url in urls.items():
    zip_path = os.path.join(base_dir, f"{name}.zip")
    download_file(url, zip_path)

    # train/val 图片解压到 images 文件夹
    if name in ["train2017", "val2017"]:
        extract_to = os.path.join(base_dir, "images")
    else:  # annotations
        extract_to = os.path.join(base_dir, "annotations")

    os.makedirs(extract_to, exist_ok=True)
    unzip_file(zip_path, extract_to)

    # 可选：下载后删除 zip 文件以节省空间
    # os.remove(zip_path)
