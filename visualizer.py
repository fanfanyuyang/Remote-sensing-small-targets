import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from data_processor import ID2CLASS, IMG_SIZE

plt.style.use('seaborn')

class Visualizer:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(os.path.join(save_dir, 'visualizations'), exist_ok=True)

    def plot_loss_curves(self, train_losses, val_losses):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
        plt.plot(range(1, len(val_losses)+1), val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Curves')
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, 'loss_curves.png'))
        plt.close()

    def postprocess_preds(self, cls_logits, bbox_preds, conf_threshold=0.5):
        """后处理预测结果：过滤低置信度，转换为绝对坐标"""
        cls_probs = F.sigmoid(cls_logits)  # (1, 15, 40, 40)
        b, c, h, w = cls_probs.shape
        grid_size = h

        # 提取高置信度预测
        preds = []
        for cls in range(c):
            mask = cls_probs[0, cls] > conf_threshold
            if not mask.any():
                continue
            ys, xs = torch.where(mask)
            probs = cls_probs[0, cls, ys, xs]

            # 转换边界框为绝对坐标
            for y, x, prob in zip(ys, xs, probs):
                cx = bbox_preds[0, 0, y, x] * IMG_SIZE
                cy = bbox_preds[0, 1, y, x] * IMG_SIZE
                gw = bbox_preds[0, 2, y, x] * IMG_SIZE
                gh = bbox_preds[0, 3, y, x] * IMG_SIZE

                xmin = cx - gw / 2
                ymin = cy - gh / 2
                xmax = cx + gw / 2
                ymax = cy + gh / 2

                preds.append({
                    'bbox': (xmin, ymin, xmax, ymax),
                    'cls': cls,
                    'conf': prob.item()
                })

        return preds

    def visualize_predictions(self, model, val_loader, device, num_samples=5, conf_threshold=0.5):
        model.eval()
        samples_plotted = 0
        with torch.no_grad():
            for batch in val_loader:
                for data in batch:
                    if samples_plotted >= num_samples:
                        return

                    # 准备输入
                    image = data['image'].to(device).unsqueeze(0)
                    true_bboxes = data['bboxes'].numpy()
                    true_labels = data['labels'].numpy()
                    image_name = data['image_name']

                    # 模型预测
                    cls_logits, bbox_preds = model(image)
                    preds = self.postprocess_preds(cls_logits, bbox_preds, conf_threshold)

                    # 还原图像（反归一化）
                    img_np = image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                    img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
                    img_np = img_np.astype(np.uint8)

                    # 绘制真实框（绿色）
                    plt.figure(figsize=(10, 10))
                    plt.imshow(img_np)
                    for bbox, label in zip(true_bboxes, true_labels):
                        xmin, ymin, xmax, ymax = bbox * IMG_SIZE  # 还原为绝对坐标
                        plt.gca().add_patch(plt.Rectangle(
                            (xmin, ymin), xmax-xmin, ymax-ymin,
                            fill=False, color='green', linewidth=2, label='Ground Truth'
                        ))
                        plt.text(xmin, ymin, ID2CLASS[label], color='green', fontsize=8)

                    # 绘制预测框（红色）
                    for pred in preds:
                        xmin, ymin, xmax, ymax = pred['bbox']
                        cls = pred['cls']
                        conf = pred['conf']
                        plt.gca().add_patch(plt.Rectangle(
                            (xmin, ymin), xmax-xmin, ymax-ymin,
                            fill=False, color='red', linewidth=2, label='Prediction'
                        ))
                        plt.text(xmin, ymin+20, f"{ID2CLASS[cls]} {conf:.2f}", color='red', fontsize=8)

                    plt.title(f"Prediction: {image_name}")
                    plt.axis('off')
                    plt.savefig(os.path.join(self.save_dir, 'visualizations', f'{image_name}_pred.png'))
                    plt.close()

                    samples_plotted += 1