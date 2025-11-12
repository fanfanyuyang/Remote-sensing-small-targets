import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
from tqdm import tqdm
import numpy as np

# Focal Loss：解决类别不平衡
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: (b, num_classes, h, w), targets: (b, n, h, w) 独热编码
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class Trainer:
    def __init__(self, model, train_loader, val_loader, device, args):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.args = args
        self.num_classes = model.num_classes

        # 优化器和调度器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=args.epochs,
            eta_min=1e-6
        )

        # 损失函数
        self.cls_criterion = FocalLoss()
        self.bbox_criterion = nn.SmoothL1Loss(reduction='mean')

        # 训练记录
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

        os.makedirs(args.save_dir, exist_ok=True)

    def prepare_targets(self, bboxes, labels, grid_size):
        """将标注转换为网格目标（与模型输出尺寸匹配）"""
        b, n = bboxes.shape[0], bboxes.shape[1]
        cls_targets = torch.zeros((b, self.num_classes, grid_size, grid_size), device=self.device)
        bbox_targets = torch.zeros((b, 4, grid_size, grid_size), device=self.device)
        mask = torch.zeros((b, 1, grid_size, grid_size), device=self.device)  # 标记有目标的网格

        for i in range(b):
            for j in range(n):
                if labels[i, j] == -1:  # 填充的无效框
                    continue
                # 归一化坐标 → 网格坐标
                xmin, ymin, xmax, ymax = bboxes[i, j]
                cx = (xmin + xmax) / 2  # 中心x
                cy = (ymin + ymax) / 2  # 中心y
                gw = xmax - xmin        # 宽度
                gh = ymax - ymin        # 高度

                # 映射到网格
                grid_x = int(cx * grid_size)
                grid_y = int(cy * grid_size)
                grid_x = torch.clamp(torch.tensor(grid_x), 0, grid_size-1).item()
                grid_y = torch.clamp(torch.tensor(grid_y), 0, grid_size-1).item()

                # 填充目标
                cls_targets[i, labels[i, j], grid_y, grid_x] = 1.0
                bbox_targets[i, 0, grid_y, grid_x] = cx  # 中心x（归一化）
                bbox_targets[i, 1, grid_y, grid_x] = cy  # 中心y（归一化）
                bbox_targets[i, 2, grid_y, grid_x] = gw  # 宽度（归一化）
                bbox_targets[i, 3, grid_y, grid_x] = gh  # 高度（归一化）
                mask[i, 0, grid_y, grid_x] = 1.0  # 标记有效网格

        return cls_targets, bbox_targets, mask

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch}")

        for batch in pbar:
            self.optimizer.zero_grad()
            batch_loss = 0.0
            batch_size = len(batch)

            # 统一处理批次数据
            images = torch.stack([data['image'] for data in batch]).to(self.device)
            max_boxes = max(len(data['bboxes']) for data in batch)

            # 填充边界框和标签到相同长度
            bboxes = torch.zeros((batch_size, max_boxes, 4), device=self.device)
            labels = torch.full((batch_size, max_boxes), -1, device=self.device, dtype=torch.long)
            for i, data in enumerate(batch):
                n = len(data['bboxes'])
                bboxes[i, :n] = data['bboxes'].to(self.device)
                labels[i, :n] = data['labels'].to(self.device)

            # 模型输出
            cls_logits, bbox_preds = self.model(images)  # (b, 15, 40, 40), (b, 4, 40, 40)
            grid_size = cls_logits.shape[2]

            # 准备目标
            cls_targets, bbox_targets, mask = self.prepare_targets(bboxes, labels, grid_size)

            # 计算损失（只对有目标的网格计算边界框损失）
            cls_loss = self.cls_criterion(cls_logits, cls_targets)
            bbox_loss = self.bbox_criterion(bbox_preds * mask, bbox_targets * mask)
            loss = cls_loss + 5.0 * bbox_loss  # 调整边界框损失权重

            # 反向传播
            loss.backward()
            self.optimizer.step()

            batch_loss = loss.item()
            total_loss += batch_loss
            pbar.set_postfix(loss=batch_loss)

        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Val Epoch {epoch}")
            for batch in pbar:
                batch_size = len(batch)
                images = torch.stack([data['image'] for data in batch]).to(self.device)
                max_boxes = max(len(data['bboxes']) for data in batch)

                bboxes = torch.zeros((batch_size, max_boxes, 4), device=self.device)
                labels = torch.full((batch_size, max_boxes), -1, device=self.device, dtype=torch.long)
                for i, data in enumerate(batch):
                    n = len(data['bboxes'])
                    bboxes[i, :n] = data['bboxes'].to(self.device)
                    labels[i, :n] = data['labels'].to(self.device)

                cls_logits, bbox_preds = self.model(images)
                grid_size = cls_logits.shape[2]
                cls_targets, bbox_targets, mask = self.prepare_targets(bboxes, labels, grid_size)

                cls_loss = self.cls_criterion(cls_logits, cls_targets)
                bbox_loss = self.bbox_criterion(bbox_preds * mask, bbox_targets * mask)
                loss = cls_loss + 5.0 * bbox_loss

                total_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(self.val_loader)
        self.val_losses.append(avg_loss)
        return avg_loss

    def train(self):
        print(f"Starting training on {self.device}...")
        for epoch in range(1, self.args.epochs + 1):
            start_time = time.time()

            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate(epoch)
            self.scheduler.step()

            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': val_loss
                }, os.path.join(self.args.save_dir, 'best_model.pth'))

            print(f"Epoch {epoch}/{self.args.epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Time: {time.time() - start_time:.2f}s\n")

        torch.save(self.model.state_dict(), os.path.join(self.args.save_dir, 'final_model.pth'))
        print("Training completed!")