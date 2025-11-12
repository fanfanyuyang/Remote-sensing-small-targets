import argparse
import torch
from data_processor import get_dataloader, DOTA_CLASSES
from trainer import Trainer
from visualizer import Visualizer
from drone_swin import DroneSwinDetector

def parse_args():
    parser = argparse.ArgumentParser(description='Train DroneSwinDetector on DOTAv1.5')
    parser.add_argument('--train_images', type=str, required=True, help='Train images directory')
    parser.add_argument('--train_labels', type=str, required=True, help='Train labels directory')
    parser.add_argument('--val_images', type=str, required=True, help='Validation images directory')
    parser.add_argument('--val_labels', type=str, required=True, help='Validation labels directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--save_dir', type=str, default='output', help='Save directory')
    parser.add_argument('--num_workers', type=int, default=4, help='Data loader workers')
    parser.add_argument('--window_size', type=int, default=8, help='Swin window size')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 加载数据
    train_loader = get_dataloader(
        images_dir=args.train_images,
        labels_dir=args.train_labels,
        batch_size=args.batch_size,
        is_train=True,
        num_workers=args.num_workers
    )
    val_loader = get_dataloader(
        images_dir=args.val_images,
        labels_dir=args.val_labels,
        batch_size=args.batch_size,
        is_train=False,
        num_workers=args.num_workers
    )

    # 2. 初始化模型
    model = DroneSwinDetector(
        num_classes=len(DOTA_CLASSES),
        window_size=args.window_size,
        img_size=640
    )

    # 3. 训练模型
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        args=args
    )
    trainer.train()

    # 4. 可视化结果
    visualizer = Visualizer(save_dir=args.save_dir)
    visualizer.plot_loss_curves(trainer.train_losses, trainer.val_losses)
    visualizer.visualize_predictions(model, val_loader, device, conf_threshold=0.3)


if __name__ == '__main__':
    main()