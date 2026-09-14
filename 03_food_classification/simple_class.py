"""
Food-11 图片分类的单文件入门版。

这个文件故意把数据、模型、训练写在一起，方便第一次学习时从上往下阅读。
理解后，再阅读 main.py 和 model_utils/ 中的模块化版本。
"""

import argparse
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


PROJECT_DIR = Path(__file__).resolve().parent
CLASS_NAMES = (
    "Bread",
    "Dairy product",
    "Dessert",
    "Egg",
    "Fried food",
    "Meat",
    "Noodles/Pasta",
    "Rice",
    "Seafood",
    "Soup",
    "Vegetable/Fruit",
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ]
)


class FoodDataset(Dataset):
    """读取有标签的训练集或验证集。"""

    def __init__(self, data_root: Path, mode: str):
        if mode not in {"train", "val"}:
            raise ValueError("mode 必须是 train 或 val")

        self.mode = mode
        self.transform = train_transform if mode == "train" else val_transform
        split = Path("training") / "labeled" if mode == "train" else Path("validation")
        self.split_dir = data_root / split

        if not self.split_dir.is_dir():
            raise FileNotFoundError(
                f"找不到数据目录：{self.split_dir}\n"
                "请先阅读 data/README.md 并放置数据集。"
            )

        self.image_paths = []
        self.labels = []

        expected = {f"{index:02d}" for index in range(len(CLASS_NAMES))}
        existing = {path.name for path in self.split_dir.iterdir() if path.is_dir()}
        ignored = sorted(existing - expected)
        if ignored:
            print("以下非 Food-11 类别目录已忽略：" + ", ".join(ignored))

        for label in range(len(CLASS_NAMES)):
            class_dir = self.split_dir / f"{label:02d}"
            if not class_dir.is_dir():
                raise FileNotFoundError(f"缺少类别目录：{class_dir}")

            class_images = sorted(
                path
                for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            self.image_paths.extend(class_images)
            self.labels.extend([label] * len(class_images))

        if not self.image_paths:
            raise RuntimeError(f"没有找到图片：{self.split_dir}")

        print(f"{mode}：读取到 {len(self.image_paths)} 张图片")

    def __getitem__(self, index: int):
        with Image.open(self.image_paths[index]) as image:
            image = image.convert("RGB")
            image_tensor = self.transform(image)

        label = torch.tensor(self.labels[index], dtype=torch.long)
        return image_tensor, label

    def __len__(self) -> int:
        return len(self.image_paths)


class SimpleCNN(nn.Module):
    """四个卷积阶段组成的入门 CNN。"""

    def __init__(self, num_classes: int = 11):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def save_curves(history, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description="Food-11 单文件入门训练")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_DIR / "data" / "food-11",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(0)

    train_set = FoodDataset(args.data_root, "train")
    val_set = FoodDataset(args.data_root, "val")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    best_val_accuracy = 0.0
    checkpoint_path = PROJECT_DIR / "checkpoints" / "best_simple_cnn.pth"
    curve_path = PROJECT_DIR / "assets" / "simple_training_curves.png"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch_index in range(args.epochs):
        start_time = time.time()
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(
                {
                    "epoch": epoch_index + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_accuracy": best_val_accuracy,
                },
                checkpoint_path,
            )

        save_curves(history, curve_path)
        print(
            f"[{epoch_index + 1:03d}/{args.epochs:03d}] "
            f"{time.time() - start_time:.2f}s | "
            f"train_loss={train_loss:.6f}, train_acc={train_accuracy:.4f} | "
            f"val_loss={val_loss:.6f}, val_acc={val_accuracy:.4f}"
        )

    print(f"训练完成，最佳验证准确率：{best_val_accuracy:.4f}")
    print(f"最佳模型保存在：{checkpoint_path}")
    print(f"训练曲线保存在：{curve_path}")


if __name__ == "__main__":
    main()
