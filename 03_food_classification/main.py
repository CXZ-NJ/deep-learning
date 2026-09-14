"""Food-11 模块化训练入口。先读 simple_class.py，再学习本文件。"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from model_utils.data import CLASS_NAMES, get_data_loader
from model_utils.model import initialize_model
from model_utils.train import train_model


PROJECT_DIR = Path(__file__).resolve().parent


def seed_everything(seed: int) -> None:
    """固定随机种子，让多次实验尽可能得到相近结果。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="训练 Food-11 图片分类模型")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_DIR / "data" / "food-11",
        help="Food-11 数据根目录",
    )
    parser.add_argument(
        "--model",
        choices=["simple_cnn", "resnet18", "resnet50", "vgg11_bn"],
        default="simple_cnn",
        help="要训练的模型",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="使用 torchvision 官方预训练权重（首次使用可能需要联网）",
    )
    parser.add_argument(
        "--linear-probing",
        action="store_true",
        help="冻结预训练模型，只训练新的分类头",
    )
    parser.add_argument(
        "--semi",
        action="store_true",
        help="达到准确率阈值后使用无标签数据生成伪标签",
    )
    parser.add_argument("--semi-start-accuracy", type=float, default=0.70)
    parser.add_argument("--pseudo-label-threshold", type=float, default=0.99)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("类别数:", len(CLASS_NAMES))

    model, input_size = initialize_model(
        args.model,
        len(CLASS_NAMES),
        linear_probing=args.linear_probing,
        use_pretrained=args.pretrained,
    )

    train_loader = get_data_loader(
        str(args.data_root),
        "train",
        args.batch_size,
        input_size=input_size,
        normalize=True,
        num_workers=args.num_workers,
    )
    val_loader = get_data_loader(
        str(args.data_root),
        "val",
        args.batch_size,
        input_size=input_size,
        normalize=True,
        num_workers=args.num_workers,
    )

    unlabeled_loader = None
    if args.semi:
        unlabeled_loader = get_data_loader(
            str(args.data_root),
            "train_unl",
            args.batch_size,
            input_size=input_size,
            normalize=True,
            num_workers=args.num_workers,
        )

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss()

    checkpoint_path = PROJECT_DIR / "checkpoints" / f"best_{args.model}.pth"
    curve_path = PROJECT_DIR / "assets" / "training_curves.png"

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=args.epochs,
        checkpoint_path=str(checkpoint_path),
        curve_path=str(curve_path),
        model_name=args.model,
        unlabeled_loader=unlabeled_loader,
        use_semi_supervised=args.semi,
        semi_start_accuracy=args.semi_start_accuracy,
        pseudo_label_threshold=args.pseudo_label_threshold,
    )


if __name__ == "__main__":
    main()
