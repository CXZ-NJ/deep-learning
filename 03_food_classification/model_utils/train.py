"""Food-11 的训练、验证、保存最佳模型和绘图逻辑。"""

import time
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import get_pseudo_label_loader


def _run_training_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
):
    total_loss = 0.0
    correct = 0
    total = 0

    model.train()
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


def _run_validation_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
):
    total_loss = 0.0
    correct = 0
    total = 0

    model.eval()
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


def _save_curves(history: Dict[str, list], output_path: Path) -> None:
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


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    epochs: int,
    checkpoint_path: str,
    curve_path: str,
    model_name: str,
    unlabeled_loader: Optional[DataLoader] = None,
    use_semi_supervised: bool = False,
    semi_start_accuracy: float = 0.70,
    pseudo_label_threshold: float = 0.99,
    pseudo_label_interval: int = 3,
) -> Dict[str, list]:
    """训练模型；按照验证准确率保存最佳 checkpoint。"""
    model = model.to(device)
    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    best_val_accuracy = 0.0
    pseudo_loader = None

    for epoch_index in range(epochs):
        start_time = time.time()

        train_loss, train_accuracy = _run_training_loader(
            model, train_loader, optimizer, criterion, device
        )

        if pseudo_loader is not None:
            pseudo_loss, pseudo_accuracy = _run_training_loader(
                model, pseudo_loader, optimizer, criterion, device
            )
            print(
                f"伪标签训练：loss={pseudo_loss:.6f}, "
                f"accuracy={pseudo_accuracy:.4f}"
            )

        val_loss, val_accuracy = _run_validation_loader(
            model, val_loader, criterion, device
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(
                {
                    "epoch": epoch_index + 1,
                    "model_name": model_name,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_accuracy": best_val_accuracy,
                },
                checkpoint,
            )

        print(
            f"[{epoch_index + 1:03d}/{epochs:03d}] "
            f"{time.time() - start_time:.2f}s | "
            f"train_loss={train_loss:.6f}, train_acc={train_accuracy:.4f} | "
            f"val_loss={val_loss:.6f}, val_acc={val_accuracy:.4f} | "
            f"best={best_val_accuracy:.4f}"
        )

        should_refresh_pseudo_labels = (
            use_semi_supervised
            and unlabeled_loader is not None
            and val_accuracy >= semi_start_accuracy
            and (epoch_index + 1) % pseudo_label_interval == 0
        )
        if should_refresh_pseudo_labels:
            pseudo_loader = get_pseudo_label_loader(
                unlabeled_loader,
                model,
                device,
                threshold=pseudo_label_threshold,
            )

        _save_curves(history, Path(curve_path))

    print(f"训练完成，最佳验证准确率：{best_val_accuracy:.4f}")
    print(f"最佳模型保存在：{checkpoint}")
    print(f"训练曲线保存在：{curve_path}")
    return history


def train_val(parameters: dict):
    """兼容原项目 ``train_val(trainpara)`` 的调用形式。"""
    return train_model(
        model=parameters["model"],
        train_loader=parameters["train_loader"],
        val_loader=parameters["val_loader"],
        optimizer=parameters["optimizer"],
        criterion=parameters.get("criterion", parameters.get("loss")),
        device=parameters["device"],
        epochs=parameters.get("epochs", parameters.get("epoch")),
        checkpoint_path=parameters.get("checkpoint_path", parameters.get("save_path")),
        curve_path=parameters.get("curve_path", "assets/training_curves.png"),
        model_name=parameters.get("model_name", "unknown"),
        unlabeled_loader=parameters.get(
            "unlabeled_loader", parameters.get("no_label_Loader")
        ),
        use_semi_supervised=parameters.get(
            "use_semi_supervised", parameters.get("do_semi", False)
        ),
        semi_start_accuracy=parameters.get(
            "semi_start_accuracy", parameters.get("acc_thres", 0.70)
        ),
        pseudo_label_threshold=parameters.get(
            "pseudo_label_threshold", parameters.get("conf_thres", 0.99)
        ),
        pseudo_label_interval=parameters.get("pseudo_label_interval", 3),
    )
