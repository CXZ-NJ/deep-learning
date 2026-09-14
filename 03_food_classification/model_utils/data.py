"""Food-11 数据读取、图像增强和伪标签数据集。"""

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import autoaugment


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
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(input_size: int = 224, normalize: bool = True):
    """分别建立训练阶段和验证/测试阶段的图片变换。"""
    train_steps = [
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        autoaugment.AutoAugment(policy=autoaugment.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
    ]
    eval_steps = [
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ]

    if normalize:
        normalizer = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        train_steps.append(normalizer)
        eval_steps.append(normalizer)

    return transforms.Compose(train_steps), transforms.Compose(eval_steps)


def _image_files(directory: Path) -> List[Path]:
    """递归取得目录中的图片，排序后保证每次读取顺序一致。"""
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


class Food11Dataset(Dataset):
    """
    Food-11 数据集。

    train/val 返回 ``(图片张量, 类别标签)``；
    train_unl/test 返回 ``(图片张量, 样本下标)``，下标用于恢复原图片路径。
    """

    SPLIT_PATHS = {
        "train": Path("training") / "labeled",
        "train_unl": Path("training") / "unlabeled",
        "val": Path("validation"),
        "test": Path("testing"),
    }

    def __init__(
        self,
        root: str,
        mode: str,
        input_size: int = 224,
        normalize: bool = True,
    ):
        if mode not in self.SPLIT_PATHS:
            raise ValueError("mode 必须是 train、train_unl、val 或 test")

        self.root = Path(root).expanduser().resolve()
        self.mode = mode
        self.split_dir = self.root / self.SPLIT_PATHS[mode]

        if not self.split_dir.is_dir():
            raise FileNotFoundError(
                f"找不到数据目录：{self.split_dir}\n"
                "请按照 data/README.md 中的结构放置 Food-11 数据集。"
            )

        train_transform, eval_transform = build_transforms(input_size, normalize)
        self.transform = train_transform if mode in {"train", "train_unl"} else eval_transform

        self.image_paths: List[Path] = []
        self.labels: Optional[List[int]] = None

        if mode in {"train", "val"}:
            self.image_paths, self.labels = self._collect_labeled_images()
        else:
            self.image_paths = _image_files(self.split_dir)

        if not self.image_paths:
            raise RuntimeError(f"目录中没有找到图片：{self.split_dir}")

        print(f"{mode}：读取到 {len(self.image_paths)} 张图片")

    def _collect_labeled_images(self) -> Tuple[List[Path], List[int]]:
        image_paths: List[Path] = []
        labels: List[int] = []

        expected_directories = {f"{index:02d}" for index in range(len(CLASS_NAMES))}
        existing_directories = {
            path.name for path in self.split_dir.iterdir() if path.is_dir()
        }

        ignored = sorted(existing_directories - expected_directories)
        if ignored:
            print(
                "提示：以下目录不属于 Food-11 的 00–10 类，已忽略："
                + ", ".join(ignored)
            )

        for label in range(len(CLASS_NAMES)):
            class_dir = self.split_dir / f"{label:02d}"
            if not class_dir.is_dir():
                raise FileNotFoundError(f"缺少类别目录：{class_dir}")

            class_images = _image_files(class_dir)
            image_paths.extend(class_images)
            labels.extend([label] * len(class_images))

        return image_paths, labels

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_tensor = self.transform(image)

        if self.labels is None:
            return image_tensor, index

        return image_tensor, torch.tensor(self.labels[index], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.image_paths)


class PseudoLabelDataset(Dataset):
    """保存通过高置信度预测筛选出的无标签图片。"""

    def __init__(
        self,
        image_paths: Sequence[Path],
        labels: Sequence[int],
        transform,
    ):
        self.image_paths = list(image_paths)
        self.labels = list(labels)
        self.transform = transform

    def __getitem__(self, index: int):
        with Image.open(self.image_paths[index]) as image:
            image = image.convert("RGB")
            image_tensor = self.transform(image)
        return image_tensor, torch.tensor(self.labels[index], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.image_paths)


def get_data_loader(
    root: str,
    mode: str,
    batch_size: int,
    input_size: int = 224,
    normalize: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """建立指定数据分区的 DataLoader。Windows 初学环境默认不开子进程。"""
    dataset = Food11Dataset(root, mode, input_size=input_size, normalize=normalize)
    shuffle = mode == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def get_pseudo_label_loader(
    unlabeled_loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    threshold: float = 0.99,
) -> Optional[DataLoader]:
    """把置信度达到阈值的无标签图片组成新的训练 DataLoader。"""
    model.eval()
    selected_paths: List[Path] = []
    selected_labels: List[int] = []

    with torch.no_grad():
        for images, indices in unlabeled_loader:
            images = images.to(device)
            probabilities = torch.softmax(model(images), dim=1)
            confidence, predicted_labels = probabilities.max(dim=1)

            keep = confidence >= threshold
            kept_indices = indices[keep.cpu()].tolist()
            kept_labels = predicted_labels[keep].cpu().tolist()

            for sample_index, label in zip(kept_indices, kept_labels):
                selected_paths.append(unlabeled_loader.dataset.image_paths[sample_index])
                selected_labels.append(label)

    if not selected_paths:
        print(f"没有置信度达到 {threshold:.2f} 的无标签图片，本轮不启用伪标签。")
        return None

    print(f"伪标签筛选出 {len(selected_paths)} 张图片。")
    dataset = PseudoLabelDataset(
        selected_paths,
        selected_labels,
        transform=unlabeled_loader.dataset.transform,
    )
    return DataLoader(
        dataset,
        batch_size=unlabeled_loader.batch_size,
        shuffle=True,
        num_workers=unlabeled_loader.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


# 兼容博主原代码中的驼峰命名，建议新代码使用 get_data_loader。
getDataLoader = get_data_loader
