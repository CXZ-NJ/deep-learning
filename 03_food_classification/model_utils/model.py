"""自定义 CNN 和 torchvision 迁移学习模型。"""

from typing import Tuple

import torch.nn as nn
from torchvision import models


class SimpleCNN(nn.Module):
    """适合第一次学习卷积网络的简洁模型。"""

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


def _freeze_parameters(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False


def initialize_model(
    model_name: str,
    num_classes: int,
    linear_probing: bool = False,
    use_pretrained: bool = False,
) -> Tuple[nn.Module, int]:
    """
    根据名称建立模型，并把最后的分类层改为 ``num_classes`` 个输出。

    linear_probing=True 时冻结原模型，只训练新分类头。
    use_pretrained=True 首次运行时可能需要联网下载官方权重。
    """
    model_name = model_name.lower()
    input_size = 224

    if linear_probing and not use_pretrained:
        raise ValueError("linear_probing 需要和 use_pretrained=True 一起使用")

    if model_name in {"simple_cnn", "mymodel"}:
        return SimpleCNN(num_classes), input_size

    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if use_pretrained else None
        model = models.resnet18(weights=weights)
        if linear_probing:
            _freeze_parameters(model)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model, input_size

    if model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if use_pretrained else None
        model = models.resnet50(weights=weights)
        if linear_probing:
            _freeze_parameters(model)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model, input_size

    if model_name in {"vgg", "vgg11_bn"}:
        weights = models.VGG11_BN_Weights.DEFAULT if use_pretrained else None
        model = models.vgg11_bn(weights=weights)
        if linear_probing:
            _freeze_parameters(model)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model, input_size

    raise ValueError(
        "不支持的模型名称。可选：simple_cnn、resnet18、resnet50、vgg11_bn"
    )


# 兼容博主原代码中的函数参数拼写。
def set_parameter_requires_grad(model: nn.Module, linear_probing: bool) -> None:
    if linear_probing:
        _freeze_parameters(model)
