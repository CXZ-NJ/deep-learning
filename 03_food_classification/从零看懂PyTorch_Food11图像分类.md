# 从零看懂 PyTorch：Food-11 图像分类

> 本项目是在前两个回归项目基础上的进一步学习：  
> 从“数值回归”进入“图像分类”，开始接触 **图像 Tensor、数据增强、卷积神经网络 CNN、Batch Normalization、交叉熵损失、分类准确率、半监督学习、伪标签、经典 CNN 网络与迁移学习**。
>
> 本文按照项目代码的学习顺序组织：
>
> 1. **先完整介绍单文件版 `simple_class.py`**，把整个图像分类流程串起来；
> 2. **再介绍多文件版**，只讲相对于单文件版新增的知识，不重复已经讲过的内容。

---

## 目录

- [1. 项目任务：Food-11 图像分类](#sec-1)
- [2. 从前两个项目到图像分类](#sec-2)
- [3. 单文件版整体结构](#sec-3)
- [4. 导入库与固定随机种子](#sec-4)
- [5. 图像数据与 Tensor 维度](#sec-5)
- [6. 图像预处理与数据增强 Transform](#sec-6)
- [7. 自定义 `food_Dataset`](#sec-7)
- [8. `DataLoader`：按 Batch 读取图像](#sec-8)
- [9. 半监督学习与伪标签](#sec-9)
- [10. 自定义 CNN：`myModel`](#sec-10)
- [11. 卷积层 `Conv2d`](#sec-11)
- [12. Batch Normalization](#sec-12)
- [13. ReLU 与最大池化](#sec-13)
- [14. CNN 中特征图尺寸如何变化](#sec-14)
- [15. `view` 展平与全连接层](#sec-15)
- [16. 分类输出、Logits、Softmax 与 Argmax](#sec-16)
- [17. `CrossEntropyLoss`](#sec-17)
- [18. 训练阶段](#sec-18)
- [19. 半监督数据的训练](#sec-19)
- [20. 验证阶段：`eval()` 与 `no_grad()`](#sec-20)
- [21. 准确率的计算](#sec-21)
- [22. 生成伪标签的时机](#sec-22)
- [23. 保存最佳模型](#sec-23)
- [24. 主程序：路径、模型与超参数](#sec-24)
- [25. 单文件版完整运行流程](#sec-25)
- [26. 单文件版代码中的几个注意点](#sec-26)
- [27. 多文件版整体结构](#sec-27)
- [28. 多文件版新增知识一：工程模块化](#sec-28)
- [29. 多文件版新增知识二：更丰富的数据增强](#sec-29)
- [30. 多文件版新增知识三：经典 CNN 网络](#sec-30)
- [31. 多文件版新增知识四：统一模型初始化](#sec-31)
- [32. 多文件版新增知识五：预训练与迁移学习](#sec-32)
- [33. 多文件版新增知识六：冻结参数与微调](#sec-33)
- [34. 多文件版 `main.py` 的作用](#sec-34)
- [35. 单文件版与多文件版的关系](#sec-35)
- [36. 本项目必须掌握的知识点](#sec-36)
- [37. 推荐学习顺序](#sec-37)

---

<a id="sec-1"></a>

# 1. 项目任务：Food-11 图像分类

Food-11 是一个食物图像分类任务。

模型的输入是一张食物图片，例如：

```text
一张面包图片
```

模型需要输出：

```text
这张图片属于 11 类食物中的哪一类
```

因此，本项目和前两个项目最大的区别是：

```text
前两个项目：
数值特征
   ↓
神经网络
   ↓
连续数值
   ↓
回归问题

本项目：
RGB 图片
   ↓
卷积神经网络 CNN
   ↓
11 个类别得分
   ↓
分类问题
```

Food-11 一共有 11 个类别，代码中使用：

```text
00
01
02
...
10
```

这些文件夹编号作为类别标签。

最终类别标签可以表示为：

```text
0, 1, 2, ..., 10
```

也就是说，本项目是一个：

> **11 分类任务（11-class classification）**

---

<a id="sec-2"></a>

# 2. 从前两个项目到图像分类

三个项目之间不是彼此独立的，而是一条逐渐增加难度的 PyTorch 学习路线。

```text
项目一：线性回归
│
├── Tensor
├── 模型
├── Loss
├── backward()
└── optimizer.step()
        ↓
项目二：COVID 回归
│
├── Dataset
├── DataLoader
├── Batch
├── nn.Module
├── Train / Validation
└── 完整训练流程
        ↓
项目三：Food-11 图像分类
│
├── 图像 Tensor
├── Transform
├── CNN
├── BatchNorm
├── CrossEntropyLoss
├── Accuracy
├── ResNet / VGG 等经典网络
├── Transfer Learning
└── Pseudo Label
```

因此，第三个项目并没有改变 PyTorch 最核心的训练逻辑。

训练仍然是：

```python
pred = model(x)
loss_value = loss(pred, target)

loss_value.backward()
optimizer.step()
```

核心流程仍然是：

```text
输入
 ↓
模型前向传播
 ↓
得到预测
 ↓
计算 Loss
 ↓
反向传播
 ↓
得到梯度
 ↓
优化器更新参数
```

只是现在的输入从普通数值变成了图像，模型也从全连接网络变成了卷积神经网络。

---

<a id="sec-3"></a>

# 3. 单文件版整体结构

`simple_class.py` 把整个项目几乎全部写在一个文件中。

大体可以分为：

```text
simple_class.py
│
├── 1. 导入库
├── 2. 固定随机种子
├── 3. 定义图像 Transform
├── 4. 定义 food_Dataset
├── 5. 定义 semiDataset
├── 6. 定义自定义 CNN
├── 7. 定义 train_val()
├── 8. 设置数据路径
├── 9. 创建 Dataset / DataLoader
├── 10. 创建模型
├── 11. 设置 Loss / Optimizer
└── 12. 开始训练
```

对于初学者，单文件版本非常适合第一次学习。

因为所有步骤都在同一个文件里，可以直接从上往下阅读，不需要频繁在多个 Python 文件之间跳转。

---

<a id="sec-4"></a>

# 4. 导入库与固定随机种子

代码使用了：

```python
import random
import torch
import torch.nn as nn
import numpy as np
import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchvision import transforms

import time
import matplotlib.pyplot as plt
```

主要作用如下。

| 库 | 作用 |
|---|---|
| `torch` | PyTorch 主库 |
| `torch.nn` | 神经网络层、损失函数 |
| `numpy` | NumPy 数组与数值计算 |
| `PIL.Image` | 读取图片 |
| `Dataset` | 定义数据集 |
| `DataLoader` | Batch 读取数据 |
| `transforms` | 图像预处理与数据增强 |
| `tqdm` | 显示读取进度条 |
| `matplotlib` | 绘制 Loss、Accuracy 曲线 |
| `os` | 文件和路径操作 |

## 4.1 为什么要固定随机种子

代码中：

```python
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    np.random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)
```

然后：

```python
seed_everything(0)
```

深度学习训练中有很多随机操作，例如：

```text
参数随机初始化
随机数据增强
DataLoader shuffle
某些 CUDA 运算
```

如果每次随机结果都不同，那么两次实验可能得到不同结果。

固定随机种子的目的就是：

> **尽可能提高实验的可复现性。**

即：

```text
相同代码
+
相同数据
+
相同随机种子
≈
尽可能相近的实验结果
```

需要注意的是，固定随机种子能够提高可复现性，但不能保证所有环境中每一次结果都绝对完全一致。

---

<a id="sec-5"></a>

# 5. 图像数据与 Tensor 维度

代码设置：

```python
HW = 224
```

表示所有图片最终统一到：

```text
224 × 224
```

对于一张 RGB 彩色图片，普通图像数据常见排列方式为：

```text
H × W × C
```

例如：

```text
224 × 224 × 3
```

其中：

```text
H = Height，高度
W = Width，宽度
C = Channel，通道数
```

RGB 图片有三个通道：

```text
R
G
B
```

但是 PyTorch 的 `Conv2d` 默认输入格式是：

```text
C × H × W
```

因此经过 `ToTensor()` 后：

```text
224 × 224 × 3
```

会变成：

```text
3 × 224 × 224
```

如果一个 Batch 有 16 张图片：

```text
[16, 3, 224, 224]
```

即：

```text
[B, C, H, W]
```

其中：

```text
B = Batch Size
C = Channel
H = Height
W = Width
```

这是以后学习 CNN 时最常见的 Tensor 形状之一。

---

<a id="sec-6"></a>

# 6. 图像预处理与数据增强 Transform

单文件中分别定义了训练集和验证集的 Transform。

## 6.1 训练集 Transform

```python
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(224),
    transforms.RandomRotation(50),
    transforms.ToTensor()
])
```

`Compose` 的作用是：

> 按照顺序，把多个图像处理操作连接起来。

流程为：

```text
NumPy 图像
 ↓
ToPILImage
 ↓
RandomResizedCrop
 ↓
RandomRotation
 ↓
ToTensor
 ↓
送入模型
```

### `ToPILImage()`

把 NumPy 数组转换为 PIL 图片。

因为 torchvision 中很多图像增强操作可以直接处理 PIL Image。

### `RandomResizedCrop(224)`

随机裁剪图像的一部分，再缩放为：

```text
224 × 224
```

这样模型不会只记住食物固定出现在图片哪个位置。

### `RandomRotation(50)`

在一定角度范围内随机旋转图片。

作用是制造更多不同姿态的数据。

### `ToTensor()`

将图片转换为 PyTorch Tensor。

同时通常会把像素从：

```text
0 ~ 255
```

转换到：

```text
0 ~ 1
```

并把维度：

```text
H × W × C
```

改成：

```text
C × H × W
```

---

## 6.2 什么是数据增强

训练集中的：

```text
随机裁剪
随机旋转
```

属于：

> **Data Augmentation，数据增强**

假设训练集中只有一张食物图片。

经过随机增强后，每个 epoch 可能看到：

```text
稍微旋转的版本
裁剪位置不同的版本
缩放不同的版本
```

相当于人为增加了训练数据的变化。

主要目的是：

```text
降低模型死记训练图片的可能
         ↓
提高泛化能力
         ↓
缓解过拟合
```

---

## 6.3 验证集为什么不做随机增强

验证集：

```python
val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])
```

验证集没有：

```text
RandomResizedCrop
RandomRotation
```

原因是验证集主要用于评价模型。

如果每次验证都随机改变图片，那么同一个模型每次看到的验证数据也不同，评价结果会更加不稳定。

因此一般：

```text
训练集：
允许随机数据增强

验证集：
使用确定性的预处理
```

---

<a id="sec-7"></a>

# 7. 自定义 `food_Dataset`

项目定义：

```python
class food_Dataset(Dataset):
```

它继承：

```python
torch.utils.data.Dataset
```

一个自定义 Dataset 通常需要重点实现：

```python
__init__()
__getitem__()
__len__()
```

---

## 7.1 `__init__`

```python
def __init__(self, path, mode="train"):
    self.mode = mode
```

这里的 `mode` 用来区分：

```text
train
val
semi
```

其中：

```text
train → 有标签训练数据
val   → 有标签验证数据
semi  → 无标签数据
```

---

## 7.2 为什么标签要转成 `LongTensor`

有标签数据读取后：

```python
self.Y = torch.LongTensor(self.Y)
```

这是因为本项目使用：

```python
nn.CrossEntropyLoss()
```

它的常见标签形式不是：

```text
[0.0, 0.0, 1.0, ...]
```

而是直接使用类别编号：

```text
0
1
2
...
10
```

也就是说：

```text
第 1 张图片 → 类别 3
第 2 张图片 → 类别 8
第 3 张图片 → 类别 0
```

类别编号通常使用整数类型：

```python
torch.long
```

---

## 7.3 `read_file()` 如何生成标签

有标签数据部分：

```python
for i in tqdm(range(11)):
    file_dir = path + "/%02d" % i
```

`range(11)`：

```text
0 ~ 10
```

`%02d` 会把数字格式化成两位：

```text
0  → 00
1  → 01
2  → 02
...
10 → 10
```

因此程序会依次进入：

```text
00/
01/
02/
...
10/
```

读取图片。

然后：

```python
yi[j] = i
```

例如当前：

```text
i = 5
```

那么 `05/` 文件夹里的图片标签全部设置为：

```text
5
```

这说明项目采用的是：

> **文件夹名称决定类别标签**

---

## 7.4 图片读取

代码：

```python
img = Image.open(img_path)
img = img.resize((HW, HW))
xi[j, ...] = img
```

流程：

```text
图片路径
 ↓
Image.open()
 ↓
Resize 到 224 × 224
 ↓
存入 NumPy 数组
```

当前代码把图片先统一读进：

```python
self.X
```

因此 Dataset 创建完成以后，大量图片已经保存在内存中。

这种写法对于学习项目比较直观。

不过完整数据集较大时，更常见的工程写法是：

```text
Dataset 只保存图片路径
        ↓
__getitem__ 被调用时
        ↓
才 Image.open() 当前这一张图片
```

这样可以降低内存占用。

---

## 7.5 `__getitem__`

有标签模式：

```python
return self.transform(self.X[item]), self.Y[item]
```

即返回：

```text
处理后的图片 Tensor + 标签
```

例如：

```text
x.shape = [3, 224, 224]
y = 4
```

无标签模式：

```python
return self.transform(self.X[item]), self.X[item]
```

这里同时返回：

```text
处理后的图片
+
原始图片
```

这是为了后面生成伪标签后，可以重新把原始图像加入训练集。

---

## 7.6 `__len__`

```python
def __len__(self):
    return len(self.X)
```

告诉 DataLoader：

> 这个数据集一共有多少个样本。

---

<a id="sec-8"></a>

# 8. `DataLoader`：按 Batch 读取图像

代码：

```python
train_loader = DataLoader(
    train_set,
    batch_size=16,
    shuffle=True
)

val_loader = DataLoader(
    val_set,
    batch_size=16,
    shuffle=False
)

no_label_loader = DataLoader(
    no_label_set,
    batch_size=16,
    shuffle=False
)
```

如果训练集一张图片是：

```text
[3, 224, 224]
```

当：

```python
batch_size = 16
```

DataLoader 一次返回：

```text
[16, 3, 224, 224]
```

以及 16 个标签：

```text
[16]
```

## 8.1 为什么训练集 `shuffle=True`

训练时打乱数据：

```python
shuffle=True
```

可以避免模型每个 epoch 都按完全相同顺序看到类别和样本。

## 8.2 为什么验证集 `shuffle=False`

验证阶段只需要稳定地评价模型。

因此：

```python
shuffle=False
```

即可。

---

<a id="sec-9"></a>

# 9. 半监督学习与伪标签

这个项目除了：

```text
有标签训练数据
```

还有：

```text
无标签数据
```

普通监督学习只能直接使用：

```text
图片 x
+
正确标签 y
```

因为训练需要：

```text
模型预测
   ↓
和真实标签比较
   ↓
计算 Loss
```

但是无标签图片没有 `y`。

因此项目使用：

> **Pseudo Label，伪标签**

基本思想：

```text
先训练一个模型
       ↓
模型预测无标签图片
       ↓
只保留置信度很高的结果
       ↓
把模型预测的类别当成“临时标签”
       ↓
加入训练
```

这属于：

> **Semi-supervised Learning，半监督学习**

因为训练同时利用：

```text
有标签数据 + 无标签数据
```

---

<a id="sec-10"></a>

# 10. 自定义 CNN：`myModel`

单文件定义了一个自己的 CNN：

```python
class myModel(nn.Module):
```

整体结构可以简化为：

```text
输入
3 × 224 × 224
      ↓
Conv + BN + ReLU + Pool
      ↓
64 × 112 × 112
      ↓
Conv + BN + ReLU + Pool
      ↓
128 × 56 × 56
      ↓
Conv + BN + ReLU + Pool
      ↓
256 × 28 × 28
      ↓
Conv + BN + ReLU + Pool
      ↓
512 × 14 × 14
      ↓
Pool
      ↓
512 × 7 × 7
      ↓
Flatten
      ↓
25088
      ↓
Linear
      ↓
1000
      ↓
Linear
      ↓
11
```

这个 CNN 的核心任务是：

> **逐层从图片中提取特征，最后完成 11 分类。**

---

<a id="sec-11"></a>

# 11. 卷积层 `Conv2d`

第一层：

```python
self.conv1 = nn.Conv2d(
    3,
    64,
    3,
    1,
    1
)
```

可以对应参数：

```python
nn.Conv2d(
    in_channels=3,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1
)
```

---

## 11.1 `in_channels=3`

输入是 RGB 图片：

```text
R
G
B
```

因此：

```text
输入通道数 = 3
```

---

## 11.2 `out_channels=64`

表示这一层要学习：

```text
64 组卷积核
```

最终得到：

```text
64 个特征通道
```

因此：

```text
3 × 224 × 224
      ↓
64 × 224 × 224
```

可以把每个输出通道理解为：

> 网络从原图中提取出的一类特征响应。

前面层可能逐渐学习：

```text
边缘
方向
颜色变化
局部纹理
```

后面的层则会逐渐组合成更复杂的特征。

---

## 11.3 `kernel_size=3`

表示卷积核大小：

```text
3 × 3
```

可以直观理解为：

> 每次观察图片中的一个局部 3×3 区域。

---

## 11.4 `stride=1`

表示卷积核每次移动：

```text
1 个像素
```

---

## 11.5 `padding=1`

在图像边缘补一圈。

当：

```text
kernel_size = 3
stride = 1
padding = 1
```

时，空间尺寸可以保持不变：

```text
224 × 224
 ↓
224 × 224
```

---

<a id="sec-12"></a>

# 12. Batch Normalization

第一层卷积之后：

```python
self.bn1 = nn.BatchNorm2d(64)
```

后面卷积块中也有：

```python
nn.BatchNorm2d(128)
nn.BatchNorm2d(256)
nn.BatchNorm2d(512)
```

BatchNorm 全称：

> **Batch Normalization，批归一化**

可以先直观理解为：

> 对神经网络中间层的特征进行标准化和重新缩放，使训练过程更加稳定。

核心标准化形式：

$$
\hat{x}
=
\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}
$$

然后再进行可学习的缩放和平移：

$$
y
=
\gamma\hat{x}
+
\beta
$$

其中：

- $\mu$：均值；
- $\sigma^2$：方差；
- $\epsilon$：防止除零；
- $\gamma$：可学习的缩放参数；
- $\beta$：可学习的平移参数。

## 12.1 `BatchNorm2d(64)` 中的 64

假设卷积输出：

```text
[B, 64, H, W]
```

BatchNorm2d 会针对：

```text
64 个 Channel
```

分别进行统计和变换。

因此：

```python
nn.Conv2d(3, 64, ...)
```

后面通常对应：

```python
nn.BatchNorm2d(64)
```

---

## 12.2 BatchNorm 与 `train()` / `eval()`

训练阶段：

```python
model.train()
```

BatchNorm 会使用训练 Batch 的统计信息，并更新内部的运行统计量。

验证阶段：

```python
model.eval()
```

BatchNorm 使用训练过程中积累的运行均值和方差。

因此：

> `model.train()` 和 `model.eval()` 不只是形式上的写法，它们会真正影响 BatchNorm 等层的行为。

---

<a id="sec-13"></a>

# 13. ReLU 与最大池化

卷积后：

```python
self.relu = nn.ReLU()
```

ReLU：

$$
\operatorname{ReLU}(x)
=
\max(0,x)
$$

也就是：

```text
负数 → 0
正数 → 保留
```

作用之一是：

> 给神经网络加入非线性表达能力。

如果网络里只有线性运算，即使堆很多层，本质上仍然很难表示复杂的非线性映射。

---

## 13.1 最大池化

代码：

```python
nn.MaxPool2d(2)
```

表示使用：

```text
2 × 2
```

区域做最大池化。

简单理解：

```text
2 × 2 区域
↓
只保留最大值
```

它会使空间尺寸大约缩小一半：

```text
224 × 224
 ↓
112 × 112
```

再池化：

```text
112 × 112
 ↓
56 × 56
```

作用可以粗略理解为：

```text
减少空间尺寸
降低计算量
保留较强特征响应
逐渐扩大有效感受区域
```

---

<a id="sec-14"></a>

# 14. CNN 中特征图尺寸如何变化

输入：

```text
[B, 3, 224, 224]
```

第一层卷积：

```python
Conv2d(3, 64, 3, 1, 1)
```

得到：

```text
[B, 64, 224, 224]
```

池化：

```text
[B, 64, 112, 112]
```

`layer1`：

```text
[B, 128, 56, 56]
```

`layer2`：

```text
[B, 256, 28, 28]
```

`layer3`：

```text
[B, 512, 14, 14]
```

最后一次池化：

```text
[B, 512, 7, 7]
```

因此这个网络有一个很典型的 CNN 特征：

```text
空间尺寸：
224 → 112 → 56 → 28 → 14 → 7

通道数：
3 → 64 → 128 → 256 → 512
```

可以理解成：

> 随着网络加深，图片的空间分辨率逐渐降低，但网络提取的特征种类和语义复杂度逐渐增加。

---

<a id="sec-15"></a>

# 15. `view` 展平与全连接层

在 `forward()` 中：

```python
x = x.view(x.size()[0], -1)
```

假设池化后的 Tensor：

```text
x.shape = [16, 512, 7, 7]
```

其中：

```python
x.size()[0]
```

就是：

```text
16
```

也就是 Batch Size。

因此：

```python
x.view(16, -1)
```

中的 `-1` 表示：

> 这一维由 PyTorch 自动计算。

由于：

$$
512\times7\times7=25088
$$

所以：

```text
[16, 512, 7, 7]
        ↓
[16, 25088]
```

这一步称为：

> **Flatten，展平**

然后进入：

```python
self.fc1 = nn.Linear(25088, 1000)
```

即：

```text
25088 个特征
      ↓
1000 个特征
```

再：

```python
self.fc2 = nn.Linear(1000, num_class)
```

如果：

```text
num_class = 11
```

最终：

```text
1000
 ↓
11
```

---

<a id="sec-16"></a>

# 16. 分类输出、Logits、Softmax 与 Argmax

模型最后：

```python
x = self.fc2(x)
return x
```

假设 Batch Size = 16：

```text
输出 shape = [16, 11]
```

也就是：

> 每张图片对应 11 个类别得分。

例如某一张图片：

```text
[-1.4, 0.2, 2.8, 0.5, ..., -0.7]
```

这些原始分数称为：

> **Logits**

Logits：

- 可以小于 0；
- 可以大于 1；
- 加起来不等于 1；
- 不是概率。

---

## 16.1 为什么取最大值就能得到类别

分类时：

```python
np.argmax(pred, axis=1)
```

表示：

> 每张图片的 11 个类别得分中，找到最大的那个位置。

例如：

```text
[0.2, 0.5, 3.1, 0.1]
```

最大值：

```text
3.1
```

位置：

```text
2
```

因此预测类别：

```text
2
```

---

## 16.2 Softmax

Softmax 可以把 Logits 转换成概率：

$$
p_i
=
\frac{e^{z_i}}
{\sum_j e^{z_j}}
$$

例如：

```text
Logits:
[1.2, 0.5, 3.0]

Softmax:
[0.13, 0.06, 0.81]
```

所有概率之和：

```text
= 1
```

---

## 16.3 为什么正常训练没有手动 Softmax

训练代码：

```python
pred = model(x)
train_bat_loss = loss(pred, target)
```

这里没有：

```python
Softmax(pred)
```

这是正确的。

因为本项目使用：

```python
nn.CrossEntropyLoss()
```

它应该直接接收：

```text
原始 Logits
```

所以不要写成：

```python
pred = torch.softmax(model(x), dim=1)
loss = criterion(pred, target)
```

---

## 16.4 为什么伪标签部分需要 Softmax

伪标签不是只需要“哪个类别最大”，还需要知道：

> 模型到底有多确信。

因此：

```python
soft = nn.Softmax(dim=1)
pred_soft = soft(pred)
```

把 Logits 变成概率。

然后：

```python
pred_max, pred_value = pred_soft.max(dim=1)
```

假设：

```text
pred_soft.shape = [16, 11]
```

`dim=1` 表示：

> 对每张图片的 11 个类别寻找最大概率。

返回：

```text
pred_max
```

表示最大概率，也就是置信度。

例如：

```text
[0.995, 0.87, 0.999, ...]
```

而：

```text
pred_value
```

表示最大概率所在类别。

例如：

```text
[8, 3, 5, ...]
```

即：

```text
第一张 → 99.5% 认为是类别 8
第二张 → 87% 认为是类别 3
第三张 → 99.9% 认为是类别 5
```

---

<a id="sec-17"></a>

# 17. `CrossEntropyLoss`

项目：

```python
loss = nn.CrossEntropyLoss()
```

交叉熵是多分类问题中非常常见的损失函数。

模型输出：

```text
[B, 11]
```

标签：

```text
[B]
```

例如：

```text
模型输出：
第1张 → 11个 Logits
第2张 → 11个 Logits
...

真实标签：
[3, 8, 1, ...]
```

CrossEntropyLoss 会鼓励：

> 正确类别的得分越来越高，错误类别相对越来越低。

在 PyTorch 中可以粗略理解为：

```text
Logits
 ↓
内部完成 LogSoftmax 等计算
 ↓
和正确类别比较
 ↓
Cross Entropy Loss
```

因此训练阶段不需要自己先 Softmax。

---

<a id="sec-18"></a>

# 18. 训练阶段

训练开始：

```python
model.train()
```

表示：

> 切换到训练模式。

然后：

```python
for batch_x, batch_y in train_loader:
```

每次读取一个 Batch。

---

## 18.1 把数据移动到设备

```python
x = batch_x.to(device)
target = batch_y.to(device)
```

如果：

```text
device = cuda
```

就把数据移动到 GPU。

如果没有 CUDA：

```text
device = cpu
```

就在 CPU 上计算。

---

## 18.2 前向传播

```python
pred = model(x)
```

即：

```text
图片
 ↓
CNN
 ↓
11 类 Logits
```

---

## 18.3 计算 Loss

```python
train_bat_loss = loss(pred, target)
```

比较：

```text
模型预测
+
真实标签
```

得到当前 Batch 的损失。

---

## 18.4 反向传播

```python
train_bat_loss.backward()
```

PyTorch 根据计算图自动计算：

```text
Loss
对
所有可训练参数
的梯度
```

也就是：

$$
\frac{\partial L}{\partial \theta}
$$

---

## 18.5 更新参数

```python
optimizer.step()
```

优化器根据梯度修改模型参数。

整体：

```text
forward
↓
loss
↓
backward
↓
gradient
↓
optimizer.step()
```

---

## 18.6 为什么要清空梯度

PyTorch 默认梯度会累加。

因此每个 Batch 需要清除上一轮梯度。

当前代码使用：

```python
optimizer.step()
optimizer.zero_grad()
```

它能够工作，因为每一轮更新后清空梯度。

更常见、更容易理解的写法是：

```python
for batch_x, batch_y in train_loader:

    optimizer.zero_grad()

    pred = model(x)
    loss_value = loss(pred, target)

    loss_value.backward()
    optimizer.step()
```

即：

```text
清梯度
↓
前向传播
↓
计算 Loss
↓
反向传播
↓
更新参数
```

---

<a id="sec-19"></a>

# 19. 半监督数据的训练

当：

```python
semi_loader != None
```

说明已经筛选出了一批高置信度伪标签数据。

训练过程和普通监督数据几乎相同：

```python
pred = model(x)
semi_bat_loss = loss(pred, target)

semi_bat_loss.backward()
optimizer.step()
optimizer.zero_grad()
```

区别只有一个：

```text
普通训练数据的 target
=
人工真实标签

semi 数据的 target
=
模型此前生成的伪标签
```

因此半监督学习的风险也很明显：

```text
模型预测错
 ↓
得到错误伪标签
 ↓
错误伪标签重新训练模型
 ↓
可能继续强化错误
```

所以代码不会把所有无标签数据都加入训练。

---

<a id="sec-20"></a>

# 20. 验证阶段：`eval()` 与 `no_grad()`

验证开始：

```python
model.eval()
```

然后：

```python
with torch.no_grad():
```

这两个作用不同。

---

## 20.1 `model.eval()`

它会切换模型工作模式。

主要影响：

```text
BatchNorm
Dropout
```

例如 BatchNorm 在训练阶段和验证阶段使用统计量的方式不同。

---

## 20.2 `torch.no_grad()`

表示：

> 当前代码不需要计算梯度。

验证阶段只需要：

```text
forward
↓
得到预测
↓
计算 Loss / Accuracy
```

不需要：

```text
backward
optimizer.step
```

因此关闭梯度可以：

```text
减少显存占用
减少不必要计算
```

---

## 20.3 `eval()` 不等于 `no_grad()`

一定要区分：

```text
model.eval()
→ 改变某些网络层的运行模式

torch.no_grad()
→ 不记录梯度
```

两者通常在验证阶段一起使用。

---

<a id="sec-21"></a>

# 21. 准确率的计算

代码：

```python
np.argmax(
    pred.detach().cpu().numpy(),
    axis=1
)
```

先获得每张图片预测类别。

然后：

```python
== target.cpu().numpy()
```

和真实类别比较。

例如：

```text
预测：
[0, 2, 5, 8]

真实：
[0, 2, 3, 8]
```

比较：

```text
True
True
False
True
```

正确：

```text
3 个
```

准确率：

$$
Accuracy
=
\frac{预测正确数量}
{总样本数量}
$$

例如：

$$
Accuracy
=
\frac{3}{4}
=
75\%
$$

---

<a id="sec-22"></a>

# 22. 生成伪标签的时机

代码并不是从第一轮就生成伪标签。

而是：

```python
if epoch % 3 == 0 and plt_val_acc[-1] > 0.6:
    semi_loader = get_semi_loader(...)
```

需要满足：

```text
条件一：
epoch 每隔一定轮次

条件二：
验证准确率 > 0.6
```

原因：

> 如果模型本身还没有基本分类能力，就让它给无标签数据生成标签，大量伪标签可能是错误的。

因此先：

```text
有标签数据训练模型
        ↓
模型达到一定效果
        ↓
再利用无标签数据
```

---

## 22.1 置信度阈值

主程序：

```python
thres = 0.99
```

生成伪标签时：

```python
if prob > thres:
```

也就是说：

```text
模型最大预测概率 > 99%
```

才使用这张图片。

例如：

```text
图片 A：
类别 5 → 99.6%
→ 保留

图片 B：
类别 3 → 74%
→ 不保留
```

设计思想是：

> 宁愿少选一些无标签数据，也尽可能减少错误伪标签。

---

<a id="sec-23"></a>

# 23. 保存最佳模型

验证结束：

```python
if val_acc > max_acc:
    torch.save(model, save_path)
    max_acc = val_acc
```

含义：

```text
当前模型
↓
验证表现超过之前最好结果
↓
保存
```

这样最终保存的不是：

```text
最后一轮模型
```

而是：

```text
训练过程中验证表现最好的模型
```

当前代码中的 `val_acc` 在这里实际累计的是：

```text
预测正确的图片数量
```

由于每轮验证集大小固定，所以：

```text
正确图片数量最大
```

和：

```text
验证准确率最大
```

排序是一致的。

不过为了代码语义更清晰，也可以改成真正比较：

```python
current_val_acc = val_acc / len(val_loader.dataset)
```

---

<a id="sec-24"></a>

# 24. 主程序：路径、模型与超参数

单文件最后设置数据路径：

```python
train_path = ...
val_path = ...
no_label_path = ...
```

然后创建 Dataset：

```python
train_set = food_Dataset(train_path, "train")
val_set = food_Dataset(val_path, "val")
no_label_set = food_Dataset(no_label_path, "semi")
```

再创建 DataLoader：

```python
train_loader = DataLoader(
    train_set,
    batch_size=16,
    shuffle=True
)

val_loader = DataLoader(
    val_set,
    batch_size=16,
    shuffle=False
)

no_label_loader = DataLoader(
    no_label_set,
    batch_size=16,
    shuffle=False
)
```

---

## 24.1 自定义 CNN 与经典网络

代码中保留：

```python
# model = myModel(11)
```

说明可以使用自己定义的 CNN。

当前又使用：

```python
model, _ = initialize_model(
    "vgg",
    11,
    use_pretrained=True
)
```

说明除了自己写 CNN，还可以调用经典 CNN 模型。

这也是单文件版本向多文件版本过渡的地方。

---

## 24.2 学习率

```python
lr = 0.001
```

Learning Rate 决定：

> 每次参数更新走多大的步子。

太大可能：

```text
训练震荡
无法稳定收敛
```

太小可能：

```text
训练非常慢
```

---

## 24.3 AdamW

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=lr,
    weight_decay=1e-4
)
```

AdamW 是常见优化器。

其中：

```python
weight_decay=1e-4
```

属于权重衰减正则化设置，可用于抑制参数无限变大，在一定程度上帮助缓解过拟合。

---

## 24.4 自动选择 CPU / GPU

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

含义：

```text
如果 CUDA 可用
→ GPU

否则
→ CPU
```

因此同一份代码可以根据环境自动选择运行设备。

---

## 24.5 Epoch

```python
epochs = 15
```

一个 Epoch 表示：

> 训练集中的所有样本整体被模型学习一遍。

例如：

```text
训练集 8000 张图片

epoch = 1
→ 8000 张总体走一遍训练

epoch = 15
→ 总体走 15 遍
```

---

<a id="sec-25"></a>

# 25. 单文件版完整运行流程

把整个 `simple_class.py` 串起来：

```text
程序启动
│
├── 固定随机种子
│
├── 定义 Transform
│
├── 读取训练图片
│      └── 生成真实标签
│
├── 读取验证图片
│      └── 生成真实标签
│
├── 读取无标签图片
│
├── 创建 Dataset
│
├── 创建 DataLoader
│
├── 创建 CNN / VGG 等模型
│
├── 创建 CrossEntropyLoss
│
├── 创建 AdamW
│
└── train_val()
       │
       ├── model.train()
       │
       ├── 有标签训练
       │      ├── forward
       │      ├── loss
       │      ├── backward
       │      └── optimizer.step
       │
       ├── 如果已有 semi_loader
       │      └── 伪标签数据继续训练
       │
       ├── model.eval()
       │
       ├── torch.no_grad()
       │
       ├── 验证
       │      ├── val loss
       │      └── val accuracy
       │
       ├── 满足条件
       │      └── 无标签数据生成伪标签
       │
       ├── 保存最佳模型
       │
       └── 进入下一 Epoch
```

这就是本项目最重要的一条主线。

---

<a id="sec-26"></a>

# 26. 单文件版代码中的几个注意点

这一节不是新增模型知识，而是当前单文件代码值得注意的实现细节。

## 26.1 `from datetime import dim` 不需要

如果代码顶部出现：

```python
from datetime import dim
```

应删除。

因为：

```python
pred_soft.max(dim=1)
```

中的 `dim` 只是函数参数名，不需要导入任何 `dim`。

---

## 26.2 读取图片时可以强制 RGB

当前：

```python
img = Image.open(img_path)
```

更稳妥的写法：

```python
img = Image.open(img_path).convert("RGB")
```

因为网络输入固定要求：

```text
3 个通道
```

如果以后遇到灰度图片或 RGBA 图片，强制转换 RGB 可以避免通道数不一致。

---

## 26.3 使用 ImageNet 预训练模型时要注意 Normalize

当前单文件 Transform 里没有 Normalize。

如果：

```python
use_pretrained=True
```

真正加载的是 torchvision 的 ImageNet 预训练权重，那么通常还应该使用和对应预训练权重匹配的归一化。

典型 ImageNet Normalize：

```python
transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
```

应放在：

```python
transforms.ToTensor()
```

之后。

这里需要注意：

> Normalize 是使用预训练模型时的数据预处理要求之一；当前代码没有写，并不影响代码语法运行，但可能影响迁移学习效果。

---

## 26.4 伪标签推理最好显式调用 `model.eval()`

`get_label()` 中已经使用：

```python
with torch.no_grad():
```

如果单独调用这个函数，更稳妥的做法是先：

```python
model.eval()
```

然后再进行无标签数据推理。

---

## 26.5 Dataset 一次性读入图片的内存问题

当前代码在 `read_file()` 中把图片都存入：

```python
self.X
```

对于 sample 数据很直观。

但是完整大数据集时：

```text
图片数量增加
↓
内存占用增加
```

更常见做法是：

```text
Dataset 保存文件路径
↓
__getitem__()
↓
临时读取当前图片
```

这是以后做更大项目时需要注意的工程问题。

---

<a id="sec-27"></a>

# 27. 多文件版整体结构

理解单文件版以后，再看多文件版。

多文件版本把原来一个文件中的不同功能拆开。

项目主要结构可以理解为：

```text
03_food_classification/
│
├── main.py
│
└── model_utils/
    ├── data.py
    ├── model.py
    └── train.py
```

对应：

```text
data.py
→ 数据读取、Transform、DataLoader

model.py
→ 模型定义、经典网络初始化

train.py
→ 训练、验证、伪标签等训练逻辑

main.py
→ 参数配置与总入口
```

下面不再重复 Dataset、DataLoader、CNN、CrossEntropy、训练验证这些单文件已经讲过的知识。

只介绍多文件版本真正新增的内容。

---

<a id="sec-28"></a>

# 28. 多文件版新增知识一：工程模块化

这是多文件版最直接的新知识。

单文件：

```text
simple_class.py
→ 所有功能放在一起
```

优点：

```text
第一次学习容易跟流程
```

缺点：

```text
代码越来越长
维护困难
不同功能混在一起
```

多文件版：

```text
数据相关 → data.py
模型相关 → model.py
训练相关 → train.py
参数入口 → main.py
```

这种设计叫：

> **模块化（Modularization）**

核心思想：

> 一个文件尽量负责一类职责。

例如以后只想修改数据增强：

```text
打开 data.py
```

想换模型：

```text
打开 model.py
```

想改训练流程：

```text
打开 train.py
```

这种代码组织方式更接近真正的深度学习工程。

---

<a id="sec-29"></a>

# 29. 多文件版新增知识二：更丰富的数据增强

相比单文件版中的：

```text
RandomResizedCrop
RandomRotation
```

多文件版训练 Transform 进一步加入了类似：

```python
transforms.RandomHorizontalFlip()
transforms.AutoAugment()
```

---

## 29.1 `RandomHorizontalFlip`

随机水平翻转图片。

例如：

```text
原图：
食物在左边

随机翻转：
食物在右边
```

模型不能简单依赖：

```text
某个物体一定出现在固定左右位置
```

从而提高泛化能力。

---

## 29.2 `AutoAugment`

`AutoAugment` 可以理解为：

> torchvision 中预定义的一组自动数据增强策略。

它会组合多种可能的图像变换。

相对于自己只写：

```text
旋转
裁剪
翻转
```

AutoAugment 可以提供更丰富的数据变化。

第一遍学习不需要记住其内部所有策略。

先掌握：

```text
数据增强
目的不是改变类别
而是增加同一类别图像的变化
```

---

## 29.3 Normalize

多文件版中还出现过 Normalize 相关设置。

如果代码里暂时注释：

```python
# transforms.Normalize(...)
```

需要知道：

```text
ToTensor
→ 把像素变成 Tensor

Normalize
→ 再按照给定 mean / std 对各通道标准化
```

二者不是一回事。

---

<a id="sec-30"></a>

# 30. 多文件版新增知识三：经典 CNN 网络

单文件版主要让我们理解：

```text
如何自己搭 CNN
```

多文件版 `model.py` 进一步提供多种经典网络选择，例如：

```text
自定义 CNN
ResNet18
ResNet50
GoogLeNet
AlexNet
VGG
SqueezeNet
```

这里真正的新知识不是“怎么写 forward”，而是：

> PyTorch / torchvision 已经实现了大量经典网络，我们可以直接调用并修改最后的分类层。

这些网络都属于 CNN 家族，但结构设计不同。

---

## 30.1 AlexNet

AlexNet 是经典深度 CNN。

历史意义很重要：

> 证明了深度卷积神经网络在大规模视觉分类任务中的巨大能力。

---

## 30.2 VGG

VGG 的结构特点可以粗略理解为：

> 大量使用小尺寸 `3×3` 卷积，并采用规则的层级堆叠。

结构直观，适合理解 CNN 深度增加的过程。

---

## 30.3 GoogLeNet

GoogLeNet 引入 Inception 结构。

核心思想之一：

> 同一个阶段使用不同尺度的卷积操作，再组合特征。

第一遍只需要知道它和普通顺序堆卷积的网络结构不同即可。

---

## 30.4 ResNet

ResNet 最重要的新知识是：

> **Residual Connection，残差连接**

普通网络：

```text
x
↓
网络层
↓
F(x)
```

ResNet：

```text
x ───────────────┐
↓                │
网络层           │
↓                │
F(x)             │
↓                │
F(x) + x ←───────┘
```

即：

$$
y
=
F(x)+x
$$

这种 Shortcut / Skip Connection 可以让非常深的网络更容易优化。

因此：

```text
ResNet18
ResNet50
```

不是说“完全不同于 CNN”。

而是：

> **ResNet 本身就是一种 CNN 架构。**

---

<a id="sec-31"></a>

# 31. 多文件版新增知识四：统一模型初始化

多文件版将不同模型的创建统一封装进：

```text
initialize_model(...)
```

这样主程序不需要分别写：

```python
resnet18(...)
vgg(...)
alexnet(...)
```

而是只通过：

```text
model_name
```

选择模型。

例如概念上：

```text
model_name = "resnet18"
        ↓
initialize_model()
        ↓
创建 ResNet18
```

或者：

```text
model_name = "vgg"
        ↓
initialize_model()
        ↓
创建 VGG
```

这样做的好处：

```text
换模型时
↓
只改一个配置
↓
训练代码不用重写
```

这体现了工程中很重要的思想：

> **把会变化的配置，与稳定的训练逻辑分离。**

---

<a id="sec-32"></a>

# 32. 多文件版新增知识五：预训练与迁移学习

多文件版最重要的新知识之一是：

> **Pretrained Model，预训练模型**

例如 ResNet18 可以先在 ImageNet 上训练。

经过大规模图像数据训练以后，网络前面的卷积层已经学习到很多通用视觉特征：

```text
边缘
纹理
颜色模式
形状
局部结构
```

现在 Food-11 数据相对更小，就可以：

```text
ImageNet 预训练模型
        ↓
保留已学习视觉特征
        ↓
修改最后分类层
        ↓
1000 类 → 11 类
        ↓
在 Food-11 上继续训练
```

这就是：

> **Transfer Learning，迁移学习**

---

## 32.1 为什么要替换最后分类层

ImageNet 模型原本通常面向：

```text
1000 类
```

而本项目：

```text
11 类
```

所以最后一层必须修改为：

```text
... → 11
```

例如 ResNet 最后的 `fc`：

```text
原：
features → 1000

改：
features → 11
```

这说明：

> 前面的特征提取器可以继续使用，但最终任务分类头必须适应当前数据集。

---

## 32.2 `use_pretrained=False`

仓库多文件版的默认设置曾采用：

```text
ResNet18
+
use_pretrained=False
```

这表示：

```text
使用 ResNet18 的网络结构
```

但是：

```text
不加载 ImageNet 已训练好的参数
```

也就是：

> **从随机初始化开始训练。**

一定要区分：

```text
使用 ResNet18
```

和：

```text
使用预训练 ResNet18
```

这是两个概念。

---

## 32.3 `use_pretrained=True`

如果：

```text
use_pretrained=True
```

则表示希望加载已有预训练参数。

这时就进入：

```text
迁移学习
```

的场景。

对于数据量不大的图像分类任务，预训练模型通常是非常常见的选择。

---

<a id="sec-33"></a>

# 33. 多文件版新增知识六：冻结参数与微调

迁移学习中还有两个重要概念。

---

## 33.1 Fine-tuning

Fine-tuning，微调。

例如：

```text
加载预训练 ResNet
↓
修改最后分类层
↓
Food-11 数据继续训练
↓
前面卷积层也允许更新
```

此时：

```text
整个网络
```

都可以根据 Food-11 调整。

---

## 33.2 冻结参数

PyTorch 可以：

```python
param.requires_grad = False
```

表示：

> 这个参数不需要计算梯度，也不参与正常参数更新。

例如：

```text
预训练 CNN 特征提取器
→ 冻结

最后分类层
→ 训练
```

这时网络前面的知识保持不变，只学习新的分类器。

---

## 33.3 Linear Probing

如果：

```text
预训练特征提取器全部冻结
+
只训练最后线性分类层
```

可以理解为一种：

> **Linear Probing**

它适合用来观察：

> 预训练模型本身学习到的特征，对于新任务是否已经足够有区分能力。

---

## 33.4 Fine-tuning 和冻结的区别

```text
Fine-tuning：
预训练参数
+
继续更新网络较多参数

Linear Probing：
冻结特征提取器
+
主要训练最后分类器
```

第一遍先掌握这个直观区别即可。

---

<a id="sec-34"></a>

# 34. 多文件版 `main.py` 的作用

在多文件结构中：

```text
main.py
```

不应该再放所有类和函数。

它更像：

> **项目总入口 + 配置中心**

例如集中设置：

```text
模型名称
类别数
Batch Size
学习率
Epoch
是否使用预训练
是否使用半监督
置信度阈值
验证准确率阈值
路径
```

此前仓库多文件版采用过类似：

```text
model_name = resnet18
num_class = 11
batch_size = 32
learning_rate = 1e-4
epochs = 10
CrossEntropyLoss
AdamW
conf_thres = 0.99
```

并通过配置决定是否启用半监督训练。

最终：

```text
main.py
 ↓
调用 data.py
 ↓
获得 DataLoader
 ↓
调用 model.py
 ↓
获得模型
 ↓
调用 train.py
 ↓
训练 / 验证
```

因此多文件版的主线是：

```text
main.py
│
├── data.py
├── model.py
└── train.py
```

---

<a id="sec-35"></a>

# 35. 单文件版与多文件版的关系

不要把它们理解成两个完全不同的项目。

它们本质上完成的是同一个 Food-11 分类任务。

区别是学习阶段不同。

## 单文件版

目标：

> **看懂完整深度学习训练流程。**

优点：

```text
所有代码在一个文件
容易顺着执行顺序阅读
适合第一次理解
```

重点学习：

```text
图像 Tensor
Transform
Dataset
DataLoader
CNN
BatchNorm
Pooling
Flatten
CrossEntropyLoss
Train / Eval
Pseudo Label
```

---

## 多文件版

目标：

> **学习如何把已经理解的流程组织成更接近真实项目的代码，并引入更多成熟模型。**

新增重点：

```text
模块化
更丰富的数据增强
经典 CNN
模型统一初始化
ResNet
预训练模型
迁移学习
冻结参数
Fine-tuning
```

因此正确学习顺序是：

```text
先单文件
↓
能够完整说出数据如何流动
↓
再看多文件
↓
重点理解新加入的工程化和迁移学习知识
```

而不是：

```text
两个版本逐行重复学两遍
```

---

<a id="sec-36"></a>

# 36. 本项目必须掌握的知识点

学完本项目以后，至少应该能够回答下面这些问题。

## 数据部分

1. 一张 RGB 图片为什么经过 `ToTensor()` 后是：

```text
[3, H, W]
```

2. 为什么一个 Batch 是：

```text
[B, C, H, W]
```

3. `Dataset` 和 `DataLoader` 分别负责什么？

4. 为什么训练集要数据增强，而验证集一般不做随机增强？

---

## CNN 部分

5. `Conv2d(3, 64, 3, 1, 1)` 每个数字是什么意思？

6. 为什么 CNN 中：

```text
H、W 越来越小
Channel 越来越多
```

7. BatchNorm 是什么？

8. ReLU 是什么？

9. MaxPool 有什么作用？

10. 为什么最后需要：

```python
x.view(x.size()[0], -1)
```

11. 为什么：

$$
512\times7\times7=25088
$$

---

## 分类部分

12. 模型最后为什么输出 11 个数？

13. Logits 是什么？

14. Softmax 是什么？

15. 为什么正常训练 `CrossEntropyLoss` 前不需要手动 Softmax？

16. 为什么 `argmax(logits)` 就可以得到预测类别？

17. 为什么标签要用 `LongTensor`？

---

## 训练部分

18. `model.train()` 和 `model.eval()` 有什么区别？

19. `torch.no_grad()` 是干什么的？

20. `loss.backward()` 做了什么？

21. `optimizer.step()` 做了什么？

22. 为什么要 `zero_grad()`？

23. Accuracy 怎么计算？

---

## 半监督部分

24. 什么是伪标签？

25. 为什么伪标签要设置高置信度阈值？

26. 为什么不能一开始就用伪标签？

27. 错误伪标签可能造成什么问题？

---

## 多文件新增部分

28. 为什么要把项目拆成：

```text
data.py
model.py
train.py
main.py
```

29. ResNet 和普通顺序 CNN 最大的结构区别是什么？

30. 什么是预训练？

31. 什么是迁移学习？

32. `use_pretrained=False` 到底意味着什么？

33. 为什么预训练模型最后的分类层需要改成 11 类？

34. 什么是冻结参数？

35. Fine-tuning 与只训练最后分类层有什么区别？

如果这些问题基本都能解释清楚，那么这个 Food-11 项目的核心知识就已经掌握得比较完整了。

---

<a id="sec-37"></a>

# 37. 推荐学习顺序

建议不要第一次就同时研究：

```text
自定义 CNN
+
VGG
+
ResNet
+
迁移学习
+
伪标签
```

容易把不同层次的知识混在一起。

可以按下面顺序学习。

---

## 第一步：只看数据如何进入网络

重点：

```text
图片
↓
PIL / NumPy
↓
Transform
↓
Tensor
↓
[B, C, H, W]
↓
DataLoader
```

---

## 第二步：只看自定义 CNN

重点：

```text
Conv2d
↓
BatchNorm
↓
ReLU
↓
MaxPool
↓
Flatten
↓
Linear
↓
11 类
```

手算一次：

```text
224
→ 112
→ 56
→ 28
→ 14
→ 7
```

并理解：

```text
512 × 7 × 7 = 25088
```

---

## 第三步：看分类训练

重点：

```text
Logits
CrossEntropyLoss
argmax
Accuracy

model.train()
model.eval()
torch.no_grad()

backward()
optimizer.step()
zero_grad()
```

---

## 第四步：学习伪标签

重点：

```text
无标签数据
↓
模型预测
↓
Softmax
↓
最大概率 + 类别
↓
confidence > threshold
↓
加入训练
```

---

## 第五步：再进入多文件版

只重点学习新知识：

```text
工程模块化
更丰富的数据增强
经典 CNN
ResNet
预训练
迁移学习
冻结参数
Fine-tuning
```

最后重新把整个项目串成一句话：

> Food-11 图像分类项目首先通过 Dataset、Transform 和 DataLoader 把 RGB 图片整理为 `[B,C,H,W]` 的 Tensor；CNN 通过卷积、BatchNorm、ReLU 和池化逐层提取图像特征，最终将特征展平并输出 11 个类别的 Logits；训练阶段使用 CrossEntropyLoss 和 AdamW 进行反向传播与参数更新，验证阶段通过 `eval()` 和 `no_grad()` 评价准确率；在此基础上还可以利用高置信度伪标签进行半监督学习，并进一步使用 ResNet、VGG 等经典网络以及预训练模型完成迁移学习。

---

# 总结

这个项目真正需要建立的不是“记住 Food-11 代码”，而是建立一套以后可以迁移到其他视觉任务中的完整思维：

```text
原始图片
   ↓
数据读取
   ↓
图像预处理 / 数据增强
   ↓
Tensor
   ↓
DataLoader
   ↓
CNN 特征提取
   ↓
分类头
   ↓
Logits
   ↓
CrossEntropyLoss
   ↓
Backward
   ↓
Optimizer
   ↓
Validation
   ↓
保存最佳模型
```

在掌握这一流程后，再学习：

```text
ResNet
Transfer Learning
Semi-supervised Learning
```

就不再是学习一套全新的训练框架，而是在已有框架上更换或增强其中某一个部分。
