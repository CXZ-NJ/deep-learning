# 从神经网络回归到 CNN：用 PyTorch 完成 Food-11 图像分类

> 对应代码：先读 `simple_class.py`，再读 `main.py` 和 `model_utils/`。  
> 适合读者：已经学过本仓库的手写线性回归和 COVID 神经网络回归，但还没有系统学过图片分类。  
> 本文目标：不默认你了解计算机视觉，从“图片为什么能交给神经网络”开始，逐步看懂数据读取、图像增强、卷积网络、交叉熵、准确率、验证、模型保存、迁移学习和半监督学习。  
> 学习建议：第一次只运行 `simple_class.py`。基础流程看懂以后，再运行模块化的 `main.py`。

---

## 目录

1. 这个项目在做什么？
2. 它和前两个项目是什么关系？
3. 回归和分类到底有什么区别？
4. Food-11 数据集是什么？
5. 先认识项目目录
6. 图片在计算机里到底是什么？
7. 高、宽、通道与 shape
8. 为什么 PyTorch 的图片 shape 是 `[C, H, W]`？
9. 为什么所有图片要变成统一大小？
10. 第一次理解图像变换 transforms
11. `RandomResizedCrop` 是什么？
12. `RandomHorizontalFlip` 是什么？
13. `AutoAugment` 是什么？
14. `ToTensor` 做了什么？
15. `Normalize` 为什么需要均值和标准差？
16. 训练集和验证集为什么使用不同变换？
17. `FoodDataset` 的职责
18. 为什么只保存图片路径，而不把所有图片一次读进内存？
19. 如何从目录名得到类别标签？
20. `__getitem__()` 如何读取一张图片？
21. 标签为什么必须是 `LongTensor`？
22. DataLoader 如何组成一个 batch？
23. CNN 为什么适合图片？
24. `Conv2d` 的五个重要参数
25. 输入通道和输出通道是什么？
26. BatchNorm、ReLU 和 MaxPool
27. `AdaptiveAvgPool2d` 解决了什么问题？
28. 为什么最后还需要全连接层？
29. 一张图片经过 SimpleCNN 时 shape 如何变化？
30. 模型输出的 11 个数字是什么？
31. Softmax 如何把输出变成概率？
32. 为什么训练时不用手动调用 Softmax？
33. `CrossEntropyLoss` 是什么？
34. 训练一个 batch 的完整过程
35. 如何计算分类准确率？
36. `model.train()` 和 `model.eval()`
37. 验证阶段为什么使用 `torch.no_grad()`？
38. 为什么保存验证集上最好的模型？
39. 修复后的 checkpoint 保存了什么？
40. 为什么训练曲线保存为图片而不是弹窗？
41. `simple_class.py` 的完整流程
42. 为什么还要拆成 `main.py + model_utils`？
43. ResNet18 和迁移学习
44. 什么是 linear probing？
45. 什么是半监督学习和伪标签？
46. 原项目中修复了哪些问题？
47. 如何安装与运行？
48. 常见报错怎么处理？
49. 初学者建议做的实验
50. 完整知识地图
51. 学完以后下一步学什么？

---

# 1. 这个项目在做什么？

假设我们给程序一张食物图片：

```text
一张披萨图片
      ↓
   神经网络
      ↓
预测它属于 Bread 类
```

Food-11 一共有 11 个类别：

| 标签 | 英文名称 | 中文理解 |
|---|---|---|
| 0 | Bread | 面包、披萨等面食 |
| 1 | Dairy product | 乳制品 |
| 2 | Dessert | 甜点 |
| 3 | Egg | 鸡蛋 |
| 4 | Fried food | 油炸食品 |
| 5 | Meat | 肉类 |
| 6 | Noodles/Pasta | 面条或意大利面 |
| 7 | Rice | 米饭 |
| 8 | Seafood | 海鲜 |
| 9 | Soup | 汤 |
| 10 | Vegetable/Fruit | 蔬菜或水果 |

模型的工作，就是给每张图片输出一个最可能的类别编号。

所以这是一个：

> 11 分类问题。

“11 分类”不是指一次输入 11 张图片，而是指答案有 11 种可能。

---

# 2. 它和前两个项目是什么关系？

三个项目并不是互相独立的。

它们组成了一条连续的学习路线：

```text
01 手写线性回归
数据 → 预测 → loss → backward → 手动 SGD
                  ↓
02 COVID 神经网络回归
Dataset → DataLoader → nn.Module → optimizer → train/val
                  ↓
03 Food-11 图像分类
图片 → transforms → CNN → 交叉熵 → accuracy
```

第一份代码让你知道“训练到底发生了什么”。

第二份代码让你知道“一个完整 PyTorch 项目怎么组织”。

第三份代码没有推翻前两份代码。它仍然遵循同一个训练骨架：

```text
取一批数据
   ↓
模型进行预测
   ↓
计算 loss
   ↓
反向传播
   ↓
优化器更新参数
```

真正新增的知识主要有四块：

1. 图片数据怎么读取和变换；
2. CNN 如何提取图片特征；
3. 分类任务如何使用交叉熵；
4. 如何使用准确率评价模型。

---

# 3. 回归和分类到底有什么区别？

COVID 项目的输出是一个连续数字：

```text
预测值 = 18.73
```

这种任务叫回归。

图片分类输出的是类别：

```text
预测类别 = 8，也就是 Seafood
```

这种任务叫分类。

二者在代码上的主要差别是：

| 内容 | COVID 回归 | Food-11 分类 |
|---|---|---|
| 模型输出 | 1 个数字 | 11 个分数 |
| 标签类型 | 浮点数 | 整数类别编号 |
| 损失函数 | MSE | CrossEntropyLoss |
| 常用评价指标 | MSE、MAE | Accuracy |
| 最终答案 | 连续数值 | 0–10 中的一个类别 |

注意：模型不会直接输出单词 `Soup`。

模型先输出 11 个分数，再选择分数最大的位置作为类别。

---

# 4. Food-11 数据集是什么？

Food-11 是一个食物图片分类数据集，共有 11 类食物。

你电脑中的完整版本采用下面的划分：

```text
training/labeled       有标签训练图片
training/unlabeled     无标签训练图片
validation             有标签验证图片
testing                无标签测试图片
```

本地完整数据共有：

```text
training     9866 张
validation    660 张
testing      3071 张
```

其中 training 又包括：

```text
有标签图片 3080 张
无标签图片 6786 张
```

为什么有些图片没有标签？

因为课程后面想让你练习半监督学习：先用有标签图片训练模型，再用模型给无标签图片猜一个标签。

第一次学习时完全可以不使用无标签图片。

---

# 5. 先认识项目目录

整理后的项目结构如下：

```text
03_food_classification/
├── simple_class.py
├── main.py
├── requirements.txt
├── 从零看懂PyTorch_Food11图像分类.md
├── model_utils/
│   ├── __init__.py
│   ├── data.py
│   ├── model.py
│   └── train.py
├── data/
│   ├── README.md
│   └── food-11/             本地放置，不上传 GitHub
├── checkpoints/
│   └── README.md
└── assets/
    └── accuracy_curve_original.png
```

建议阅读顺序：

```text
本文
 ↓
simple_class.py
 ↓
model_utils/data.py
 ↓
model_utils/model.py
 ↓
model_utils/train.py
 ↓
main.py
```

`simple_class.py` 把所有核心代码放在一个文件里。

模块化版本只是把同样的职责拆到不同文件，不代表训练原理变了。

---

# 6. 图片在计算机里到底是什么？

人看到一张图片，会说：

> 这是一碗汤。

计算机看到的不是“汤”，而是一大堆数字。

一张彩色图片可以理解成三个数字表：

```text
红色通道 R
绿色通道 G
蓝色通道 B
```

每个像素都由三个数字描述。

例如：

```text
[255, 0, 0]     很红
[0, 255, 0]     很绿
[0, 0, 255]     很蓝
[255, 255, 255] 白色
[0, 0, 0]       黑色
```

所以图片可以被保存为多维数组，也可以转换为 PyTorch Tensor。

模型并不知道“披萨”“米饭”这些人类概念。

模型只能通过大量数字之间的规律，逐渐学习哪些纹理、颜色和形状更可能对应某个类别。

---

# 7. 高、宽、通道与 shape

假设图片大小为 224×224，并且是 RGB 彩色图片。

在 PIL 或 NumPy 的常见表示中，shape 通常是：

```text
[224, 224, 3]
```

三个数字分别代表：

```text
224：高度 Height
224：宽度 Width
3：颜色通道 Channel
```

常缩写成：

```text
H × W × C
```

如果一次有 32 张图片，还需要在最前面增加 batch 维度。

---

# 8. 为什么 PyTorch 的图片 shape 是 `[C, H, W]`？

经过 `transforms.ToTensor()` 后，单张图片通常变成：

```text
[3, 224, 224]
```

也就是：

```text
C × H × W
```

DataLoader 把 32 张图片组成一个 batch 后，shape 变成：

```text
[32, 3, 224, 224]
```

四个维度分别是：

```text
32  = batch size
3   = RGB 通道
224 = 图片高度
224 = 图片宽度
```

PyTorch 的 `Conv2d` 默认就期待这种顺序：

```text
[N, C, H, W]
```

其中 N 表示一个 batch 中的图片数量。

---

# 9. 为什么所有图片要变成统一大小？

原始图片的大小不一定相同：

```text
图片 A：640×480
图片 B：300×300
图片 C：1024×768
```

DataLoader 想把多张图片堆成一个 Tensor，就要求它们的 shape 一致。

否则无法得到：

```text
[batch_size, 3, height, width]
```

所以代码把图片统一处理成 224×224。

224 也是许多经典图片模型常用的输入大小，例如 ResNet18。

---

# 10. 第一次理解图像变换 transforms

单文件版训练变换是：

```python
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
```

`Compose` 的意思是：

> 按照列表顺序，把一系列处理步骤连起来。

一张图片会依次经过：

```text
随机裁剪
  ↓
随机水平翻转
  ↓
转换成 Tensor
  ↓
标准化
```

这条流水线就叫 transform。

---

# 11. `RandomResizedCrop` 是什么？

代码：

```python
transforms.RandomResizedCrop(224)
```

它会从原图中随机选择一块区域，再缩放到 224×224。

同一张图片在不同 epoch 中，可能得到略微不同的裁剪结果。

这样做不是在制造错误，而是在告诉模型：

> 食物稍微大一点、小一点、靠左一点、靠右一点，类别仍然不变。

这种技术叫数据增强。

它可以增加训练数据的变化，降低模型死记训练图片的可能。

---

# 12. `RandomHorizontalFlip` 是什么？

代码：

```python
transforms.RandomHorizontalFlip()
```

它会以一定概率把图片左右翻转。

例如：

```text
原图：盘子在左边
翻转：盘子在右边
```

翻转以后，它仍然是同一种食物。

因此可以用来增加训练样本的变化。

验证集不能随机翻转，因为我们希望每次验证都使用稳定、可重复的数据。

---

# 13. `AutoAugment` 是什么？

模块化版本还使用：

```python
autoaugment.AutoAugment(
    policy=autoaugment.AutoAugmentPolicy.IMAGENET
)
```

它会从一组已经设计好的增强策略中选择操作，例如改变颜色、对比度或进行轻微旋转。

第一次阅读时，只需要记住：

> AutoAugment 是更丰富的训练图片增强组合。

如果你想确认基础流程，可以先读不含 AutoAugment 的 `simple_class.py`。

---

# 14. `ToTensor` 做了什么？

代码：

```python
transforms.ToTensor()
```

它主要完成两件事。

第一，把 PIL 图片转换成 PyTorch Tensor。

第二，把常见的像素范围：

```text
0 到 255
```

缩放到：

```text
0.0 到 1.0
```

shape 也会从常见的：

```text
[H, W, C]
```

变成 PyTorch 使用的：

```text
[C, H, W]
```

---

# 15. `Normalize` 为什么需要均值和标准差？

代码使用：

```python
transforms.Normalize(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)
```

这三个 mean 分别对应 R、G、B 三个通道的均值。

三个 std 分别对应三个通道的标准差。

每个通道都会进行近似下面的计算：

```text
标准化后的值 = (原值 - mean) / std
```

这和 COVID 项目中的标准化思想是一样的。

区别只是：

- COVID 项目标准化 93 个表格特征；
- 图片项目标准化 RGB 三个通道。

代码使用的是 ImageNet 常用的均值和标准差，因此也适合后面使用 ImageNet 预训练模型。

---

# 16. 训练集和验证集为什么使用不同变换？

训练集：

```text
允许随机裁剪、翻转、颜色变化
```

验证集：

```text
只统一尺寸、转 Tensor、标准化
```

原因是：

训练阶段需要给模型制造变化，提高泛化能力。

验证阶段需要稳定地衡量模型，不能让每次验证使用随机变化后的不同图片。

因此模块化代码中：

```python
self.transform = (
    train_transform
    if mode in {"train", "train_unl"}
    else eval_transform
)
```

---

# 17. `FoodDataset` 的职责

在 COVID 项目里，`CovidDataset` 负责读取 CSV。

在这里，`FoodDataset` 负责：

1. 找到训练集或验证集目录；
2. 找出各类别中的图片路径；
3. 根据目录名记录标签；
4. 需要某个样本时读取图片；
5. 对图片执行 transform；
6. 返回图片 Tensor 和标签。

它仍然继承：

```python
class FoodDataset(Dataset):
```

因此仍然需要实现：

```python
__getitem__
__len__
```

这正是第二个项目学过的知识。

---

# 18. 为什么只保存图片路径，而不把所有图片一次读进内存？

博主原代码先建立一个很大的 NumPy 数组：

```python
np.zeros((图片数量, 224, 224, 3), dtype=np.uint8)
```

然后把所有图片一次性读入内存。

完整数据有上万张图片，这种做法可能占用数 GB 内存。

修复后的代码在初始化 Dataset 时只保存：

```text
图片路径列表
标签列表
```

真正访问某个下标时，才读取对应图片：

```python
with Image.open(self.image_paths[index]) as image:
    image = image.convert("RGB")
    image_tensor = self.transform(image)
```

这种方式叫按需读取或懒加载。

优点是：

- 内存占用小很多；
- 可以训练完整数据集；
- Dataset 的职责更清楚。

代价是训练过程中会不断从磁盘读取图片。

这是正常的数据加载方式。

---

# 19. 如何从目录名得到类别标签？

Food-11 的有标签目录是：

```text
00/
01/
02/
...
10/
```

代码使用：

```python
for label in range(len(CLASS_NAMES)):
    class_dir = self.split_dir / f"{label:02d}"
```

`len(CLASS_NAMES)` 是 11，所以 label 会依次取：

```text
0, 1, 2, ..., 10
```

`f"{label:02d}"` 表示把整数显示成至少两位：

```text
0  → 00
1  → 01
9  → 09
10 → 10
```

目录 `08` 中的图片就会得到标签 8。

---

# 20. `__getitem__()` 如何读取一张图片？

核心代码：

```python
def __getitem__(self, index):
    with Image.open(self.image_paths[index]) as image:
        image = image.convert("RGB")
        image_tensor = self.transform(image)

    label = torch.tensor(self.labels[index], dtype=torch.long)
    return image_tensor, label
```

一步一步看：

```python
self.image_paths[index]
```

取得第 index 张图片的路径。

```python
Image.open(...)
```

读取图片。

```python
image.convert("RGB")
```

保证图片拥有三个颜色通道。

如果某些图片是灰度图或带透明通道，这一步可以避免通道数不一致。

```python
self.transform(image)
```

执行裁剪、Tensor 转换和标准化。

最后返回：

```text
(图片张量, 类别标签)
```

---

# 21. 标签为什么必须是 `LongTensor`？

分类标签是类别编号：

```text
0, 1, 2, ..., 10
```

`CrossEntropyLoss` 需要标签表示“正确类别的位置”，所以标签必须是整数类型。

代码写成：

```python
torch.tensor(label, dtype=torch.long)
```

`torch.long` 通常就是 64 位整数。

如果错误地把标签变成 float，经常会看到类似错误：

```text
expected scalar type Long but found Float
```

与 COVID 回归对比：

```text
回归标签：float，例如 18.73
分类标签：long，例如 8
```

---

# 22. DataLoader 如何组成一个 batch？

代码：

```python
train_loader = DataLoader(
    train_set,
    batch_size=16,
    shuffle=True,
)
```

假设每张图片的 shape 是：

```text
[3, 224, 224]
```

16 张图片组成 batch 后：

```text
images.shape = [16, 3, 224, 224]
labels.shape = [16]
```

labels 可能类似：

```text
[2, 8, 1, 5, 5, 9, ...]
```

训练集使用：

```python
shuffle=True
```

每个 epoch 都打乱顺序。

验证集使用：

```python
shuffle=False
```

保持稳定顺序。

---

# 23. CNN 为什么适合图片？

COVID 项目使用全连接层：

```python
nn.Linear(93, 128)
```

因为每条数据只是 93 个普通特征。

图片有空间结构：

- 相邻像素往往属于同一个物体；
- 边缘由附近像素的变化形成；
- 小纹理可以组合成更复杂的形状。

CNN 使用一个小窗口在图片上移动，寻找局部规律。

前面的卷积层可能逐渐学到：

```text
边缘、颜色变化、简单纹理
```

后面的卷积层可能逐渐组合成：

```text
盘子形状、面条纹理、肉块轮廓、汤的表面
```

所以 CNN 很适合处理图片。

---

# 24. `Conv2d` 的五个重要参数

代码：

```python
nn.Conv2d(
    in_channels=3,
    out_channels=64,
    kernel_size=3,
    padding=1,
)
```

重要参数如下。

## `in_channels`

输入有多少个通道。

RGB 图片有 3 个通道，所以第一层是：

```python
in_channels=3
```

## `out_channels`

这一层想得到多少张新的特征图。

```python
out_channels=64
```

表示得到 64 个输出通道。

## `kernel_size`

卷积窗口大小。

```python
kernel_size=3
```

表示使用 3×3 的窗口。

## `stride`

卷积窗口每次移动多少格。

代码没有写时，默认是 1。

## `padding`

在图片边缘补多少圈。

3×3 卷积配合 `padding=1`、`stride=1` 时，通常可以保持高宽不变。

---

# 25. 输入通道和输出通道是什么？

第一层卷积：

```text
输入：[batch, 3, 224, 224]
输出：[batch, 64, 224, 224]
```

这里的 64 不是颜色通道。

它是神经网络学习出来的 64 种特征表示。

下一层写成：

```python
nn.Conv2d(64, 128, kernel_size=3, padding=1)
```

原因是上一层输出 64 个通道，所以这一层必须接收 64 个通道。

通道的连接关系必须对上：

```text
3 → 64 → 128 → 256 → 512
```

---

# 26. BatchNorm、ReLU 和 MaxPool

一个常见卷积阶段是：

```text
Conv2d
  ↓
BatchNorm2d
  ↓
ReLU
  ↓
MaxPool2d
```

## BatchNorm2d

```python
nn.BatchNorm2d(64)
```

它对一个 batch 中的特征进行规范化处理，并且有可以学习的缩放和平移参数。

入门阶段先记住：

> BatchNorm 通常可以让训练过程更稳定。

## ReLU

```python
nn.ReLU(inplace=True)
```

它大致执行：

```text
小于 0 → 0
大于 0 → 保留
```

ReLU 给网络加入非线性能力。

如果没有非线性激活，多层线性运算组合以后仍然只是线性变换。

## MaxPool2d

```python
nn.MaxPool2d(2)
```

它通常让高和宽都缩小一半：

```text
224×224 → 112×112
112×112 → 56×56
```

这样可以减少后续计算量，同时保留局部区域中比较明显的响应。

---

# 27. `AdaptiveAvgPool2d` 解决了什么问题？

博主原模型写了：

```python
self.fc = nn.Linear(25088, 512)
```

这个 25088 来自：

```text
512 × 7 × 7
```

它与输入图片大小和池化次数紧密绑定。

只要前面尺寸稍微改变，最后就可能出现矩阵乘法 shape 不匹配。

修复后的模型使用：

```python
nn.AdaptiveAvgPool2d((1, 1))
```

它会把每个通道的空间大小统一变成 1×1：

```text
[batch, 512, H, W]
        ↓
[batch, 512, 1, 1]
```

展平后固定是：

```text
[batch, 512]
```

分类层因此可以稳定写成：

```python
nn.Linear(512, 11)
```

---

# 28. 为什么最后还需要全连接层？

卷积部分的任务是提取图片特征。

最后还需要把特征转换为 11 个类别分数：

```python
self.classifier = nn.Linear(512, num_classes)
```

输入：

```text
每张图片的 512 个特征
```

输出：

```text
11 个类别分数
```

这个最后的 Linear 常被称为分类头。

---

# 29. 一张图片经过 SimpleCNN 时 shape 如何变化？

假设 batch size 是 16：

| 阶段 | 输出 shape |
|---|---|
| 输入 | `[16, 3, 224, 224]` |
| 第一次卷积 | `[16, 64, 224, 224]` |
| 第一次池化 | `[16, 64, 112, 112]` |
| 第二次卷积 | `[16, 128, 112, 112]` |
| 第二次池化 | `[16, 128, 56, 56]` |
| 第三次卷积 | `[16, 256, 56, 56]` |
| 第三次池化 | `[16, 256, 28, 28]` |
| 第四次卷积 | `[16, 512, 28, 28]` |
| 自适应平均池化 | `[16, 512, 1, 1]` |
| flatten | `[16, 512]` |
| 分类层 | `[16, 11]` |

所以模型一次为 16 张图片分别输出 11 个分数。

---

# 30. 模型输出的 11 个数字是什么？

模型输出可能类似：

```text
[-0.8, 1.2, 0.3, -1.0, 2.7, 0.5, 0.1, -0.4, 1.8, 0.2, 0.6]
```

这些数叫 logits。

它们不是概率：

- 可以是负数；
- 不要求加起来等于 1；
- 数值越大，模型越偏向相应类别。

上面最大值是位置 4 的 `2.7`。

因此预测类别是：

```text
4 → Fried food
```

代码取最大位置：

```python
predicted = logits.argmax(dim=1)
```

---

# 31. Softmax 如何把输出变成概率？

Softmax 会把 11 个 logits 转成 11 个 0 到 1 之间的数，并让它们加起来等于 1。

代码：

```python
probabilities = torch.softmax(logits, dim=1)
```

假设结果是：

```text
[0.01, 0.08, 0.03, 0.01, 0.55, 0.04, 0.02, 0.01, 0.18, 0.02, 0.05]
```

位置 4 的概率最大，所以预测为类别 4。

半监督学习还会关心最大概率是多少，因为它用这个概率表示置信度。

---

# 32. 为什么训练时不用手动调用 Softmax？

训练代码直接写：

```python
logits = model(images)
loss = criterion(logits, labels)
```

没有先写：

```python
torch.softmax(logits, dim=1)
```

这是因为：

```python
nn.CrossEntropyLoss()
```

已经在内部组合了适合分类的对数 Softmax 和负对数似然计算。

把原始 logits 直接交给它，数值上更稳定。

所以记住：

```text
训练计算 CrossEntropyLoss：直接传 logits
展示概率或筛选伪标签：再调用 softmax
```

---

# 33. `CrossEntropyLoss` 是什么？

代码：

```python
criterion = nn.CrossEntropyLoss()
```

它会比较：

```text
模型对 11 个类别给出的分数
```

与：

```text
真实类别编号
```

如果真实类别是 8，但模型更偏向类别 2，loss 会比较大。

如果模型给类别 8 很高的分数，loss 会比较小。

训练的目标仍然是：

```text
不断调整模型参数，让 loss 下降
```

它和前两个项目的核心思想没有变化，只是损失函数换成了更适合分类的形式。

---

# 34. 训练一个 batch 的完整过程

核心代码：

```python
optimizer.zero_grad()
logits = model(images)
loss = criterion(logits, labels)
loss.backward()
optimizer.step()
```

逐步解释。

## 第一步：清空旧梯度

```python
optimizer.zero_grad()
```

PyTorch 默认会累积梯度，所以每个 batch 开始前要清空上一批梯度。

## 第二步：前向传播

```python
logits = model(images)
```

图片经过 CNN，得到 11 个类别分数。

## 第三步：计算损失

```python
loss = criterion(logits, labels)
```

比较预测分数与真实类别。

## 第四步：反向传播

```python
loss.backward()
```

计算每个可训练参数应该如何变化。

## 第五步：更新参数

```python
optimizer.step()
```

优化器根据梯度更新参数。

这仍然是第一份线性回归中的训练四连：

```text
预测 → loss → backward → 更新参数
```

---

# 35. 如何计算分类准确率？

先取得模型预测类别：

```python
predicted = logits.argmax(dim=1)
```

再和真实标签比较：

```python
predicted == labels
```

结果类似：

```text
[True, False, True, True, False]
```

统计正确数量：

```python
correct += (predicted == labels).sum().item()
```

最后：

```text
accuracy = 正确数量 / 总数量
```

例如 100 张图片预测正确 73 张：

```text
accuracy = 73 / 100 = 0.73
```

也就是 73%。

注意 accuracy 和 loss 不是一回事：

- accuracy 只关心最终类别是否正确；
- loss 还关心模型对正确类别的信心程度。

---

# 36. `model.train()` 和 `model.eval()`

训练前：

```python
model.train()
```

验证前：

```python
model.eval()
```

这两个函数不会自动帮你训练或验证。

它们是在告诉模型当前处于什么阶段。

BatchNorm、Dropout 等层在训练和验证阶段的行为不同。

因此正确顺序是：

```text
model.train()
执行训练循环
     ↓
model.eval()
执行验证循环
```

---

# 37. 验证阶段为什么使用 `torch.no_grad()`？

验证代码：

```python
model.eval()
with torch.no_grad():
    for images, labels in val_loader:
        logits = model(images)
```

验证阶段只需要计算结果，不需要反向传播。

关闭梯度记录可以：

- 减少内存占用；
- 提高运行速度；
- 避免意外进行训练操作。

但要注意：

```text
model.eval()       改变部分网络层的工作模式
torch.no_grad()    关闭梯度记录
```

它们负责不同的事情，验证时通常两个都需要。

---

# 38. 为什么保存验证集上最好的模型？

训练轮数增加时，训练集准确率往往会继续提高。

但验证集准确率不一定一直提高。

模型可能开始过度记忆训练集，这叫过拟合。

所以代码记录：

```python
best_val_accuracy = 0.0
```

每次验证以后判断：

```python
if val_accuracy > best_val_accuracy:
    保存模型
```

这样最终得到的不是“最后一个 epoch 的模型”，而是“验证集表现最好的模型”。

---

# 39. 修复后的 checkpoint 保存了什么？

博主原代码直接保存完整模型：

```python
torch.save(model, save_path)
```

修复后的代码保存一个字典：

```python
torch.save(
    {
        "epoch": epoch_index + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_accuracy": best_val_accuracy,
    },
    checkpoint_path,
)
```

各字段含义：

```text
epoch                    保存时训练到了第几轮
model_state_dict         模型参数
optimizer_state_dict     优化器状态
best_val_accuracy        当时最佳验证准确率
```

这种方式更容易明确知道文件里保存了什么，也更适合以后恢复训练。

生成的 `.pth` 文件仍然可能很大，所以 `.gitignore` 会阻止它上传 GitHub。

---

# 40. 为什么训练曲线保存为图片而不是弹窗？

博主原代码使用：

```python
plt.show()
```

在某些环境中，它会弹出窗口并阻塞程序。

修复后的代码使用：

```python
figure.savefig(output_path, dpi=150)
plt.close(figure)
```

训练过程中曲线会保存到：

```text
assets/training_curves.png
```

这样：

- 不需要手动关闭窗口；
- 远程服务器也能运行；
- 训练结束后直接打开图片查看；
- 程序不会因为窗口停在中间。

---

# 41. `simple_class.py` 的完整流程

单文件版从上到下分为：

```text
1. 导入库
2. 设置项目路径和类别名称
3. 固定随机种子
4. 定义训练/验证 transforms
5. 定义 FoodDataset
6. 定义 SimpleCNN
7. 定义 train_one_epoch
8. 定义 validate
9. 定义训练曲线保存函数
10. 读取运行参数
11. main() 中创建数据、模型、损失和优化器
12. 执行 epoch 循环
13. 保存最佳模型与曲线
```

第一次学习建议：

1. 先不运行，完整读一遍；
2. 在 `__getitem__` 中临时打印一张图片的 shape；
3. 只运行 1 个 epoch；
4. 观察 train loss、train acc、val loss、val acc；
5. 再运行更多 epoch。

---

# 42. 为什么还要拆成 `main.py + model_utils`？

代码变长以后，把所有内容放在一个文件里会越来越难找。

模块化版本按职责拆分：

```text
model_utils/data.py
负责数据路径、Dataset、transforms、DataLoader、伪标签数据

model_utils/model.py
负责 SimpleCNN、ResNet、VGG 和分类头替换

model_utils/train.py
负责训练、验证、保存 checkpoint 和绘制曲线

main.py
负责选择参数，把各部分连接起来
```

可以把它想成一家小餐厅：

```text
data.py   准备食材
model.py  定义厨具
train.py  规定烹饪流程
main.py   决定今天做什么以及使用哪些参数
```

拆分并没有增加新的训练原理，只是让文件职责更单一。

---

# 43. ResNet18 和迁移学习

模块化版本支持：

```bash
python main.py --model resnet18
```

ResNet18 是一个经典卷积网络。

如果不加 `--pretrained`：

```text
使用 ResNet18 架构
参数从随机状态开始训练
```

如果加上：

```bash
python main.py --model resnet18 --pretrained
```

程序会使用已经在 ImageNet 上训练过的参数。

这叫迁移学习。

直观理解：

> 模型以前已经学过边缘、纹理和常见形状，现在把这些基础能力迁移到 Food-11。

预训练模型原本输出 ImageNet 的类别。

代码会把最后一层替换为：

```python
model.fc = nn.Linear(model.fc.in_features, 11)
```

让它输出 Food-11 的 11 个类别。

第一次使用预训练权重时，torchvision 可能需要联网下载模型文件。

---

# 44. 什么是 linear probing？

运行：

```bash
python main.py \
  --model resnet18 \
  --pretrained \
  --linear-probing
```

代码先冻结原模型参数：

```python
for parameter in model.parameters():
    parameter.requires_grad = False
```

然后替换最后的分类头。

新的分类头默认可以训练。

所以训练时：

```text
ResNet18 原来的特征提取部分：不更新
新的 11 分类层：更新
```

这种方法训练更快、需要的显存更少，适合先验证预训练特征是否有用。

如果不冻结参数，而是让整个预训练模型继续训练，通常叫 fine-tuning，也就是微调。

---

# 45. 什么是半监督学习和伪标签？

有标签图片：

```text
图片 + 正确答案
```

无标签图片：

```text
只有图片，没有答案
```

伪标签的基本流程是：

```text
先用有标签数据训练模型
          ↓
模型预测无标签图片
          ↓
只保留置信度很高的预测
          ↓
把预测类别当作临时标签
          ↓
加入后续训练
```

运行时加：

```bash
python main.py --semi
```

默认设置是：

```text
验证准确率至少达到 0.70
每 3 个 epoch 更新一次伪标签
只接收最大概率至少为 0.99 的图片
```

为什么阈值设置很高？

因为伪标签可能是错的。

如果把大量错误答案重新喂给模型，模型可能越学越错。

第一次学习时不要开启 `--semi`。

先完全看懂普通有监督训练，再把半监督作为进阶内容。

---

# 46. 原项目中修复了哪些问题？

这次整理保留了博主代码的学习思路，但修复了会影响运行和上传的问题。

## 46.1 删除旧电脑绝对路径

原代码使用：

```text
F:\pycharm\beike\classification\...
```

这个路径只在原电脑有效。

修复后默认以当前项目目录为基础：

```python
PROJECT_DIR = Path(__file__).resolve().parent
```

也可以通过 `--data-root` 指定任意数据目录。

## 46.2 增加 main 入口保护

现在训练只会在直接运行文件时启动：

```python
if __name__ == "__main__":
    main()
```

导入模块时不会突然开始读取数据和训练。

## 46.3 修复半监督函数参数数量

原 `train.py` 给 `get_semi_loader` 传了五个参数，但函数只接收四个参数。

修复后统一为：

```python
get_pseudo_label_loader(
    unlabeled_loader,
    model,
    device,
    threshold,
)
```

## 46.4 修复无标签 loader 变量混乱

原代码同时使用 `semi_loader`、`no_label_Loader` 等相近名称，容易把“原始无标签数据”和“伪标签数据”混在一起。

修复后区分为：

```text
unlabeled_loader   原始无标签图片
pseudo_loader      经过模型筛选后的伪标签图片
```

## 46.5 验证集不再打乱

现在只有训练集使用：

```python
shuffle=True
```

验证集、测试集和无标签数据均使用稳定顺序。

## 46.6 不再一次性把所有图片装入内存

Dataset 只保存路径，需要某张图片时再读取。

## 46.7 明确处理多余的 `11` 类目录

Food-11 的合法目录是 `00–10`。

你原来的 sample 数据中还有 `11`，修复后的代码会显示提示并忽略它，避免悄悄造成类别数混乱。

建议以后重新制作 sample 数据时，只保留 `00–10`。

## 46.8 修复最佳准确率比较

原模块化代码在配置中传入 `max_acc`，进入函数后又把它重设为 0，而且一部分代码拿“正确数量”与“比例”比较。

修复后始终使用 0 到 1 之间的验证准确率：

```python
if val_accuracy > best_val_accuracy:
```

## 46.9 不在验证阶段保存 GPU 输出列表

原代码不断把 `val_pred` 加进 `val_rel`，但后面没有使用，可能增加显存占用。

修复后删除了这部分。

## 46.10 使用 PyTorch Tensor 计算正确数量

不再把每批输出转换成 NumPy 后计算：

```python
(logits.argmax(dim=1) == labels).sum().item()
```

这样更直接，也减少 CPU 和 GPU 之间不必要的数据转换。

## 46.11 使用新的 torchvision 权重写法

修复后的 ResNet18 使用：

```python
models.ResNet18_Weights.DEFAULT
```

不再使用逐渐弃用的 `pretrained=True` 参数。

## 46.12 删除没有使用的依赖

原文件导入了 `cv2`、`sklearn`、`timm` 等当前代码没有使用的内容。

修复后删除这些导入，降低安装难度。

## 46.13 模型和大数据不上传 GitHub

完整数据约 914.6 MiB，原模型目录约 576.9 MiB。

其中一个模型文件超过 500 MB，普通 GitHub 仓库无法直接接收。

修复后的 `.gitignore` 会忽略数据和 `.pth` 文件。

仓库只保存代码、笔记、数据下载说明和小型示例图。

---

# 47. 如何安装与运行？

## 47.1 安装依赖

进入项目目录：

```bash
cd 03_food_classification
```

安装：

```bash
pip install -r requirements.txt
```

如果你需要 GPU 版本的 PyTorch，应按照自己 CUDA 环境选择 PyTorch 官方提供的安装命令。

## 47.2 放置数据

默认位置：

```text
03_food_classification/data/food-11/
```

完整结构见：

```text
data/README.md
```

也可以不复制数据，直接指向电脑上的现有目录。

## 47.3 第一次只运行 1 个 epoch

```bash
python simple_class.py \
  --data-root "D:/Desktop/李哥深度学习/第四五节，分类代码/food_classification/food-11_sample" \
  --epochs 1
```

Windows PowerShell 中也可以写成一行：

```powershell
python simple_class.py --data-root "D:/Desktop/李哥深度学习/第四五节，分类代码/food_classification/food-11_sample" --epochs 1
```

## 47.4 使用完整数据训练

```powershell
python simple_class.py --data-root "D:/Desktop/李哥深度学习/第四五节，分类代码/food_classification/food-11" --epochs 10
```

## 47.5 运行模块化 SimpleCNN

```powershell
python main.py --data-root "D:/Desktop/李哥深度学习/第四五节，分类代码/food_classification/food-11" --model simple_cnn --epochs 10
```

## 47.6 使用预训练 ResNet18

```powershell
python main.py --data-root "D:/Desktop/李哥深度学习/第四五节，分类代码/food_classification/food-11" --model resnet18 --pretrained --epochs 10
```

## 47.7 最后再尝试半监督

```powershell
python main.py --data-root "D:/Desktop/李哥深度学习/第四五节，分类代码/food_classification/food-11" --model resnet18 --pretrained --semi --epochs 15
```

---

# 48. 常见报错怎么处理？

## 找不到数据目录

错误类似：

```text
FileNotFoundError: 找不到数据目录
```

检查 `--data-root` 指向的是 `food-11` 根目录，而不是直接指向 `training/labeled`。

正确：

```text
--data-root D:/.../food-11
```

错误：

```text
--data-root D:/.../food-11/training/labeled
```

## 缺少类别目录

如果提示缺少 `00`、`01` 等目录，说明数据结构不完整。

训练集和验证集都应包含 `00–10`。

## CUDA out of memory

表示显存不足。

先减小 batch size：

```powershell
python main.py --batch-size 8
```

还不够就使用：

```powershell
python main.py --batch-size 4
```

## 训练速度很慢

确认输出中的：

```text
device: cuda
```

如果是 `device: cpu`，模型正在使用 CPU。

CPU 也能运行，但图片分类会明显更慢。

第一次可以使用 sample 数据和 1 个 epoch 验证流程。

## 下载预训练权重失败

去掉：

```text
--pretrained
```

就不会下载权重，但模型将从随机参数开始训练。

## 标签类型错误

如果 CrossEntropyLoss 提示需要 Long，检查标签是否使用：

```python
dtype=torch.long
```

## shape 不匹配

先打印：

```python
print(images.shape)
print(logits.shape)
print(labels.shape)
```

正常情况下应类似：

```text
images: [batch, 3, 224, 224]
logits: [batch, 11]
labels: [batch]
```

---

# 49. 初学者建议做的实验

不要一次修改很多东西。

每次只改一个变量，并记录结果。

## 实验 1：修改 batch size

比较：

```text
batch_size = 8
batch_size = 16
batch_size = 32
```

观察：

- 训练速度；
- 显存占用；
- loss 波动；
- 验证准确率。

## 实验 2：关闭水平翻转

暂时删除：

```python
transforms.RandomHorizontalFlip()
```

观察验证准确率是否变化。

## 实验 3：改变学习率

比较：

```text
1e-3
1e-4
1e-5
```

学习率过大，loss 可能剧烈波动。

学习率过小，loss 下降可能很慢。

## 实验 4：去掉 BatchNorm

只在复制的实验代码中暂时去掉 BatchNorm，观察训练是否更加不稳定。

## 实验 5：减少卷积通道

把：

```text
64 → 128 → 256 → 512
```

改为：

```text
32 → 64 → 128 → 256
```

观察速度、参数量与准确率。

## 实验 6：SimpleCNN 对比 ResNet18

分别运行：

```powershell
python main.py --model simple_cnn --epochs 10
python main.py --model resnet18 --epochs 10
```

保证其他参数尽量一致。

## 实验 7：预训练对比随机初始化

比较：

```powershell
python main.py --model resnet18 --epochs 10
python main.py --model resnet18 --pretrained --epochs 10
```

## 实验 8：只训练分类头

```powershell
python main.py --model resnet18 --pretrained --linear-probing --epochs 10
```

观察训练速度和准确率。

## 实验 9：查看不同置信度阈值

学完普通训练后，再比较：

```text
0.90
0.95
0.99
```

阈值越高，伪标签数量通常越少，但可信度通常更高。

---

# 50. 完整知识地图

```text
图片文件
  ↓ PIL.Image.open
RGB 图片
  ↓ transforms
统一尺寸、增强、Tensor、标准化
  ↓ Dataset.__getitem__
[3, 224, 224] + 整数标签
  ↓ DataLoader
[batch, 3, 224, 224] + [batch]
  ↓ CNN
卷积提取局部特征
  ↓
BatchNorm + ReLU + Pool
  ↓
更高层图片特征
  ↓ AdaptiveAvgPool + flatten
[batch, 512]
  ↓ Linear
[batch, 11] logits
  ↓ CrossEntropyLoss
一个 loss 数字
  ↓ backward
计算梯度
  ↓ optimizer.step
更新参数
  ↓
重复多个 batch 和 epoch
  ↓
验证集 accuracy
  ↓
保存最佳 checkpoint
```

半监督是在这条主线之外增加：

```text
无标签图片
  ↓ 当前模型预测
概率和类别
  ↓ 置信度筛选
伪标签数据集
  ↓
加入训练
```

---

# 51. 学完以后下一步学什么？

如果你能解释下面这些问题，说明你已经真正理解了这个项目：

1. 图片为什么可以表示为 Tensor？
2. `[32, 3, 224, 224]` 中每个数字代表什么？
3. 为什么训练集和验证集的 transform 不同？
4. Dataset 和 DataLoader 分别负责什么？
5. 卷积层的输入通道和输出通道是什么？
6. 池化为什么会改变图片特征图大小？
7. 模型为什么输出 11 个数？
8. logits 和概率有什么区别？
9. 为什么 CrossEntropyLoss 不需要手动 Softmax？
10. accuracy 是怎样计算的？
11. 为什么验证时需要 `eval()` 和 `no_grad()`？
12. 为什么保存最佳验证模型而不是最后一个模型？
13. 预训练与从零训练有什么区别？
14. 伪标签为什么可能帮助训练，也可能伤害训练？

下一步可以学习：

```text
加载 checkpoint 做单张图片预测
        ↓
混淆矩阵与每类准确率
        ↓
学习率调度器
        ↓
早停 Early Stopping
        ↓
更系统的迁移学习和微调
        ↓
目标检测 / 图像分割
```

---

# 一句话总结

Food-11 图片分类虽然比 COVID 回归多了图片变换、CNN、交叉熵和准确率，但训练本质没有变化：

$$
\boxed{
图片
\rightarrow
模型输出类别分数
\rightarrow
计算交叉熵
\rightarrow
反向传播
\rightarrow
更新参数
\rightarrow
验证准确率
}
$$

真正需要掌握的不是背住每一行代码，而是能把下面这条主线用自己的话讲出来：

```text
数据怎么进来
模型怎么预测
loss 怎么计算
参数怎么更新
效果怎么验证
最佳模型怎么保存
```

理解这条主线以后，换成 ResNet、VGG，甚至以后学习目标检测，都会更容易找到代码中的共同结构。
