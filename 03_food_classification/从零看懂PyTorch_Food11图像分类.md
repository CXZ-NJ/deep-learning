# 从零看懂 PyTorch Food-11 图像分类

> 对应代码：`simple_class.py`、`main.py`、`model_utils/data.py`、`model_utils/model.py`、`model_utils/train.py`  
> 适合读者：没有系统学过机器学习、深度学习，概率论也忘得差不多，但已经完成前两个回归项目的初学者。  
> 文档目标：不是把代码包装成复杂的科研框架，而是把你现在仓库里的代码逐层讲清楚。  
> 重要说明：本文只说明当前代码实际做了什么、修复过什么以及仍有哪些局限，不加入与本项目无关的扩展内容。

---

## 阅读这份文档后，你应该能回答什么？

这份文档默认你已经读过前两个项目，所以不再从头解释 Tensor、梯度、`backward()`、Dataset、DataLoader、`nn.Module`、Epoch、学习率、训练集和验证集等通用概念。

本文只重点解释 Food-11 相比前两个项目新增或发生变化的内容。

学完以后，你应该能用自己的话回答：

1. Food-11 数据目录为什么同时包含有标签、无标签、验证和测试图片？
2. 一张图片怎样从 HWC 变成 CHW，一个 batch 为什么是 BCHW？
3. 训练图片为什么需要随机裁剪、旋转、翻转等数据增强？
4. Food-11 的 Dataset 与前一个表格 Dataset 有什么不同？
5. 卷积、BatchNorm 和池化怎样改变图片特征的 shape？
6. 自定义 CNN 为什么最后得到 25088 个特征？
7. 为什么分类模型输出 11 个 logits？
8. Softmax、argmax 与 CrossEntropyLoss 分别在什么地方使用？
9. 自定义 CNN、VGG、ResNet、预训练、微调和线性探测是什么关系？
10. 当前代码怎样计算分类准确率并选择最佳模型？
11. 伪标签怎样把无标签图片变成临时训练数据？
12. `simple_class.py` 与 `main.py + model_utils` 两套代码怎样运行？
13. 原始代码修改了什么，当前版本还存在哪些局限？

如果这些问题大部分能够回答，就说明你已经抓住了这个图像分类项目的新增知识。

---

## 目录

### 第一部分：先建立全局认识
1. 这个项目到底在做什么？
2. 它和前两个项目有什么联系？
3. Food-11 的 11 个类别
4. 有标签训练集、无标签训练集、验证集和测试集
5. 当前项目目录中每个文件负责什么？
6. 两套代码应该按什么顺序阅读？

### 第二部分：图片进入神经网络之前
7. 图片在计算机里是什么？
8. HWC、CHW 和 BCHW
9. 像素类型与数值范围
10. 为什么统一为 224×224？
11. transform 是什么？
12. 训练集的数据增强
13. 验证集为什么不使用随机增强？
14. Normalize 是什么？当前代码有没有使用？

### 第三部分：Dataset 与 DataLoader
15. `food_Dataset.__init__()` 逐步解释
16. 如何读取有标签图片？
17. 如何从文件夹编号得到类别标签？
18. 如何读取无标签图片？
19. `__getitem__()` 与 `__len__()`
20. 图片经过 DataLoader 后为什么变成 BCHW？
21. 为什么无标签 DataLoader 必须保持顺序？
22. 模块化版本 `foodDataset` 的四种模式
23. 为什么 Dataset 会同时返回变换后的图片和原图？
24. 当前读取方式为什么占内存？

### 第四部分：卷积神经网络
25. 什么是卷积？
26. `Conv2d(3, 64, 3, 1, 1)` 逐项解释
27. 卷积输出尺寸怎样计算？
28. BatchNorm2d 是什么？
29. MaxPool2d 是什么？
30. `nn.Sequential` 为什么有用？
31. 自定义 CNN 中 shape 的完整变化
32. 为什么要展平？
33. 全连接层怎样完成分类？

### 第五部分：分类输出与损失函数
34. 为什么输出 11 个 logits？
35. Softmax 怎样把分数变成概率？
36. argmax 怎样得到预测类别？
37. CrossEntropyLoss 的直观理解
38. 为什么标签必须是 LongTensor？
39. 为什么训练时不需要手动先做 Softmax？

### 第六部分：模型选择与迁移学习
40. 自定义 CNN、VGG 和 ResNet 的关系
41. 什么是预训练模型？
42. 为什么要替换最后的分类层？
43. 从头训练、微调和线性探测
44. `initialize_model()` 如何选择不同模型？
45. 当前两套入口实际使用什么模型？

### 第七部分：训练与验证
46. 图像分类训练与前两个项目哪里相同、哪里不同？
47. 为什么这个项目使用 AdamW？
48. 一个图片 batch 在训练代码中怎样流动？
49. 分类准确率怎样计算？
50. 怎样阅读这个分类项目的曲线？

### 第八部分：半监督学习与伪标签
51. 为什么要使用无标签图片？
52. 伪标签的完整流程
53. `semiDataset` / `noLabDataset` 在做什么？
54. 置信度阈值为什么设置得很高？
55. 伪标签什么时候加入训练？
56. 半监督学习有什么风险？

### 第九部分：两套代码逐层串联
57. `simple_class.py` 从上到下怎样运行？
58. `main.py` 从上到下怎样运行？
59. `trainpara` 字典中每个参数是什么意思？
60. `model_utils/data.py` 的职责
61. `model_utils/model.py` 的职责
62. `model_utils/train.py` 的职责
63. 单文件版与模块化版的差别

### 第十部分：修复、局限、运行与实验
64. 相对原始教学代码修改了什么？
65. 哪些核心学习代码没有改？
66. 当前代码仍然有哪些局限？
67. 怎样准备数据并运行？
68. 常见报错与排查顺序
69. 只针对本项目新增知识的实验
70. 本项目新增术语表
71. 最后总结

---

# 第一部分：先建立全局认识

# 1. 这个项目到底在做什么？

这个项目的任务是：

> 输入一张食物图片，让模型判断它属于 Food-11 中的哪一种食物类别。

输入是一张图片，例如一张面包照片。

输出不是文字“面包”，而是一个类别编号：

```text
0
```

程序再把类别编号 0 对应回 Bread，也就是面包。

完整流程可以先记成：

```text
硬盘中的图片文件
        ↓
PIL 读取图片
        ↓
统一为 224×224
        ↓
进行随机裁剪、旋转等数据增强
        ↓
转换为 PyTorch Tensor
        ↓
DataLoader 组成一个 batch
        ↓
送入 CNN / VGG / ResNet
        ↓
输出 11 个类别分数
        ↓
CrossEntropyLoss 计算预测错误程度
        ↓
反向传播计算梯度
        ↓
优化器更新模型参数
        ↓
在验证集上计算准确率
        ↓
保存表现最好的模型
```

这里最重要的不是一次记住所有名词，而是先看出它仍然遵循深度学习的共同主线：

```text
数据 → 模型 → 预测 → 损失 → 梯度 → 更新 → 验证
```

---

# 2. 它和前两个项目有什么联系？

你的三个项目并不是三个互不相关的代码文件，而是一条逐渐增加难度的路线。

```text
01 手写线性回归
    ↓
认识 Tensor、模型参数、loss、backward、梯度、SGD

02 COVID 神经网络回归
    ↓
认识 Dataset、DataLoader、nn.Module、训练集、验证集、优化器

03 Food-11 图像分类
    ↓
认识图片 Tensor、CNN、数据增强、交叉熵、准确率、迁移学习、伪标签
```

三个项目的训练核心其实非常接近。

Food-11 中仍然有：

```python
pred = model(x)
bat_loss = loss(pred, target)
bat_loss.backward()
optimizer.step()
optimizer.zero_grad()
```

它们分别表示：

```text
模型根据输入进行预测
        ↓
比较预测与正确答案
        ↓
根据错误计算梯度
        ↓
更新参数
        ↓
清空本轮梯度
```

第三个项目难度增加，主要不是因为训练逻辑完全改变了，而是因为：

- 输入从表格数字变成了图片；
- 模型从全连接网络变成了卷积神经网络；
- 输出从一个连续数字变成了 11 个类别分数；
- 损失函数从 MSE 变成了交叉熵；
- 评价指标增加了准确率；
- 又加入了迁移学习和半监督学习。

---

# 3. Food-11 的 11 个类别

本项目用 0 到 10 表示 11 类食物。

| 文件夹 | 数字标签 | 英文类别 | 中文理解 |
|---|---:|---|---|
| `00` | 0 | Bread | 面包 |
| `01` | 1 | Dairy product | 乳制品 |
| `02` | 2 | Dessert | 甜点 |
| `03` | 3 | Egg | 鸡蛋 |
| `04` | 4 | Fried food | 油炸食品 |
| `05` | 5 | Meat | 肉类 |
| `06` | 6 | Noodles/Pasta | 面条或意大利面 |
| `07` | 7 | Rice | 米饭 |
| `08` | 8 | Seafood | 海鲜 |
| `09` | 9 | Soup | 汤 |
| `10` | 10 | Vegetable/Fruit | 蔬菜或水果 |

为什么标签要用整数，而不是直接把“面包”传给神经网络？

因为计算机中的模型和损失函数最终都需要用数字进行运算。

例如：

```text
面包 → 0
海鲜 → 8
汤   → 9
```

模型输出 11 个分数后，程序找出最大分数的位置，再把位置映射回类别名称。

当前代码中有：

```python
for i in tqdm(range(11)):
```

`range(11)` 产生：

```text
0, 1, 2, ..., 10
```

所以即使 sample 数据中额外存在 `11` 文件夹，它也不会被当前代码读取。

---

# 4. 有标签训练集、无标签训练集、验证集和测试集

Food-11 数据大致按照下面的方式组织：

```text
food-11_sample/
├── training/
│   ├── labeled/
│   │   ├── 00/
│   │   ├── 01/
│   │   ├── ...
│   │   └── 10/
│   └── unlabeled/
│       └── 00/
├── validation/
│   ├── 00/
│   ├── 01/
│   ├── ...
│   └── 10/
└── testing/
    └── 00/
```

## 4.1 有标签训练集

位置：

```text
training/labeled
```

这里的图片具有正确类别。

用途是：

```text
图片输入模型
→ 得到预测
→ 与正确标签比较
→ 计算 loss
→ 更新模型参数
```

## 4.2 无标签训练集

位置：

```text
training/unlabeled
```

这里有图片，但没有人工提供的类别答案。

普通监督学习不能直接计算：

```text
预测与正确标签之间的 loss
```

因为正确标签不存在。

当前项目后面通过“伪标签”尝试使用这些数据。

## 4.3 验证集

位置：

```text
validation
```

验证集也有正确标签，但它不用于更新参数。

它的作用是回答：

> 模型在没有参与训练的数据上表现怎么样？

如果只看训练集，模型可能只是记住了训练图片。

## 4.4 测试集

位置：

```text
testing
```

测试集常常没有公开答案，用于最后生成预测结果。

当前代码虽然在 `data.py` 中准备了 `test` 读取模式，但两个训练入口都没有完成测试集预测文件的导出。

因此当前项目真正完成的是：

```text
训练 + 验证 + 保存模型
```

还没有完成完整的：

```text
加载最佳模型 + 测试集推理 + 输出提交文件
```

---

# 5. 当前项目目录中每个文件负责什么？

当前核心目录可以理解为：

```text
03_food_classification/
├── simple_class.py
├── main.py
├── model_utils/
│   ├── data.py
│   ├── model.py
│   └── train.py
├── data/
│   └── README.md
├── model_save/
│   └── README.md
├── assets/
├── checkpoints/
│   └── README.md
└── 从零看懂PyTorch_Food11图像分类.md
```

## `simple_class.py`

单文件版本。

它把下面这些内容都写在一个文件中：

- 随机种子；
- 图像变换；
- Dataset；
- DataLoader；
- 自定义 CNN；
- 伪标签；
- 训练；
- 验证；
- 保存模型；
- 绘制曲线；
- 启动程序。

优点是第一次学习时可以从上往下看。

缺点是文件较长，后期不方便维护。

## `main.py`

模块化版本的入口。

主要负责：

- 固定随机种子；
- 选择模型；
- 设置类别数；
- 设置 batch size、学习率和 epoch；
- 创建 DataLoader；
- 创建优化器；
- 组织训练参数；
- 调用训练函数。

## `model_utils/data.py`

负责与数据有关的内容：

- 图像增强；
- 读取训练、验证、测试、无标签图片；
- Dataset；
- DataLoader；
- 生成伪标签数据集；
- 显示样本图片。

## `model_utils/model.py`

负责与模型有关的内容：

- 自定义 CNN；
- VGG；
- ResNet；
- AlexNet；
- GoogLeNet；
- DenseNet；
- SqueezeNet；
- Inception；
- 修改不同模型的最后分类层；
- 控制参数是否参与训练。

## `model_utils/train.py`

负责：

- 有标签数据训练；
- 伪标签数据训练；
- 验证；
- 计算 loss；
- 计算准确率；
- 保存最佳模型；
- 绘制 loss 和 accuracy 曲线。

## `data/README.md`

只保存数据下载和放置说明。

真正的数据集体积较大，不应该上传 GitHub。

## `model_save/README.md`

说明训练生成的模型保存在这里。

实际的 `.pth` 模型通常较大，不应该直接提交到普通 Git 仓库。

## `assets/`

用于保存 README 中要展示的图片，例如训练曲线或运行截图。

当前目录为空时，GitHub 不会展示这个空目录。

## `checkpoints/README.md`

需要特别注意：当前 Python 代码实际把模型保存到 `model_save/`，没有使用 `checkpoints/`。

而且当前代码使用的是：

```python
torch.save(model, save_path)
```

它保存的是完整模型对象，并不是 `checkpoints/README.md` 中描述的“只保存参数、优化器状态和 epoch”。

因此 `checkpoints/README.md` 目前与真实代码不一致。学习当前程序时可以忽略它。

---

# 6. 两套代码应该按什么顺序阅读？

推荐顺序：

```text
第一遍：simple_class.py
第二遍：main.py
第三遍：model_utils/data.py
第四遍：model_utils/model.py
第五遍：model_utils/train.py
```

为什么先看 `simple_class.py`？

因为它把完整路线写在一个文件中，阅读时不需要不断跳转。

第一遍只寻找四块：

```text
数据在哪里？
模型在哪里？
训练在哪里？
程序从哪里开始？
```

第二遍再看模块化版本：

```text
main.py
  ├── 调用 data.py 读取数据
  ├── 调用 model.py 创建模型
  └── 调用 train.py 开始训练
```

这样就能理解：模块化只是把原来集中在一起的内容拆到不同文件，并没有发明另一种训练原理。

---

# 第二部分：图片进入神经网络之前

# 7. 图片在计算机里是什么？

人眼看到的是一张完整照片，计算机看到的是许多像素数字。

彩色图片通常由三个颜色通道组成：

```text
R：Red，红色
G：Green，绿色
B：Blue，蓝色
```

假设图片大小是 224×224，则可以把它理解为三个 224×224 的数字表格叠在一起。

```text
红色通道：224×224
绿色通道：224×224
蓝色通道：224×224
```

图片在 NumPy 中常见 shape 是：

```text
[224, 224, 3]
```

分别表示：

```text
高度 H
宽度 W
通道 C
```

这叫 HWC 排列。

---

# 8. HWC、CHW 和 BCHW

## 8.1 HWC

PIL 图片转为 NumPy 数组以后，常见排列是：

```text
[Height, Width, Channel]
```

也就是：

```text
[224, 224, 3]
```

## 8.2 CHW

PyTorch 卷积层处理单张图片时，通常需要：

```text
[Channel, Height, Width]
```

也就是：

```text
[3, 224, 224]
```

`transforms.ToTensor()` 会帮助完成 HWC 到 CHW 的变换。

## 8.3 BCHW

DataLoader 把多张图片组成 batch 后，shape 变为：

```text
[Batch, Channel, Height, Width]
```

例如 batch size 是 16：

```text
[16, 3, 224, 224]
```

四个维度分别表示：

```text
16  ：本批有 16 张图片
3   ：RGB 三个通道
224 ：图片高度
224 ：图片宽度
```

这是理解图片张量时非常关键的一点。

---

# 9. 像素类型与数值范围

读取图片时，当前代码建立数组：

```python
np.zeros((len(file_list), HW, HW, 3), dtype=np.uint8)
```

`uint8` 表示无符号 8 位整数，取值范围是：

```text
0 到 255
```

例如：

```text
0   表示颜色强度最低
255 表示颜色强度最高
```

`ToTensor()` 之后通常会：

1. 把数据变成 PyTorch Tensor；
2. 把排列从 HWC 改为 CHW；
3. 把 `uint8` 的 0～255 缩放成浮点数 0～1。

因此一张图片可能发生：

```text
NumPy:  [224, 224, 3], uint8, 0～255
   ↓ ToTensor
Tensor: [3, 224, 224], float32, 0～1
```

---

# 10. 为什么统一为 224×224？

真实图片大小可能不同：

```text
320×240
640×480
1024×768
```

但是一个 batch 中的 Tensor 通常必须具有统一 shape，才能叠在一起输入模型。

当前代码设置：

```python
HW = 224
```

读取时又执行：

```python
img = img.resize((HW, HW))
```

所以图片先被统一成：

```text
224×224
```

224 也是 VGG、ResNet 等经典模型常用的输入尺寸。

需要注意：直接 resize 成正方形可能改变原图长宽比例，使图像发生拉伸。

训练变换中的 `RandomResizedCrop(224)` 又会随机裁剪并调整到 224×224，用于增加训练数据的变化。

---

# 11. transform 是什么？

transform 可以理解为：

> 图片送入模型前需要经过的一连串处理步骤。

当前单文件版本写法是：

```python
train_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224),
        transforms.RandomRotation(50),
        transforms.ToTensor()
    ]
)
```

`Compose` 的意思是按顺序组合操作：

```text
NumPy 图片
→ 转为 PIL 图片
→ 随机裁剪
→ 随机旋转
→ 转为 Tensor
```

每次 `Dataset.__getitem__()` 取图片时，都会执行这些操作。

因此同一张训练图片在不同 epoch 中，可能产生不同的随机版本。

---

# 12. 训练集的数据增强

数据增强不是凭空创造新的真实数据，而是对已有图片进行合理变化。

目的可以理解为：

> 不让模型只记住某张图片的固定位置、角度和裁剪方式。

## 12.1 `ToPILImage()`

```python
transforms.ToPILImage()
```

把 NumPy 数组转换成 PIL 图片，方便后续使用 torchvision 的图像变换。

## 12.2 `RandomResizedCrop(224)`

```python
transforms.RandomResizedCrop(224)
```

从图片中随机选择一个区域，再调整为 224×224。

模型不能总依赖食物位于图片正中央。

## 12.3 `RandomRotation(50)`

```python
transforms.RandomRotation(50)
```

在一定角度范围内随机旋转图片。

这里的 50 度比较大，可能产生明显倾斜。

是否有帮助需要通过验证集实验判断，并不是越强越好。

## 12.4 `RandomHorizontalFlip()`

模块化版本包含：

```python
transforms.RandomHorizontalFlip()
```

它会以一定概率把图片左右翻转。

对很多食物图片来说，左右翻转后类别仍然不变，所以是合理增强。

## 12.5 `AutoAugment()`

模块化版本还包含：

```python
autoaugment.AutoAugment()
```

它会按照预设策略组合多种图像增强操作。

对于初学者，先知道它是“自动组合的数据增强”即可，不需要马上掌握内部全部策略。

## 12.6 `ToTensor()`

```python
transforms.ToTensor()
```

完成图片到 Tensor 的转换，使图片能够进入 PyTorch 模型。

---

# 13. 验证集为什么不使用随机增强？

当前验证变换比较简单：

```python
val_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ToTensor()
    ]
)
```

模块化版本的验证集使用：

```python
test_transform = transforms.Compose([
    transforms.ToTensor(),
])
```

验证集的任务是稳定、公平地评价当前模型。

如果每次验证都随机旋转、随机裁剪，评价对象会不断变化，准确率也会多一层随机波动。

因此通常：

```text
训练集：允许随机增强
验证集：使用固定、确定的处理
测试集：使用固定、确定的处理
```

验证集不随机增强，不代表它完全不需要预处理。

尺寸调整、Tensor 转换、标准化等固定处理仍然可以使用。

---

# 14. Normalize 是什么？当前代码有没有使用？

模块化代码中定义了：

```python
imagenet_norm = [
    [0.485, 0.456, 0.406],
    [0.229, 0.224, 0.225]
]
```

训练变换中还保留了注释：

```python
# transforms.Normalize(
#     [0.485, 0.456, 0.406],
#     [0.229, 0.224, 0.225]
# )
```

Normalize 可以把每个通道按下面思路变换：

```text
标准化后的值 = (原值 - 均值) / 标准差
```

如果使用 ImageNet 预训练模型，通常应该采用与预训练时匹配的输入预处理。

但当前代码的 Normalize 被注释掉了，所以实际没有执行。

这意味着：

- `main.py` 使用 `use_pretrained=False` 时，主要影响相对小一些；
- `simple_class.py` 使用预训练 VGG 时，输入分布与模型预训练时不完全一致，可能影响迁移学习效果。

这是“当前代码的局限”，不是你现在必须马上修改的内容。

---

# 第三部分：Dataset 与 DataLoader

# 15. `food_Dataset.__init__()` 逐步解释

单文件版本中：

```python
class food_Dataset(Dataset):
    def __init__(self, path, mode="train"):
        self.mode = mode
```

`path` 是数据路径。

`mode` 表示当前数据用途，例如：

```text
train：有标签训练数据
val：验证数据
semi：无标签数据
```

接下来：

```python
if mode == "semi":
    self.X = self.read_file(path)
else:
    self.X, self.Y = self.read_file(path)
    self.Y = torch.LongTensor(self.Y)
```

意思是：

- 无标签数据只读取图片 `X`；
- 有标签数据读取图片 `X` 和标签 `Y`；
- 分类标签转换为 `LongTensor`。

最后选择 transform：

```python
if mode == "train":
    self.transform = train_transform
else:
    self.transform = val_transform
```

所以：

```text
train → 随机增强
val   → 固定变换
semi  → 这里先使用固定变换进行模型预测
```

生成伪标签以后，新的 `semiDataset` 会再对被选中的原图使用训练增强。

---

# 16. 如何读取有标签图片？

核心循环：

```python
for i in tqdm(range(11)):
    file_dir = path + "/%02d" % i
    file_list = os.listdir(file_dir)
```

## 16.1 `range(11)`

依次遍历 11 个类别：

```text
0, 1, 2, ..., 10
```

## 16.2 `"%02d" % i`

把整数格式化为两位数字：

```text
0  → 00
1  → 01
...
10 → 10
```

这样就能对应数据文件夹名称。

## 16.3 `os.listdir()`

```python
file_list = os.listdir(file_dir)
```

读取当前类别目录下的所有文件名。

## 16.4 先创建空数组

```python
xi = np.zeros((len(file_list), HW, HW, 3), dtype=np.uint8)
yi = np.zeros(len(file_list), dtype=np.uint8)
```

可以理解为先准备两个容器：

```text
xi：保存这个类别的全部图片
yi：保存这个类别的全部标签
```

如果这个目录有 100 张图片：

```text
xi.shape = [100, 224, 224, 3]
yi.shape = [100]
```

## 16.5 逐张读图

```python
for j, img_name in enumerate(file_list):
    img_path = os.path.join(file_dir, img_name)
    img = Image.open(img_path)
    img = img.resize((HW, HW))
    xi[j, ...] = img
    yi[j] = i
```

执行过程是：

```text
拼接完整路径
→ 打开图片
→ 调整到 224×224
→ 放入 xi 的第 j 个位置
→ 把类别 i 放入 yi 的第 j 个位置
```

## 16.6 合并 11 类数据

```python
if i == 0:
    X = xi
    Y = yi
else:
    X = np.concatenate((X, xi), axis=0)
    Y = np.concatenate((Y, yi), axis=0)
```

第一类直接作为初始数据。

后面的类别不断沿第 0 维，也就是“图片数量维”拼接进去。

最后得到：

```text
X：全部有标签图片
Y：每张图片对应的数字标签
```

---

# 17. 如何从文件夹编号得到类别标签？

当前代码并不是从图片文件名解析标签，而是通过图片所在的文件夹确定标签。

例如：

```text
training/labeled/03/abc.jpg
```

因为图片位于 `03` 文件夹，所以：

```text
标签 = 3
```

代码中：

```python
yi[j] = i
```

这里的 `i` 就是当前文件夹编号。

这种结构的优点是直观。

缺点是代码默认文件夹必须完整存在：

```text
00, 01, 02, ..., 10
```

如果缺少其中一个目录，`os.listdir()` 会报找不到路径。

---

# 18. 如何读取无标签图片？

无标签数据没有 11 个类别文件夹，因为我们不知道图片属于哪类。

单文件版本直接读取传入目录中的所有图片：

```python
file_list = os.listdir(path)
xi = np.zeros((len(file_list), HW, HW, 3), dtype=np.uint8)
```

然后逐张打开并保存：

```python
for j, img_name in enumerate(file_list):
    img_path = os.path.join(path, img_name)
    img = Image.open(img_path)
    img = img.resize((HW, HW))
    xi[j, ...] = img
```

它只返回：

```python
return xi
```

而不是：

```python
return X, Y
```

因为无标签数据没有真实 `Y`。

模块化版本约定无标签图片位于：

```text
training/unlabeled/00/
```

这里的 `00` 只是为了统一目录结构，不表示这些图片真的都是第 0 类。

---

# 19. `__getitem__()` 与 `__len__()`

单文件版本：

```python
def __getitem__(self, item):
    if self.mode == "semi":
        return self.transform(self.X[item]), self.X[item]
    else:
        return self.transform(self.X[item]), self.Y[item]
```

有标签数据返回：

```text
变换后的图片 Tensor + 正确标签
```

无标签数据返回：

```text
变换后的图片 Tensor + 原始 NumPy 图片
```

为什么无标签数据还要返回原图？

因为后面选出高置信度预测后，需要把对应原图保存到伪标签数据集中，再进行训练增强。

长度函数：

```python
def __len__(self):
    return len(self.X)
```

告诉 DataLoader 数据集中有多少张图片。

---

# 20. 图片经过 DataLoader 后为什么变成 BCHW？

前一个 COVID 项目已经介绍过 Dataset 与 DataLoader 的通用分工，这里只看图片数据新增的 shape。

单张图片经过 `ToTensor()` 后通常是：

```text
[3, 224, 224]
```

DataLoader 把多张图片叠成一个 batch。

例如：

```python
train_loader = DataLoader(
    train_set,
    batch_size=16,
    shuffle=True
)
```

得到的 `batch_x` 通常是：

```text
[16, 3, 224, 224]
```

也就是：

```text
Batch × Channel × Height × Width
```

标签为：

```text
batch_y.shape = [16]
```

最后一批可能不足 16 张。

还要区分：

```text
len(train_loader)         → batch 数量
len(train_loader.dataset) → 图片数量
```

当前代码计算平均 batch loss 时使用前者，计算准确率时使用后者。

# 21. 为什么无标签 DataLoader 必须保持顺序？

训练集和验证集的 `shuffle` 已经在第二个项目中讲过：训练集通常打乱，验证集通常不打乱。

Food-11 新增的重点是无标签数据：

```python
no_label_loader = DataLoader(
    no_label_set,
    batch_size=16,
    shuffle=False
)
```

生成伪标签时，代码先按顺序收集每张图片的预测结果，之后再通过：

```python
dataloader.dataset[index][1]
```

取回第 `index` 张原图。

因此预测顺序必须与 Dataset 原始索引保持一致。

如果这里使用 `shuffle=True`，预测标签就可能与错误的原图配对。

伪标签生成完成以后，新的伪标签训练 DataLoader 可以使用 `shuffle=True`，因为此时图片与伪标签已经组成了一条完整样本。

# 22. 模块化版本 `foodDataset` 的四种模式

模块化版本使用：

```python
pathDict = {
    "train": "training/labeled",
    "train_unl": "training/unlabeled",
    "val": "validation",
    "test": "testing"
}
```

根据 `mode` 自动拼接子目录。

| mode | 数据位置 | 是否有标签 | transform |
|---|---|---|---|
| `train` | `training/labeled` | 有 | 随机增强 |
| `train_unl` | `training/unlabeled` | 无 | 训练增强 |
| `val` | `validation` | 有 | 固定变换 |
| `test` | `testing` | 无 | 固定变换 |

创建训练集：

```python
train_loader = getDataLoader(filepath, "train", batchSize)
```

函数内部会完成：

```text
创建 foodDataset
→ 判断是否需要 shuffle
→ 创建 DataLoader
→ 返回 DataLoader
```

---

# 23. 为什么 Dataset 会同时返回变换后的图片和原图？

模块化版本有标签数据返回：

```python
return xT, y, orix
```

分别是：

```text
xT   ：变换后的 Tensor，用于模型训练
y    ：正确标签
orix ：原始 NumPy 图片，用于观察或后续处理
```

训练函数只使用前两个：

```python
x, target = data[0].to(device), data[1].to(device)
```

第三项原图并没有参与普通训练。

无标签数据返回：

```python
return xT, orix
```

后面伪标签数据集通过原图与模型生成的类别组成新的训练样本。

---

# 24. 当前读取方式为什么占内存？

当前 Dataset 在初始化时就把所有图片读取并 resize，然后保存在一个 NumPy 数组中。

一张 224×224 RGB、`uint8` 图片大约需要：

```text
224 × 224 × 3 ≈ 150,528 字节
```

也就是大约 147 KB，不计算其他开销。

如果有一万张图片，仅原始数组就可能达到约 1.4 GB。

因此：

- sample 数据通常还能使用；
- 完整数据可能占用较多内存；
- 伪标签又会复制一部分原图，进一步增加占用。

更常见的工程写法是：

```text
Dataset 中只保存图片路径
→ __getitem__() 被调用时才读取这一张图片
```

但当前项目保留了博主的教学写法，因为它更直观。

这不妨碍你学习 Dataset 的基本概念，只要知道它不是大型科研数据集的最佳实现即可。

---

# 第四部分：卷积神经网络

# 25. 什么是卷积？

全连接层把输入中的每个数字都与很多参数相连。

图片像素非常多，如果一开始就全部连接，参数数量会非常大，而且没有充分利用图片的空间结构。

卷积层使用一个较小的卷积核，在图片上滑动，寻找局部特征。

早期卷积层可能学习：

- 边缘；
- 颜色变化；
- 简单纹理。

更深的卷积层可能逐渐组合出：

- 圆形结构；
- 条纹；
- 食物表面纹理；
- 盘子与食物轮廓；
- 更完整的类别特征。

卷积层不是由程序员提前规定“这个卷积核必须检测面包边缘”。

卷积核中的数字也是模型参数，会通过反向传播自动学习。

---

# 26. `Conv2d(3, 64, 3, 1, 1)` 逐项解释

单文件版本第一层：

```python
self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
```

完整参数含义是：

```text
in_channels  = 3
out_channels = 64
kernel_size  = 3
stride       = 1
padding      = 1
```

## 26.1 输入通道 3

输入是 RGB 彩色图片：

```text
R、G、B → 3 个通道
```

所以第一层 `in_channels=3`。

## 26.2 输出通道 64

模型使用 64 个不同的卷积核组，产生 64 张特征图。

输出 shape 中的通道数变成 64。

可以把它粗略理解为：

> 模型尝试从不同角度寻找 64 种初级特征。

## 26.3 卷积核大小 3

`kernel_size=3` 表示卷积核空间大小为 3×3。

每次观察图片中一个较小的局部区域。

## 26.4 步长 1

`stride=1` 表示卷积核每次移动一个像素位置。

## 26.5 填充 1

`padding=1` 表示在图片边缘补一圈。

对 3×3 卷积、步长 1 来说，这样可以让卷积前后的高和宽保持不变。

因此：

```text
输入：[B, 3,   224, 224]
输出：[B, 64,  224, 224]
```

通道数变了，高宽暂时没变。

---

# 27. 卷积输出尺寸怎样计算？

二维卷积的一维输出尺寸可以写成：

$$
\text{output}
=
\left\lfloor
\frac{\text{input}+2\times\text{padding}-\text{kernel size}}
{\text{stride}}
\right\rfloor+1
$$

当前第一层：

```text
input = 224
padding = 1
kernel size = 3
stride = 1
```

代入：

```text
(224 + 2×1 - 3) / 1 + 1
= 224
```

所以高宽仍是 224×224。

初学阶段不要求背公式，但要知道：

- 卷积核大小会影响输出；
- stride 增大通常会缩小输出；
- padding 可以控制边缘和输出尺寸。

---

# 28. BatchNorm2d 是什么？

代码：

```python
self.bn1 = nn.BatchNorm2d(64)
```

BatchNorm 会对一批特征进行标准化和可学习变换，使训练通常更加稳定。

它维护训练阶段统计得到的均值和方差。

这就是为什么训练和验证要区分：

```python
model.train()
model.eval()
```

在训练模式中，BatchNorm 使用当前 batch 的统计信息并更新内部记录。

在验证模式中，BatchNorm 使用训练期间积累的统计信息，不应该继续更新。

因此忘记 `model.eval()` 可能导致验证结果不稳定或不准确。

BatchNorm 中也有可以训练的缩放和偏移参数。

---

# 29. MaxPool2d 是什么？

代码：

```python
self.pool1 = nn.MaxPool2d(2)
```

2×2 最大池化会在每个 2×2 区域中保留最大值。

当步长默认等于 2 时，高宽通常减半：

```text
224×224 → 112×112
112×112 → 56×56
56×56   → 28×28
```

池化的作用可以粗略理解为：

- 缩小特征图；
- 减少后续计算；
- 保留较显著的局部响应；
- 扩大后续神经元看到的有效区域。

通道数不会因为普通 MaxPool 自动改变。

---

# 30. `nn.Sequential` 为什么有用？

模块化模型中：

```python
self.layer1 = nn.Sequential(
    nn.Conv2d(64, 128, 3, 1, 1),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.MaxPool2d(2)
)
```

`nn.Sequential` 把多个层按照顺序组合成一个模块。

调用：

```python
x = self.layer1(x)
```

相当于依次执行：

```text
Conv2d
→ BatchNorm2d
→ ReLU
→ MaxPool2d
```

它让 `forward()` 更简洁，也方便把重复结构看成一个整体。

---

# 31. 自定义 CNN 中 shape 的完整变化

以 `simple_class.py` 中的 `myModel` 为例。

输入：

```text
[B, 3, 224, 224]
```

其中 `B` 是 batch size。

## 第一部分

```text
Conv2d(3→64)     [B, 3, 224, 224] → [B, 64, 224, 224]
BatchNorm + ReLU                         shape 不变
MaxPool2d(2)      [B, 64, 224, 224] → [B, 64, 112, 112]
```

## 第二部分

```text
Conv2d(64→128)   [B, 64, 112, 112] → [B, 128, 112, 112]
MaxPool2d(2)                            → [B, 128, 56, 56]
```

## 第三部分

```text
Conv2d(128→256)  [B, 128, 56, 56] → [B, 256, 56, 56]
MaxPool2d(2)                          → [B, 256, 28, 28]
```

## 第四部分

```text
Conv2d(256→512)  [B, 256, 28, 28] → [B, 512, 28, 28]
MaxPool2d(2)                          → [B, 512, 14, 14]
```

## 最后一次池化

```text
[B, 512, 14, 14] → [B, 512, 7, 7]
```

## 展平

每张图片拥有：

```text
512 × 7 × 7 = 25088
```

个特征。

所以展平以后：

```text
[B, 512, 7, 7] → [B, 25088]
```

## 分类头

```text
[B, 25088]
→ Linear(25088, 1000)
→ ReLU
→ Linear(1000, 11)
→ [B, 11]
```

这就是为什么代码中的数字不是随便写的：

```python
self.fc1 = nn.Linear(25088, 1000)
```

如果输入图片尺寸或池化次数改变，25088 可能也需要改变。

---

# 32. 为什么要展平？

卷积部分输出是四维 Tensor：

```text
[B, 512, 7, 7]
```

普通全连接层希望每个样本是一条特征向量：

```text
[B, 特征数量]
```

所以代码执行：

```python
x = x.view(x.size()[0], -1)
```

`x.size()[0]` 是 batch size。

`-1` 表示让 PyTorch 自动计算剩余维度。

因此：

```text
[B, 512, 7, 7]
→ [B, 512×7×7]
→ [B, 25088]
```

展平只是改变数据观察方式，不会凭空增加或减少元素数量。

---

# 33. 全连接层怎样完成分类？

代码：

```python
self.fc1 = nn.Linear(25088, 1000)
self.relu2 = nn.ReLU()
self.fc2 = nn.Linear(1000, num_class)
```

卷积部分负责提取图片特征。

全连接部分把这些特征组合起来，得到每个类别的分数。

当 `num_class=11`：

```text
每张图片 → 11 个类别分数
```

这里可以用“特征提取器 + 分类头”理解：

```text
卷积部分：提取图片特征
全连接部分：根据特征判断类别
```

---

# 第五部分：分类输出与损失函数

# 34. 为什么输出 11 个 logits？

模型最后返回：

```text
[batch_size, 11]
```

假设 batch 中只有一张图片，输出可能是：

```text
[1.2, -0.3, 2.1, 0.8, 0.1, 1.6, -1.0, 0.4, 3.2, 0.5, 1.1]
```

这些原始分数叫 logits。

它们：

- 可以是正数；
- 可以是负数；
- 不要求加起来等于 1；
- 不是直接的概率。

第 8 个位置分数 3.2 最大，所以预测标签为 8。

注意：Python 从 0 开始计数。

---

# 35. Softmax 怎样把分数变成概率？

Softmax 把一组 logits 转换为总和等于 1 的正数。

公式是：

$$
p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

初学阶段不必推导，只需要理解：

```text
较大的 logit → 较大的概率
所有类别概率都大于 0
所有类别概率加起来等于 1
```

代码：

```python
soft = nn.Softmax(dim=1)
pred_soft = soft(pred)
```

`pred` 的 shape 是：

```text
[batch, 11]
```

`dim=1` 表示对每张图片的 11 个类别分数执行 Softmax。

如果忘记指定维度，旧代码可能报错或产生不明确行为。

---

# 36. argmax 怎样得到预测类别？

训练代码中：

```python
np.argmax(pred.detach().cpu().numpy(), axis=1)
```

可以拆成：

```text
pred
→ 从计算图分离
→ 移到 CPU
→ 转为 NumPy
→ 在类别维寻找最大值位置
```

`axis=1` 对应 11 个类别这一维。

如果有三张图片，结果可能是：

```text
[8, 0, 4]
```

表示：

```text
第 1 张预测为类别 8
第 2 张预测为类别 0
第 3 张预测为类别 4
```

实际上不做 Softmax 也能用 argmax 得到相同类别，因为 Softmax 不会改变分数从大到小的顺序。

---

# 37. CrossEntropyLoss 的直观理解

分类项目使用：

```python
loss = nn.CrossEntropyLoss()
```

它比较：

```text
模型输出的 11 个 logits
与
正确类别编号
```

假设正确标签是 3：

```text
target = 3
```

如果模型给第 3 类较高分数，loss 较小。

如果模型给错误类别很高分数，而给第 3 类很低分数，loss 较大。

交叉熵背后与概率、对数和最大似然有关。

初学阶段先记住：

> 它会鼓励正确类别分数变高，并惩罚对错误类别过于自信的预测。

对一个 batch：

```text
pred.shape   = [B, 11]
target.shape = [B]
```

得到的 loss 通常是一个标量：

```text
loss.shape = []
```

这个标量代表本批预测的平均错误程度。

---

# 38. 为什么标签必须是 LongTensor？

代码中：

```python
self.Y = torch.LongTensor(self.Y)
```

CrossEntropyLoss 的类别标签不是概率向量，而是类别索引：

```text
0, 1, 2, ..., 10
```

PyTorch 要求这种类别索引通常使用 64 位整数，也就是 `torch.long`。

如果标签是浮点数，可能出现类似报错：

```text
expected scalar type Long but found Float
```

回归项目的标签通常是 float，因为答案是连续数值。

分类项目的标签通常是 long，因为答案是类别编号。

---

# 39. 为什么训练时不需要手动先做 Softmax？

训练时直接写：

```python
pred = model(x)
train_bat_loss = loss(pred, target)
```

模型输出 logits，没有先执行 Softmax。

这是正确用法，因为 `nn.CrossEntropyLoss()` 内部已经以数值更稳定的方式结合了相关计算。

如果训练时手动做：

```python
prob = softmax(pred)
loss = CrossEntropyLoss(prob, target)
```

反而不符合它预期的输入。

因此要区分：

```text
计算训练 loss：直接使用 logits
展示类别概率：可以使用 Softmax
选择预测类别：logits 直接 argmax 即可
伪标签置信度：需要 Softmax 概率
```

---

# 第六部分：模型选择与迁移学习

# 40. 自定义 CNN、VGG 和 ResNet 的关系

项目中存在两类模型。

## 40.1 自己写的 CNN

`simple_class.py` 中叫：

```python
class myModel(nn.Module):
```

`model_utils/model.py` 中叫：

```python
class MyModel(nn.Module):
```

它们结构相似：

```text
卷积 + BatchNorm + ReLU + 池化
重复多次
→ 展平
→ 全连接层
→ 11 类输出
```

这种模型适合理解 CNN 每一步 shape 变化。

## 40.2 torchvision 经典模型

代码还能选择：

- ResNet18；
- ResNet50；
- VGG11_bn；
- GoogLeNet；
- AlexNet；
- SqueezeNet；
- DenseNet121；
- Inception v3。

这些模型结构已经由 torchvision 实现，不需要自己重新写每一层。

---

# 41. 什么是预训练模型？

预训练模型是在大型数据集上提前训练过的模型。

它已经学到一些通用图片特征，例如：

- 边缘；
- 颜色；
- 纹理；
- 形状；
- 物体局部结构。

再把它用于 Food-11 时，不必完全从随机参数开始。

代码参数：

```python
use_pretrained=True
```

表示尝试加载预训练权重。

```python
use_pretrained=False
```

表示只使用模型结构，参数从头训练。

使用预训练权重通常需要第一次联网下载相应文件。

当前代码使用的是 torchvision 较旧的 `pretrained=` 参数写法。新版 torchvision 更推荐使用 `weights=`，但旧写法在一些版本中仍能运行并产生弃用警告。

---

# 42. 为什么要替换最后的分类层？

预训练模型原本可能用于 1000 类 ImageNet 分类。

Food-11 只有 11 类，所以最后一层必须修改。

ResNet：

```python
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, num_classes)
```

VGG：

```python
num_ftrs = model_ft.classifier[6].in_features
model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
```

不同模型的分类头位置不同：

| 模型 | 需要替换的位置 |
|---|---|
| ResNet | `model.fc` |
| VGG / AlexNet | `model.classifier[6]` |
| DenseNet | `model.classifier` |
| SqueezeNet | `model.classifier[1]` |

这说明 `initialize_model()` 中大量 `if/elif` 不是重复写同一件事，而是在适配不同模型的最后一层结构。

---

# 43. 从头训练、微调和线性探测

## 43.1 从头训练

```text
不加载预训练参数
所有参数从随机值开始学习
```

当前 `main.py`：

```python
initialize_model(
    model_name,
    num_class,
    use_pretrained=False
)
```

属于从头训练 ResNet18。

## 43.2 微调

```text
加载预训练参数
继续更新全部或部分参数
```

当前 `simple_class.py` 调用预训练 VGG，而且没有冻结参数，因此更接近对整个模型进行微调。

## 43.3 线性探测

代码中：

```python
def set_parameter_requires_grad(model, linear_probing):
    if linear_probing:
        for param in model.parameters():
            param.requires_grad = False
```

当 `linear_probing=True` 时，先把原模型参数冻结。

随后替换的新分类层默认仍可训练。

可以理解为：

```text
预训练特征提取器：不动
新的 Food-11 分类头：训练
```

这样训练参数更少、速度更快，但模型适应新任务的能力也可能受限。

注意代码参数名中存在拼写不统一：

```text
函数形参：linear_prob
辅助函数形参：linear_probing
```

含义指向同一个概念。

---

# 44. `initialize_model()` 如何选择不同模型？

函数入口：

```python
def initialize_model(
    model_name,
    num_classes,
    linear_prob=False,
    use_pretrained=True
):
```

四个参数分别表示：

| 参数 | 含义 |
|---|---|
| `model_name` | 想使用的模型名称 |
| `num_classes` | 最终分类数量 |
| `linear_prob` | 是否冻结已有模型参数 |
| `use_pretrained` | 是否加载预训练权重 |

函数内部根据字符串选择模型：

```python
if model_name == "MyModel":
    ...
elif model_name == "resnet18":
    ...
elif model_name == "vgg":
    ...
```

最后返回：

```python
return model_ft, input_size
```

`model_ft` 是创建好的模型。

`input_size` 是该模型常用输入尺寸。

需要注意：当前数据增强固定使用 `HW=224`。如果选择 Inception，函数返回 `input_size=299`，但数据变换不会自动跟着变成 299，仍需要人工同步修改。

---

# 45. 当前两套入口实际使用什么模型？

## 单文件版本

```python
model, _ = initialize_model(
    "vgg",
    11,
    use_pretrained=True
)
```

实际使用：

```text
VGG11_bn
11 个输出类别
加载预训练权重
全部参数可训练
```

虽然文件里定义了 `myModel`，但这行被注释：

```python
# model = myModel(11)
```

所以当前运行时不会使用自定义 CNN。

## 模块化版本

`main.py` 中：

```python
model_name = "resnet18"
```

创建时：

```python
model, input_size = initialize_model(
    model_name,
    num_class,
    use_pretrained=False
)
```

实际使用：

```text
ResNet18
11 个输出类别
不加载预训练权重
从头训练
```

因此不能笼统地说“这个项目使用了一个固定模型”。

应该说：

> 项目提供自定义 CNN 和多种 torchvision 模型。当前单文件入口默认使用预训练 VGG11_bn，模块化入口默认使用不带预训练权重的 ResNet18。

---

# 第七部分：训练与验证

# 46. 图像分类训练与前两个项目哪里相同、哪里不同？

训练骨架与前两个项目相同：

```python
pred = model(x)
bat_loss = loss(pred, target)
bat_loss.backward()
optimizer.step()
optimizer.zero_grad()
```

这里不再重复解释 `backward()`、梯度和参数更新。

Food-11 真正新增的是每一步的数据形状和分类计算：

```text
x.shape      = [B, 3, 224, 224]
target.shape = [B]
pred.shape   = [B, 11]
bat_loss     = CrossEntropyLoss(pred, target)
```

一个 epoch 的项目特有流程是：

```text
有标签图片训练
→ 如果已经生成伪标签，再训练伪标签图片
→ 在验证图片上计算分类 loss 和 accuracy
→ 根据验证准确率保存模型
→ 满足条件时重新筛选无标签图片
```

# 47. 为什么这个项目使用 AdamW？

前两个项目已经介绍过优化器、学习率、`model.parameters()` 和 L2 正则化。

Food-11 当前使用：

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=1e-4
)
```

与第二个项目的 SGD 相比，AdamW 会根据历史梯度信息，自适应地调整不同参数的更新幅度。

这里的：

```text
lr           → 学习率
weight_decay → 权重衰减
```

`weight_decay=1e-4` 与前一个项目显式加入 L2 惩罚的目的相近，都是希望约束参数、减轻过拟合；但 AdamW 对权重衰减的实现方式与直接把 L2 项加进 loss 并不完全相同。

当前阶段只需要知道：优化器从 SGD 换成了 AdamW，训练主线没有改变。

# 48. 一个图片 batch 在训练代码中怎样流动？

前两个项目已经解释过一次 batch 的前向传播、反向传播和参数更新，这里只跟踪 Food-11 的 Tensor。

```python
for batch_x, batch_y in train_loader:
    x = batch_x.to(device)
    target = batch_y.to(device)
    pred = model(x)
    train_bat_loss = loss(pred, target)
    train_bat_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

对应 shape：

```text
batch_x             [B, 3, 224, 224]
batch_y             [B]
model(x)            [B, 11]
CrossEntropyLoss    一个标量
```

与 COVID 回归项目最关键的区别是：

```text
COVID：pred 通常是 [B] 或 [B, 1]，target 是浮点数
Food： pred 是 [B, 11]，target 是 0～10 的 long 类别索引
```

`backward()` 和优化器仍按前两个项目已经学过的方式工作。

# 49. 分类准确率怎样计算？

核心代码：

```python
np.argmax(pred.detach().cpu().numpy(), axis=1)
```

得到每张图片的预测类别。

再与正确标签比较：

```python
预测类别 == target
```

比较结果类似：

```text
[True, False, True, True]
```

NumPy 求和时：

```text
True  → 1
False → 0
```

所以：

```python
train_acc += np.sum(...)
```

累计预测正确的图片数量。

最后：

```python
train_acc / train_loader.dataset.__len__()
```

就是：

$$
\text{accuracy}
=
\frac{\text{预测正确数量}}{\text{图片总数量}}
$$

例如 100 张图片预测正确 73 张：

```text
accuracy = 73 / 100 = 0.73 = 73%
```

准确率适合直观观察整体分类效果。

如果类别数量非常不均衡，只看准确率可能不够，还需要 precision、recall、F1、混淆矩阵等指标。当前代码没有实现这些指标。

---

# 50. 怎样阅读这个分类项目的曲线？

第二个项目已经解释过 train/val loss 和过拟合，这里只补充分类项目新增的 accuracy 曲线。

Food-11 同时记录：

```text
train loss
val loss
train accuracy
val accuracy
```

loss 衡量预测错误程度，accuracy 表示预测正确的图片比例。二者相关，但不是同一个量。

可能出现：

```text
loss 继续小幅下降
accuracy 暂时不变
```

因为 logits 的置信程度发生变化时，loss 会变化；只要最大值位置没有改变，accuracy 就可能不变。

还可能出现：

```text
accuracy 相同
loss 不同
```

因为两个模型虽然答对数量一样，但对正确或错误答案的置信程度不同。

当前单文件版使用 `plt.show()` 显示曲线，没有自动保存。

模块化版本执行：

```python
plt.savefig("acc.png")
```

它会把准确率图保存到运行命令时的当前工作目录，不会自动保存到 `assets/`。

# 第八部分：半监督学习与伪标签

# 51. 为什么要使用无标签图片？

给图片人工标注类别需要时间和成本。

现实中经常出现：

```text
少量有标签数据
+
大量无标签数据
```

普通监督训练只能直接使用有标签数据。

半监督学习希望利用无标签数据中的信息。

当前项目采用一种直观方法：伪标签。

---

# 52. 伪标签的完整流程

整体流程：

```text
先用有标签数据训练模型
        ↓
模型对无标签图片进行预测
        ↓
Softmax 得到类别概率
        ↓
取最大概率及对应类别
        ↓
只保留高于阈值的预测
        ↓
把预测类别当成临时标签
        ↓
图片 + 伪标签组成新 Dataset
        ↓
加入后续训练
```

例如模型对一张无标签图片输出：

```text
面包：0.01
乳制品：0.01
甜点：0.02
...
海鲜：0.992
...
```

如果阈值是：

```text
0.99
```

最大概率 0.992 超过阈值，这张图片会被选中，伪标签设为海鲜对应的类别 8。

如果最大概率只有 0.72，则不会使用。

---

# 53. `semiDataset` / `noLabDataset` 在做什么？

单文件版本叫：

```python
class semiDataset(Dataset):
```

模块化版本叫：

```python
class noLabDataset(Dataset):
```

名称不同，但主要任务相同。

## 53.1 让模型进入设备

```python
model = model.to(device)
```

## 53.2 关闭梯度

```python
with torch.no_grad():
```

生成伪标签只是预测，不需要更新参数。

## 53.3 得到 logits

```python
pred = model(bat_x)
```

## 53.4 转为概率

```python
soft = nn.Softmax(dim=1)
pred_soft = soft(pred)
```

## 53.5 找最大概率和位置

```python
pred_max, pred_value = pred_soft.max(1)
```

返回：

```text
pred_max   ：每张图片的最大类别概率，也就是置信度
pred_value ：最大概率所在位置，也就是预测标签
```

## 53.6 只保留高置信度样本

```python
if prob > thres:
    x.append(...)
    y.append(labels[index])
```

## 53.7 没有样本时怎么办？

```python
if x == []:
    self.flag = False
```

如果没有任何预测超过阈值，就不创建可用的伪标签 DataLoader。

调用函数会返回：

```python
None
```

训练循环判断：

```python
if semi_loader != None:
```

只有存在伪标签数据时才训练它。

---

# 54. 置信度阈值为什么设置得很高？

当前单文件版本：

```python
thres = 0.99
```

模块化入口：

```python
"conf_thres": 0.99
```

高阈值意味着只接受模型非常自信的预测。

优点：

- 减少明显错误伪标签；
- 初期更加保守。

缺点：

- 可能几乎选不到样本；
- 模型可能只选择容易样本；
- 置信度高不等于一定正确。

阈值降低时，选中的数据更多，但错误标签风险也会增加。

因此它是准确性与数量之间的权衡。

---

# 55. 伪标签什么时候加入训练？

单文件版本条件：

```python
if epoch % 3 == 0 and plt_val_acc[-1] > 0.6:
    semi_loader = get_semi_loader(...)
```

含义是：

```text
当前 epoch 编号能被 3 整除
并且
验证准确率超过 60%
```

才重新生成伪标签。

模块化版本条件：

```python
if (
    do_semi
    and plt_val_acc[-1] > acc_thres
    and i % semi_epoch == 0
):
```

默认：

```text
do_semi = True
acc_thres = 0.7
semi_epoch = 10
```

也就是启用半监督、验证准确率超过 70%，并且 epoch 编号满足间隔时才生成。

为什么不从第一批训练开始就给无标签数据打标签？

因为模型刚开始几乎不会分类，生成的伪标签大多不可信。

先让模型在有标签数据上具备一定能力，再使用伪标签更合理。

---

# 56. 半监督学习有什么风险？

伪标签不是人工真实标签，而是模型自己的预测。

如果模型预测错了，又把错误结果当成答案继续训练，可能形成：

```text
错误预测
→ 错误伪标签
→ 使用错误标签训练
→ 模型更加相信错误结果
```

这常被称为确认偏差的一种表现。

还可能出现：

- 模型偏向容易类别；
- 部分类别获得大量伪标签，其他类别很少；
- 概率没有校准，0.99 并不真的代表 99% 正确；
- 随机增强后的图片与生成标签时看到的图片差异较大；
- 错误伪标签使验证效果下降。

因此科研工作中通常需要：

- 观察每类伪标签数量；
- 检查伪标签准确率或抽样质量；
- 设计阈值策略；
- 做有无半监督的对照实验；
- 固定随机种子并重复实验。

当前项目的半监督部分适合理解基本思想，但不能直接当作成熟科研方案。

---

# 第九部分：两套代码逐层串联

# 57. `simple_class.py` 从上到下怎样运行？

直接运行：

```powershell
python simple_class.py
```

Python 大致按照下面顺序执行。

## 57.1 导入工具

```python
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
```

## 57.2 固定随机种子

```python
seed_everything(0)
```

尝试使多次运行更容易复现。

## 57.3 定义全局图片尺寸

```python
HW = 224
```

## 57.4 定义训练和验证 transform

此时只是在创建变换规则，还没有真正处理图片。

## 57.5 定义 Dataset 类

```text
food_Dataset
semiDataset
```

定义类不等于马上创建数据集。

## 57.6 定义模型类

```python
class myModel(nn.Module):
```

同样只是定义结构。

## 57.7 定义训练函数

```python
def train_val(...):
```

此时还没有进入训练。

## 57.8 进入 `main()`

文件末尾：

```python
if __name__ == "__main__":
    main()
```

直接运行文件时开始执行 `main()`。

## 57.9 计算项目路径

```python
base_dir = os.path.dirname(os.path.abspath(__file__))
```

得到 `simple_class.py` 所在目录。

然后：

```python
data_dir = os.path.join(
    base_dir,
    "data",
    "food-11_sample"
)
```

避免依赖博主电脑上的固定 F 盘路径。

## 57.10 创建三个 Dataset

```python
train_set = food_Dataset(train_path, "train")
val_set = food_Dataset(val_path, "val")
no_label_set = food_Dataset(no_label_path, "semi")
```

创建时会直接读取图片到内存。

## 57.11 创建三个 DataLoader

```python
train_loader = DataLoader(..., shuffle=True)
val_loader = DataLoader(..., shuffle=False)
no_label_loader = DataLoader(..., shuffle=False)
```

## 57.12 创建模型

```python
model, _ = initialize_model(
    "vgg",
    11,
    use_pretrained=True
)
```

## 57.13 设置训练参数

```text
learning rate = 0.001
loss = CrossEntropyLoss
optimizer = AdamW
epochs = 15
pseudo-label threshold = 0.99
```

## 57.14 创建保存目录

```python
os.makedirs(
    os.path.dirname(save_path),
    exist_ok=True
)
```

`exist_ok=True` 表示目录已经存在时不报错。

## 57.15 调用训练函数

```python
train_val(...)
```

此时才正式进入 epoch 和 batch 循环。

---

# 58. `main.py` 从上到下怎样运行？

模块化版本大致执行：

```text
导入 model_utils 中的三个功能
→ 固定随机种子
→ 确定项目目录
→ 设置超参数
→ 创建训练、验证和无标签 DataLoader
→ 创建 ResNet18
→ 创建 AdamW
→ 设置模型保存路径
→ 把参数放进 trainpara 字典
→ 调用 model_utils.train.train_val()
```

三个关键导入：

```python
from model_utils.model import initialize_model
from model_utils.train import train_val
from model_utils.data import getDataLoader
```

可以读成：

```text
从 model.py 拿创建模型的函数
从 train.py 拿训练函数
从 data.py 拿创建 DataLoader 的函数
```

路径：

```python
base_dir = os.path.dirname(os.path.abspath(__file__))
filepath = os.path.join(
    base_dir,
    "data",
    "food-11_sample"
)
```

模型：

```python
model_name = "resnet18"
model, input_size = initialize_model(
    model_name,
    num_class,
    use_pretrained=False
)
```

入口：

```python
if __name__ == "__main__":
    train_val(trainpara)
```

需要再次提醒：DataLoader 和模型是在这个判断之前创建的。

所以如果别的文件只是 `import main`，仍然会执行数据读取和模型创建。

---

# 59. `trainpara` 字典中每个参数是什么意思？

当前字典：

```python
trainpara = {
    "model": model,
    "train_loader": train_loader,
    "val_loader": val_loader,
    "no_label_Loader": no_label_Loader,
    "optimizer": optimizer,
    "batchSize": batchSize,
    "loss": loss,
    "epoch": epoch,
    "device": device,
    "save_path": save_path,
    "save_acc": True,
    "max_acc": 0.5,
    "val_epoch": 1,
    "acc_thres": 0.7,
    "conf_thres": 0.99,
    "do_semi": True,
    "pre_path": None
}
```

逐个理解：

| 参数 | 当前值或对象 | 作用 |
|---|---|---|
| `model` | ResNet18 | 要训练的模型 |
| `train_loader` | DataLoader | 有标签训练数据 |
| `val_loader` | DataLoader | 验证数据 |
| `no_label_Loader` | DataLoader | 原始无标签数据 |
| `optimizer` | AdamW | 更新模型参数 |
| `batchSize` | 32 | 记录 batch size，但训练函数当前没有直接使用这个字段 |
| `loss` | CrossEntropyLoss | 衡量分类错误 |
| `epoch` | 10 | 训练轮数 |
| `device` | cuda/cpu | 运算设备 |
| `save_path` | `model_save/model.pth` | 模型保存位置 |
| `save_acc` | True | 当前训练函数读取了，但没有根据它控制保存逻辑 |
| `max_acc` | 0.5 | 初始最佳验证准确率 |
| `val_epoch` | 1 | 每隔多少个 epoch 验证一次 |
| `acc_thres` | 0.7 | 启用伪标签生成前要求的验证准确率 |
| `conf_thres` | 0.99 | 单张无标签图片被接纳的置信度阈值 |
| `do_semi` | True | 是否启用半监督逻辑 |
| `pre_path` | None | 是否从已有完整模型继续 |

这里有两个“阈值”容易混淆：

```text
acc_thres  ：整个模型验证准确率要达到多少
conf_thres ：某一张无标签图片的预测概率要达到多少
```

---

# 60. `model_utils/data.py` 的职责

可以把文件拆成五块。

## 第一块：全局尺寸与变换

```text
HW
test_transform
train_transform
```

## 第二块：普通数据集

```python
class foodDataset(Dataset):
```

负责训练、验证、测试和无标签数据。

## 第三块：伪标签数据集

```python
class noLabDataset(Dataset):
```

使用当前模型从无标签图片中选出高置信度样本。

## 第四块：DataLoader 工厂函数

```python
get_semi_loader(...)
getDataLoader(...)
```

把创建 Dataset 和 DataLoader 的过程包装起来。

## 第五块：图片展示函数

```python
samplePlot(...)
```

用于随机显示 3×3 图片，帮助观察增强前后的效果。

训练主流程没有调用 `samplePlot()`。

---

# 61. `model_utils/model.py` 的职责

这个文件包含：

```text
set_parameter_requires_grad
MyModel
initialize_model
prilearn_para
init_para
```

其中当前 `main.py` 真正直接使用的是：

```python
initialize_model
```

它又会使用：

```python
set_parameter_requires_grad
```

`MyModel` 只有在选择：

```python
model_name == "MyModel"
```

时才会使用。

`prilearn_para()` 用于打印和收集需要训练的参数，但当前入口没有调用。

`init_para()` 用于给卷积层和 BatchNorm 自定义初始化，但当前入口也没有调用。

所以阅读优先级是：

```text
initialize_model
→ set_parameter_requires_grad
→ MyModel
→ 其他辅助函数
```

---

# 62. `model_utils/train.py` 的职责

训练文件可以分成：

```text
1. 从 para 字典取出参数
2. 根据 pre_path 决定是否加载已有模型
3. 把模型移动到 device
4. 创建曲线记录列表
5. 循环 epoch
6. 训练有标签数据
7. 训练已经选出的伪标签数据
8. 验证
9. 保存最佳模型
10. 根据条件生成下一阶段伪标签
11. 画图
```

其中变量名比较容易混淆：

```python
semi_loader = para["no_label_Loader"]
no_label_Loader = None
```

实际上：

```text
semi_loader      ：原始无标签图片的 DataLoader
no_label_Loader  ：筛选后带伪标签的 DataLoader
```

变量名与实际含义没有完全对齐。

训练条件：

```python
if no_label_Loader != None:
```

表示只有已经生成了伪标签数据，才进入半监督训练。

生成条件：

```python
no_label_Loader = get_semi_loader(
    semi_loader,
    model,
    device,
    conf_thres
)
```

表示用原始无标签 loader 生成新的伪标签 loader。

---

# 63. 单文件版与模块化版的差别

| 对比项 | `simple_class.py` | `main.py + model_utils` |
|---|---|---|
| 文件组织 | 全部集中 | 按数据、模型、训练拆分 |
| 默认模型 | 预训练 VGG11_bn | 从头训练 ResNet18 |
| batch size | 16 | 32 |
| 学习率 | 0.001 | 0.0001 |
| epoch | 15 | 10 |
| 验证集 shuffle | False | False |
| 伪标签阈值 | 0.99 | 0.99 |
| 生成伪标签条件 | val acc > 0.6，且 epoch%3=0 | val acc > 0.7，且 epoch%10=0 |
| 保存位置 | `model_save/best_model.pth` | `model_save/model.pth` |
| 曲线保存 | 只显示 | accuracy 同时保存为相对路径 `acc.png` |

两套代码不是完全等价的重构版本，因为默认模型和部分超参数不同。

它们更像：

```text
单文件教学示例
+
功能相似但参数不同的模块化示例
```

比较实验时，必须控制模型、预训练、batch size、学习率和 epoch，否则不能把结果差异简单归因于“单文件还是模块化”。

---

# 第十部分：修复、局限、运行与实验

# 64. 相对原始教学代码修改了什么？

下面列出的才是当前代码相对原始文件的主要修改。

## 64.1 路径改为项目相对路径

原单文件代码依赖博主电脑路径：

```text
F:\pycharm\...
```

当前使用：

```python
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(
    base_dir,
    "data",
    "food-11_sample"
)
```

这样项目换电脑后只需按说明放置数据。

## 64.2 增加单文件入口保护

当前：

```python
if __name__ == "__main__":
    main()
```

避免导入 `simple_class.py` 时自动训练。

## 64.3 指定 Softmax 维度

原来：

```python
nn.Softmax()
```

现在：

```python
nn.Softmax(dim=1)
```

表示沿 11 个类别维计算概率。

## 64.4 修复半监督 loss 统计

原来半监督循环错误累计了普通训练 loss：

```python
semi_loss += train_bat_loss.cpu().item()
```

现在改为：

```python
semi_loss += semi_bat_loss.cpu().item()
```

## 64.5 修复半监督准确率分母

原来除以普通训练集长度。

现在除以：

```python
semi_loader.dataset.__len__()
```

## 64.6 验证集不再打乱

当前：

```python
shuffle=False
```

## 64.7 修复模块化伪标签函数参数

原调用多传了一次 loader。

当前调用：

```python
get_semi_loader(
    semi_loader,
    model,
    device,
    conf_thres
)
```

与函数定义的四个参数一致。

## 64.8 不再覆盖传入的 `max_acc`

训练函数已经从字典读取：

```python
max_acc = para["max_acc"]
```

当前不会紧接着再强制设为 0。

## 64.9 删除未使用导入

从相关文件中删除了未使用的：

- `cv2`；
- `sklearn`；
- `timm`；
- 部分 NumPy 导入。

这样减少了不必要的安装要求，不改变核心训练逻辑。

---

# 65. 哪些核心学习代码没有改？

为了保持与博主教学代码一致，下面这些核心思路仍然保留：

- Dataset 初始化时把图片读入 NumPy 数组；
- 224×224 图片输入；
- 训练 transform 和验证 transform；
- 自定义 CNN 的卷积结构；
- torchvision 多模型选择；
- VGG 和 ResNet 的分类头替换方式；
- CrossEntropyLoss；
- AdamW；
- 训练、验证循环；
- 通过准确率保存最佳模型；
- 高置信度伪标签；
- 使用 `torch.save(model, ...)` 保存完整模型；
- 使用 Matplotlib 显示曲线。

所以当前版本不是重新发明了一套你没学过的框架。

它是在保留原学习路线的基础上，处理路径和明确错误。

---

# 66. 当前代码仍然有哪些局限？

这部分非常重要。

能够指出局限，不代表项目没有价值，而是说明你开始区分“教学代码”和“科研工程代码”。

## 66.1 所有图片一次读入内存

sample 数据可以使用，完整数据可能占用大量内存。

## 66.2 没有完成测试集推理

虽然 Dataset 支持 `test`，训练入口没有生成测试预测文件。

## 66.3 保存完整模型对象

```python
torch.save(model, save_path)
```

容易受类定义、Python 和 PyTorch 版本影响。

## 66.4 没有保存优化器和 epoch

程序中断后，不能完整恢复到原训练状态继续。

## 66.5 使用较旧的预训练参数接口

```python
pretrained=True
```

新版 torchvision 可能给出弃用警告。

## 66.6 预训练 VGG 没有配套 ImageNet Normalize

输入预处理与预训练阶段不完全匹配。

## 66.7 指标较少

当前主要只有 loss 和 accuracy，没有：

- 每类准确率；
- confusion matrix；
- precision；
- recall；
- F1。

## 66.8 实验配置没有自动记录

模型、学习率、随机种子、数据增强等没有与结果一起保存。

后续很难只根据一个 `acc.png` 判断它来自哪次实验。

## 66.9 没有单元测试

没有自动检查：

- 数据 shape 是否正确；
- 标签范围是否在 0～10；
- 模型输出是否为 `[B, 11]`；
- 各路径是否存在。

## 66.10 半监督实现比较基础

没有统计伪标签类别分布，也没有验证伪标签质量。

## 66.11 模块中存在未使用变量或函数

例如：

- `batchSize` 字段在 `train.py` 中未直接使用；
- `save_acc` 被读取但没有控制行为；
- `val_rel` 保存验证输出但之后没有使用；
- `samplePlot` 被导入到训练文件但没有调用；
- `prilearn_para()` 和 `init_para()` 当前入口没有调用。

## 66.12 `val_rel` 可能增加内存占用

`train.py` 每个验证 batch 都执行：

```python
val_rel.append(val_pred)
```

但列表没有被使用或清空。

当验证轮数较多时，可能保留越来越多 Tensor。

## 66.13 曲线文件保存位置不固定

```python
plt.savefig("acc.png")
```

由当前工作目录决定，不会自动保存到 `assets/`。

## 66.14 `checkpoints/README.md` 与代码不一致

真实训练代码使用 `model_save/`，不是 `checkpoints/`。

这些局限说明：

> 当前项目适合入门理解完整分类流程，但还不适合直接作为长期科研代码框架。

---

# 67. 怎样准备数据并运行？

## 67.1 确认 Python 环境

项目当前没有 `requirements.txt`。

根据实际导入，主要需要：

```text
torch
torchvision
numpy
Pillow
tqdm
matplotlib
```

如果 PyTorch 已经按你的 CPU 或 CUDA 环境安装，可以再检查其他库。

示例：

```powershell
python -m pip install numpy pillow tqdm matplotlib
```

PyTorch 与 torchvision 建议使用彼此兼容的版本。

## 67.2 放置 sample 数据

目标位置：

```text
03_food_classification/
└── data/
    └── food-11_sample/
        ├── training/
        │   ├── labeled/
        │   └── unlabeled/
        └── validation/
```

其中有标签目录应包含：

```text
00, 01, 02, ..., 10
```

## 67.3 先做最小检查

第一次可以把 epoch 暂时改为 1。

目的不是取得高准确率，而是确认：

- 数据能读取；
- batch shape 正确；
- 模型可以前向传播；
- loss 可以计算；
- backward 可以执行；
- 模型可以保存。

## 67.4 运行单文件版本

进入项目目录：

```powershell
cd D:\Desktop\deep-learning\03_food_classification
```

运行：

```powershell
python simple_class.py
```

第一次使用预训练 VGG，可能需要联网下载权重。

## 67.5 运行模块化版本

```powershell
python main.py
```

当前 `main.py` 默认 ResNet18 不加载预训练权重，因此不会因为模型权重下载而等待，但从头训练可能更慢、需要更多数据和 epoch 才能获得较好效果。

## 67.6 运行后可能看到什么？

- 读取数据数量；
- 模型输入尺寸；
- tqdm 训练进度条；
- 每个 epoch 的 train loss；
- 每个 epoch 的 validation loss；
- train accuracy；
- validation accuracy；
- Matplotlib 曲线窗口；
- `model_save/` 中的模型文件；
- 模块化版本当前工作目录中的 `acc.png`。

---

# 68. 常见报错与排查顺序

遇到错误不要一次改很多地方。

按下面顺序检查。

## 68.1 `FileNotFoundError`

先确认：

```text
data/food-11_sample/training/labeled/00
```

等目录真实存在。

不要把数据只放在旧的项目根目录。

## 68.2 某个类别文件夹不存在

代码会遍历 00～10。

缺少任何一个都会在 `os.listdir()` 时报错。

## 68.5 CUDA out of memory

优先降低 batch size：

```text
32 → 16 → 8 → 4
```

## 68.6 预训练模型下载失败

单文件版本默认 `use_pretrained=True`。

如果网络不可用，可能卡在下载或报错。

可以确认权重是否已有缓存，或者临时使用不加载预训练权重的方式做运行检查。

## 68.7 shape 不匹配

打印：

```python
print(x.shape)
print(pred.shape)
```

正常分类输入输出应接近：

```text
x.shape    = [B, 3, 224, 224]
pred.shape = [B, 11]
```

## 68.8 标签类型错误

CrossEntropyLoss 目标标签通常要求：

```text
dtype = torch.int64 / torch.long
```

## 68.10 内存占用过高

优先使用 sample 数据。

降低 batch size 只能减少模型每批显存，不一定解决 Dataset 一次加载全部图片造成的内存占用。

---

# 69. 只针对本项目新增知识的实验

前两份文档已经安排过修改 epoch、batch size、学习率和观察过拟合的实验，这里不再重复。

每次实验仍然只修改一个变量，并记录模型、预训练设置、数据增强和验证准确率。

## 实验 1：打印图片 batch 的 shape

在训练前临时观察：

```python
for batch_x, batch_y in train_loader:
    print(batch_x.shape)
    print(batch_y.shape)
    break
```

目标是确认：

```text
图片：[B, 3, 224, 224]
标签：[B]
```

## 实验 2：观察数据增强

使用项目已有的 `samplePlot()` 或自己取同一张训练图片多次，观察随机裁剪、翻转和 AutoAugment 产生的不同结果。

目标是理解：Dataset 中保存的是同一张原图，但每次 `__getitem__()` 可能得到不同训练版本。

## 实验 3：去掉 `RandomRotation(50)`

单文件版本比较：

```text
有随机旋转
无随机旋转
```

观察这个幅度较大的增强是否有利于验证集。

## 实验 4：自定义 CNN 与 VGG

把：

```python
model, _ = initialize_model("vgg", 11, use_pretrained=True)
```

换成项目已有的：

```python
model = myModel(11)
```

比较参数规模、训练速度和验证准确率。

注意记录 VGG 使用了预训练权重，否则实验解释不完整。

## 实验 5：ResNet18 预训练与从头训练

比较：

```text
use_pretrained=True
use_pretrained=False
```

观察 sample 小数据集上预训练权重的作用。

## 实验 6：关闭半监督

模块化版本设置：

```python
"do_semi": False
```

先得到纯监督结果，再与启用伪标签的结果比较。

## 实验 7：修改伪标签阈值

比较：

```text
0.90、0.95、0.99
```

应该同时关心：

- 选中了多少张无标签图片；
- 不同类别各有多少伪标签；
- 验证准确率是否改善。

当前代码没有完整打印这些统计，因此这个实验也能帮助发现代码还缺少哪些观察手段。

# 70. 本项目新增术语表

前两份文档已经解释过的 Tensor、Dataset、DataLoader、Epoch、Loss、Gradient、Backward、Optimizer、Learning Rate、训练集、验证集等词不再重复。

| 术语 | 在本项目中的含义 |
|---|---|
| RGB Channel | 图片的红、绿、蓝三个颜色通道 |
| HWC | NumPy 图片常见排列：高度、宽度、通道 |
| CHW | PyTorch 单张图片排列：通道、高度、宽度 |
| BCHW | 图片 batch 排列：批量、通道、高度、宽度 |
| Transform | 图片进入模型前的处理组合 |
| Data Augmentation | 对训练图片进行随机裁剪、旋转、翻转等合理变化 |
| CNN | 使用卷积处理图片空间结构的神经网络 |
| Convolution | 使用局部卷积核提取特征 |
| Feature Map | 卷积层输出的特征图 |
| BatchNorm2d | 对一批图像特征进行标准化和可学习变换的层 |
| Pooling | 缩小特征图高宽的操作 |
| Flatten | 把 `[C,H,W]` 特征展开为一条向量 |
| Logits | 模型输出的原始类别分数 |
| Softmax | 把一组 logits 转换为总和为 1 的概率 |
| Argmax | 找到最大分数所在的类别位置 |
| CrossEntropyLoss | 当前多分类任务使用的损失函数 |
| Pretrained | 使用模型在大型数据集上提前学到的参数 |
| Fine-tuning | 在预训练参数基础上继续训练 |
| Linear Probing | 冻结原特征提取器，只训练新分类头 |
| Semi-supervised Learning | 同时利用有标签与无标签数据 |
| Pseudo-label | 模型为无标签图片生成的临时类别 |
| Confidence | 模型对伪标签预测的自信程度 |

# 71. 最后总结

这个项目真正新增的知识是：

```text
如何读取图片
如何把图片转换成 Tensor
如何理解 BCHW
如何使用数据增强
如何用 CNN 提取特征
如何输出 11 个类别分数
如何用 CrossEntropyLoss 训练分类模型
如何用准确率评价模型
如何使用 VGG 和 ResNet
如何理解预训练与微调
如何用高置信度伪标签利用无标签数据
```

但它与前两个项目共享同一个核心：

$$
\boxed{
\text{数据}
\rightarrow
\text{模型预测}
\rightarrow
\text{计算损失}
\rightarrow
\text{反向传播}
\rightarrow
\text{更新参数}
\rightarrow
\text{验证效果}
}
$$

第一次学习不需要同时熟练掌握所有模型和半监督算法。

可以按两层检查自己是否已经理解当前项目：

## 第一层：必须掌握

1. 图片 batch 的四个维度；
2. Dataset 与 DataLoader；
3. 模型为什么输出 11 个数；
4. CrossEntropyLoss 的输入；
5. backward、step 和 zero_grad；
6. 训练与验证的区别；
7. 准确率的计算方法。

## 第二层：深入理解

1. 卷积、BatchNorm、ReLU、池化；
2. 自定义 CNN 的 shape 变化；
3. VGG 与 ResNet；
4. 预训练、微调和线性探测；
5. 过拟合与数据增强。


能够解释第一层，就说明你已经抓住图像分类的主线。

能够解释第二层，就说明你已经理解了这个项目中的主要模型与训练设计。


最后请记住：

> 读懂代码不是一次性把每个公式和每个函数全部背下来，而是先抓住数据怎样流动，再逐步理解每一步为什么存在。
