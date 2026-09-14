# 从神经网络回归到图像分类：用 PyTorch 完成 Food-11 项目

> 对应项目：`simple_class.py`、`main.py` 和 `model_utils/`  
> 适合读者：已经读过本仓库的手写线性回归和 COVID 神经网络回归，目前第一次学习 CNN 和图像分类。  
> 本文原则：按照你跟着博主写下来的代码讲，不重新发明一套陌生框架。代码中只修复会报错、路径失效或影响上传的问题。  
> 阅读顺序：先看 `simple_class.py`，再看拆分后的 `main.py + model_utils`。

---

## 目录

1. 这个项目到底在做什么？
2. 它和前两个项目有什么联系？
3. 回归与分类有什么区别？
4. Food-11 的 11 个类别
5. 训练集、验证集、测试集和无标签集
6. 项目中各文件负责什么？
7. 图片在计算机里是什么？
8. 图片 shape：H、W、C
9. 为什么要把图片变成 224×224？
10. transform 是什么？
11. 训练集的数据增强
12. 验证集为什么不做随机增强？
13. `food_Dataset` 是什么？
14. `__init__()` 如何准备数据？
15. `read_file()` 如何读取有标签图片？
16. 如何从文件夹得到类别标签？
17. 如何读取无标签图片？
18. `__getitem__()` 返回什么？
19. 标签为什么必须是 LongTensor？
20. DataLoader 如何组成 batch？
21. `myModel` 的整体结构
22. 什么是卷积层？
23. `Conv2d(3, 64, 3, 1, 1)` 逐项解释
24. BatchNorm、ReLU 与 MaxPool
25. 图片经过模型时 shape 如何变化？
26. 为什么要展平？
27. 为什么最后输出 11 个数字？
28. logits、Softmax 和概率
29. CrossEntropyLoss 是什么？
30. 训练循环逐行理解
31. 分类准确率怎么计算？
32. 验证流程逐行理解
33. 为什么保存最佳模型？
34. VGG、ResNet 与迁移学习
35. `main.py` 如何把模块连接起来？
36. 半监督学习和伪标签
37. 原代码修复了什么？
38. 如何运行项目？
39. 常见错误
40. 初学者实验
41. 完整知识地图
42. 一句话总结

---

# 1. 这个项目到底在做什么？

上一份 COVID 项目是根据 93 个数字预测一个连续数值。

这次输入不再是一排 CSV 数字，而是一张食物图片。

程序要回答：

> 这张图片最像 11 类食物中的哪一类？

整体流程可以先看成：

```text
食物图片
   ↓
读取并统一大小
   ↓
转换成 Tensor
   ↓
送入卷积神经网络
   ↓
得到 11 个类别分数
   ↓
选择分数最高的类别
```

例如，模型可能认为一张图片属于：

```text
0 面包：          0.05
1 乳制品：        0.03
2 甜点：          0.08
3 鸡蛋：          0.02
4 油炸食品：      0.07
5 肉类：          0.12
6 面条/意大利面： 0.04
7 米饭：          0.06
8 海鲜：          0.45
9 汤：            0.05
10 蔬菜/水果：    0.13
```

其中海鲜的概率最高，所以最终预测类别是 8。

---

# 2. 它和前两个项目有什么联系？

三个项目是一条连续的学习路线：

```text
01 手写线性回归
    ↓
认识 Tensor、loss、backward、梯度和 SGD
    ↓
02 COVID 神经网络回归
    ↓
认识 Dataset、DataLoader、nn.Module、optimizer、train/eval
    ↓
03 Food-11 图像分类
    ↓
认识图片 Tensor、数据增强、CNN、交叉熵和准确率
```

第三个项目虽然看起来代码更多，但训练核心没有变化。

仍然是：

```python
pred = model(x)
bat_loss = loss(pred, target)
bat_loss.backward()
optimizer.step()
optimizer.zero_grad()
```

和第一份代码对比：

```text
第一份：自己写模型公式，自己写 SGD 更新
第二份：使用 nn.Module 和 torch.optim
第三份：换成 CNN，loss 换成交叉熵
```

因此不要被“图片”“卷积”“ResNet”吓到。

数据类型变了，但训练主线还是原来的主线。

---

# 3. 回归与分类有什么区别？

## 回归

回归输出连续数字：

```text
房价：123.5 万
温度：26.8 ℃
COVID 指标：18.73
```

COVID 项目最后一层是：

```python
nn.Linear(128, 1)
```

因为只需要输出一个预测数字。

## 分类

分类输出有限的类别：

```text
猫 / 狗
正常 / 异常
Food-11 的 11 类食物
```

本项目最后一层需要输出 11 个分数：

```python
self.fc2 = nn.Linear(1000, num_class)
```

当 `num_class=11` 时，输出 shape 是：

```text
[batch_size, 11]
```

## 代码对比

| 项目 | 模型输出 | 标签 | 损失函数 | 常用指标 |
|---|---|---|---|---|
| COVID 回归 | 1 个连续数值 | float | MSE | MSE/MAE |
| Food-11 分类 | 11 个类别分数 | long | CrossEntropyLoss | accuracy |

---

# 4. Food-11 的 11 个类别

代码用整数表示类别：

| 文件夹 | 标签 | 类别 |
|---|---:|---|
| `00` | 0 | Bread，面包 |
| `01` | 1 | Dairy product，乳制品 |
| `02` | 2 | Dessert，甜点 |
| `03` | 3 | Egg，鸡蛋 |
| `04` | 4 | Fried food，油炸食品 |
| `05` | 5 | Meat，肉类 |
| `06` | 6 | Noodles/Pasta，面条/意大利面 |
| `07` | 7 | Rice，米饭 |
| `08` | 8 | Seafood，海鲜 |
| `09` | 9 | Soup，汤 |
| `10` | 10 | Vegetable/Fruit，蔬菜/水果 |

为什么不用中文或英文字符串当标签？

因为神经网络和损失函数使用数字计算。

训练结束后，再把预测数字映射回类别名称即可。

你当前的 `food-11_sample` 中还有一个 `11` 文件夹。

但代码是：

```python
for i in tqdm(range(11)):
```

`range(11)` 只会产生 0 到 10，所以 `11` 文件夹不会被读取。

这次没有强行修改你的数据，而是在数据说明中明确记录了这件事。

---

# 5. 训练集、验证集、测试集和无标签集

完整数据的目录大致是：

```text
food-11/
├── training/
│   ├── labeled/
│   └── unlabeled/
├── validation/
└── testing/
```

## training/labeled

这里的图片有正确答案。

用途：

```text
计算预测
计算 loss
反向传播
更新模型参数
```

## validation

这里也有正确答案，但不用于更新参数。

用途：

```text
检查模型在没有参加训练的数据上表现如何
```

## testing

测试集通常没有公开答案。

用途：

```text
训练完成以后生成最终预测
```

当前代码主要完成训练和验证，还没有单独编写测试集提交文件部分。

## training/unlabeled

这里的图片没有人工标签。

普通有监督训练不能直接使用它们。

项目后半部分使用伪标签，让模型尝试利用这些图片。

---

# 6. 项目中各文件负责什么？

```text
simple_class.py
```

所有主要内容放在同一个文件中，适合第一次从上往下阅读。

```text
main.py
```

模块化版本的入口，负责设置模型名、batch size、学习率、epoch 等参数。

```text
model_utils/data.py
```

负责：

- 图像变换；
- `foodDataset`；
- `noLabDataset`；
- DataLoader；
- 伪标签数据集；
- 图片展示。

```text
model_utils/model.py
```

负责：

- 自定义 `MyModel`；
- ResNet18、ResNet50、VGG 等 torchvision 模型；
- 修改最后的分类层。

```text
model_utils/train.py
```

负责：

- 训练；
- 验证；
- 计算 loss 和准确率；
- 保存最佳模型；
- 启用半监督数据；
- 绘制曲线。

这和 COVID 项目的一个文件相比，只是把不同职责拆开了。

---

# 7. 图片在计算机里是什么？

人看到的是一张食物照片。

计算机看到的是很多像素数字。

彩色图片通常有三个通道：

```text
R：Red，红色
G：Green，绿色
B：Blue，蓝色
```

一个像素可以写成：

```text
[R, G, B]
```

例如：

```text
[255, 0, 0]       红色
[0, 255, 0]       绿色
[0, 0, 255]       蓝色
[255, 255, 255]   白色
[0, 0, 0]         黑色
```

所以一张图片本质上可以转换成一个多维数组。

神经网络不直接理解“面包”，它通过图片数字学习不同类别的特征规律。

---

# 8. 图片 shape：H、W、C

代码设置：

```python
HW = 224
```

每张图片被 resize 成 224×224。

放在 NumPy 中时，shape 是：

```text
[224, 224, 3]
```

也就是：

```text
Height × Width × Channel
```

经过 `ToTensor()` 后，PyTorch 会把顺序变成：

```text
[3, 224, 224]
```

也就是：

```text
Channel × Height × Width
```

组成 batch 后：

```text
[16, 3, 224, 224]
```

第一个 16 是 batch size。

---

# 9. 为什么要把图片变成 224×224？

原始图片可能大小不同：

```text
640×480
300×300
1024×768
```

如果想把多张图片放进同一个 batch，它们必须拥有相同 shape。

所以代码使用：

```python
img = img.resize((HW, HW))
```

或者在 transform 中使用：

```python
transforms.RandomResizedCrop(224)
```

224 也是 VGG、ResNet 等经典模型常见的输入尺寸。

---

# 10. transform 是什么？

代码：

```python
train_transform = transforms.Compose([...])
```

`Compose` 的意思是把多个处理步骤按顺序连接起来。

例如：

```text
原始图片
  ↓
随机裁剪
  ↓
随机旋转
  ↓
转换成 Tensor
```

每次执行：

```python
self.transform(self.X[item])
```

就会把当前图片依次交给这些步骤处理。

---

# 11. 训练集的数据增强

`simple_class.py` 中：

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

## ToPILImage

`self.X` 中保存的是 NumPy 图片。

许多 torchvision 随机变换更适合接收 PIL 图片，因此先转换：

```python
transforms.ToPILImage()
```

## RandomResizedCrop

随机裁剪图片的一部分，再缩放到 224×224。

它让模型看到同一张图片的不同区域和比例。

## RandomRotation

```python
transforms.RandomRotation(50)
```

在一定角度范围内随机旋转图片。

## ToTensor

把图片变成 PyTorch Tensor，并把像素从常见的 0–255 缩放到 0–1。

数据增强的目的不是把图片变漂亮，而是减少模型死记训练图片。

---

# 12. 验证集为什么不做随机增强？

验证变换是：

```python
val_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ToTensor()
    ]
)
```

验证的目标是稳定衡量模型。

如果每次验证都随机旋转、裁剪，那么同一个模型每次面对的图片都不完全相同，结果会更难比较。

因此：

```text
训练集：可以随机增强
验证集：保持稳定
```

DataLoader 也做了类似区分：

```text
训练集 shuffle=True
验证集 shuffle=False
```

---

# 13. `food_Dataset` 是什么？

类定义：

```python
class food_Dataset(Dataset):
```

它继承 PyTorch 的 Dataset。

与 COVID 项目一样，需要完成三个核心部分：

```text
__init__      初始化数据
__getitem__   根据下标返回一个样本
__len__       返回数据数量
```

这里的“一个样本”是：

```text
一张图片 + 一个类别标签
```

无标签模式下则是：

```text
变换后的图片 + 原始图片
```

---

# 14. `__init__()` 如何准备数据？

代码先保存模式：

```python
self.mode = mode
```

如果是半监督无标签模式：

```python
if mode == "semi":
    self.X = self.read_file(path)
```

只得到图片，没有 Y。

其他模式：

```python
else:
    self.X, self.Y = self.read_file(path)
    self.Y = torch.LongTensor(self.Y)
```

得到图片 X 和标签 Y。

接着选择 transform：

```python
if mode == "train":
    self.transform = train_transform
else:
    self.transform = val_transform
```

所以训练模式使用随机增强，验证和半监督读取使用较稳定的变换。

---

# 15. `read_file()` 如何读取有标签图片？

核心循环：

```python
for i in tqdm(range(11)):
```

依次处理 11 个类别。

拼出类别目录：

```python
file_dir = path + "/%02d" % i
```

当 i=0：

```text
path/00
```

当 i=8：

```text
path/08
```

列出图片文件：

```python
file_list = os.listdir(file_dir)
```

准备放图片和标签的数组：

```python
xi = np.zeros((len(file_list), HW, HW, 3), dtype=np.uint8)
yi = np.zeros(len(file_list), dtype=np.uint8)
```

`xi` 的 shape：

```text
[当前类别图片数量, 224, 224, 3]
```

`yi` 保存当前类别所有图片的标签。

---

# 16. 如何从文件夹得到类别标签？

循环中的 i 就是当前类别。

```python
yi[j] = i
```

例如正在读取目录 `05`：

```text
i = 5
```

目录中每张图片的标签都会设置成 5，也就是 Meat。

读完一个类别后，把它与之前类别合并：

```python
X = np.concatenate((X, xi), axis=0)
Y = np.concatenate((Y, yi), axis=0)
```

最终得到：

```text
X：全部图片
Y：每张图片对应的标签
```

这种写法直观，适合看懂数据结构。

需要注意，它会把所有 resize 后的图片放进内存。运行完整数据时会占用较多内存，所以第一次建议使用 sample。

---

# 17. 如何读取无标签图片？

半监督模式中没有类别子目录循环。

代码直接列出文件：

```python
file_list = os.listdir(path)
```

然后逐张读取：

```python
img = Image.open(img_path)
img = img.resize((HW, HW))
xi[j, ...] = img
```

最终只返回：

```python
return xi
```

因为它没有人工标签，所以没有 Y。

---

# 18. `__getitem__()` 返回什么？

有标签模式：

```python
return self.transform(self.X[item]), self.Y[item]
```

返回：

```text
变换后的图片 Tensor
类别标签
```

半监督模式：

```python
return self.transform(self.X[item]), self.X[item]
```

返回：

```text
变换后的图片 Tensor
原始 NumPy 图片
```

为什么还要返回原始图片？

因为模型筛选出高置信度图片后，要把原图保存进新的伪标签数据集，再对它执行训练 transform。

---

# 19. 标签为什么必须是 LongTensor？

代码：

```python
self.Y = torch.LongTensor(self.Y)
```

分类标签不是普通连续小数，而是正确类别的位置：

```text
0, 1, 2, ..., 10
```

`CrossEntropyLoss` 要求 target 是整数类别编号，一般使用 `torch.long`。

如果标签是 float，可能报错：

```text
expected scalar type Long but found Float
```

与回归项目对比：

```text
回归 target：float
分类 target：long
```

---

# 20. DataLoader 如何组成 batch？

代码：

```python
train_loader = DataLoader(
    train_set,
    batch_size=16,
    shuffle=True
)
```

每次循环：

```python
for batch_x, batch_y in train_loader:
```

会取得一批图片和标签。

常见 shape：

```text
batch_x：[16, 3, 224, 224]
batch_y：[16]
```

`batch_y` 中的 16 个数字分别是 16 张图片的正确类别。

DataLoader 仍然负责：

- 分批；
- 打乱训练顺序；
- 调用 Dataset 的 `__getitem__()`；
- 把多个样本组合成 batch。

---

# 21. `myModel` 的整体结构

自定义模型：

```python
class myModel(nn.Module):
```

它大致分成：

```text
第一组：Conv + BatchNorm + ReLU + Pool
第二组：Conv + BatchNorm + ReLU + Pool
第三组：Conv + BatchNorm + ReLU + Pool
第四组：Conv + BatchNorm + ReLU + Pool
再次池化
展平
全连接层
ReLU
11 分类层
```

前半部分负责从图片中提取特征。

后半部分负责根据特征判断类别。

---

# 22. 什么是卷积层？

卷积可以先理解成一个小窗口在图片上移动。

这个小窗口会寻找局部规律，例如：

- 横向边缘；
- 纵向边缘；
- 颜色变化；
- 简单纹理。

第一层学到简单特征后，后面的层会把简单特征组合起来。

可能逐渐形成：

```text
边缘 → 小纹理 → 局部形状 → 更复杂的食物特征
```

卷积适合图片，是因为图片中相邻像素之间通常有关联。

---

# 23. `Conv2d(3, 64, 3, 1, 1)` 逐项解释

代码：

```python
self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
```

完整参数含义：

```text
in_channels  = 3
out_channels = 64
kernel_size  = 3
stride       = 1
padding      = 1
```

## 输入通道 3

因为输入是 RGB 图片。

## 输出通道 64

这一层要学习 64 组卷积核，产生 64 张特征图。

## 卷积核 3

表示窗口是 3×3。

## stride 1

窗口每次移动 1 格。

## padding 1

在图片四周补一圈，使 3×3 卷积后的高宽保持不变。

所以：

```text
[batch, 3, 224, 224]
        ↓ Conv2d
[batch, 64, 224, 224]
```

---

# 24. BatchNorm、ReLU 与 MaxPool

## BatchNorm2d

```python
self.bn1 = nn.BatchNorm2d(64)
```

它对卷积得到的特征进行规范化，并拥有可学习参数。

入门阶段先记住：

> BatchNorm 常用来让训练更稳定。

## ReLU

```python
self.relu = nn.ReLU()
```

它大致执行：

```text
负数 → 0
正数 → 保留
```

它给网络加入非线性能力。

## MaxPool2d

```python
self.pool1 = nn.MaxPool2d(2)
```

通常会把高宽减半：

```text
224×224 → 112×112
```

通道数不会因为这个池化改变。

---

# 25. 图片经过模型时 shape 如何变化？

假设 batch size 是 16：

| 操作 | shape |
|---|---|
| 输入 | `[16, 3, 224, 224]` |
| conv1 | `[16, 64, 224, 224]` |
| pool1 | `[16, 64, 112, 112]` |
| layer1 | `[16, 128, 56, 56]` |
| layer2 | `[16, 256, 28, 28]` |
| layer3 | `[16, 512, 14, 14]` |
| pool2 | `[16, 512, 7, 7]` |
| 展平 | `[16, 25088]` |
| fc1 | `[16, 1000]` |
| fc2 | `[16, 11]` |

为什么 25088？

```text
512 × 7 × 7 = 25088
```

这也是代码写：

```python
self.fc1 = nn.Linear(25088, 1000)
```

的原因。

这套计算依赖输入是 224×224。

如果随意改变输入尺寸，最后的 25088 可能对不上。

---

# 26. 为什么要展平？

卷积输出是四维：

```text
[batch, channel, height, width]
```

全连接层希望每个样本是一排特征：

```text
[batch, feature]
```

所以代码：

```python
x = x.view(x.size()[0], -1)
```

第一维保留 batch size。

`-1` 表示剩余维度自动计算。

于是：

```text
[16, 512, 7, 7]
        ↓
[16, 25088]
```

---

# 27. 为什么最后输出 11 个数字？

代码：

```python
self.fc2 = nn.Linear(1000, num_class)
```

如果：

```python
num_class = 11
```

那么每张图片输出 11 个数。

它们分别对应类别 0 到类别 10。

模型不会直接输出一个“正确类别整数”，因为训练时需要知道每个类别的相对分数。

---

# 28. logits、Softmax 和概率

模型原始输出叫 logits。

例如：

```text
[-0.5, 0.8, 2.1, 0.3, ...]
```

logits：

- 可以是负数；
- 不要求在 0 到 1 之间；
- 不要求总和等于 1。

Softmax：

```python
soft = nn.Softmax(dim=1)
pred_soft = soft(pred)
```

会把每张图片的 11 个 logits 转换成概率分布。

`dim=1` 表示沿类别维度计算。

修复前代码没有写 dim。现在明确写成 `dim=1`，避免不同版本产生警告或不明确行为。

再通过：

```python
pred_max, pred_value = pred_soft.max(1)
```

得到：

```text
pred_max    最大概率，也就是置信度
pred_value  最大概率所在位置，也就是预测类别
```

---

# 29. CrossEntropyLoss 是什么？

代码：

```python
loss = nn.CrossEntropyLoss()
```

它用于多分类问题。

输入：

```text
pred：[batch, 11] 的原始 logits
```

target：

```text
[batch] 的正确类别编号
```

训练计算 CrossEntropyLoss 时，不要先手动对 pred 使用 Softmax。

因为 CrossEntropyLoss 内部已经包含适合训练的相关计算。

记忆：

```text
计算训练 loss：直接传 logits
查看概率或做伪标签：使用 Softmax
```

---

# 30. 训练循环逐行理解

训练开始：

```python
model.train()
```

取一批数据：

```python
for batch_x, batch_y in train_loader:
```

送到 CPU 或 GPU：

```python
x = batch_x.to(device)
target = batch_y.to(device)
```

模型预测：

```python
pred = model(x)
```

计算交叉熵：

```python
train_bat_loss = loss(pred, target)
```

反向传播：

```python
train_bat_loss.backward()
```

更新参数：

```python
optimizer.step()
```

清空梯度：

```python
optimizer.zero_grad()
```

记录 loss：

```python
train_loss += train_bat_loss.cpu().item()
```

这和前两个项目的训练流程完全一致。

---

# 31. 分类准确率怎么计算？

原代码：

```python
np.argmax(pred.detach().cpu().numpy(), axis=1)
```

含义是：

1. `detach()`：从计算图中分离；
2. `cpu()`：移动到 CPU；
3. `numpy()`：转成 NumPy；
4. `argmax(axis=1)`：对每张图片的 11 个分数取最大位置。

再和真实标签比较：

```python
预测类别 == target.cpu().numpy()
```

相等会得到 True，不相等得到 False。

NumPy 求和时，True 当作 1，False 当作 0。

所以：

```python
train_acc += np.sum(...)
```

是在累计预测正确的图片数量。

最后除以数据总数：

```python
train_acc / train_loader.dataset.__len__()
```

得到 0 到 1 之间的准确率。

---

# 32. 验证流程逐行理解

切换验证模式：

```python
model.eval()
```

关闭梯度：

```python
with torch.no_grad():
```

遍历验证集：

```python
for batch_x, batch_y in val_loader:
```

预测与计算 loss：

```python
val_pred_y = model(val_x)
val_batch_loss = loss(val_pred_y, val_y)
```

验证阶段没有：

```python
backward()
optimizer.step()
```

因为验证集只用来检查效果，不能偷偷参加模型更新。

`model.eval()` 和 `torch.no_grad()` 不是同一件事：

```text
eval()      改变 BatchNorm、Dropout 等层的工作状态
no_grad()   不记录梯度
```

验证时通常两个都使用。

---

# 33. 为什么保存最佳模型？

代码比较验证准确率：

```python
if val_acc > max_acc:
    torch.save(model, save_path)
    max_acc = val_acc
```

模型训练越久，不代表验证效果一定越好。

可能出现：

```text
训练准确率持续上升
验证准确率先上升，后来下降
```

这通常说明模型越来越会记训练数据，却不一定更会处理新图片。

保存验证准确率最高的模型，可以保留训练过程中泛化表现最好的一次。

模型文件很大，所以只保存在本地 `model_save/`，不上传 GitHub。

---

# 34. VGG、ResNet 与迁移学习

自定义模型可以写：

```python
model = myModel(11)
```

原代码实际使用：

```python
model, _ = initialize_model("vgg", 11, use_pretrained=True)
```

`initialize_model()` 还支持：

```text
resnet18
resnet50
googlenet
alexnet
vgg
squeezenet
densenet
inception
```

这些是 torchvision 提供的经典模型。

使用预训练：

```python
use_pretrained=True
```

表示先加载模型在大型图片数据集上学习到的参数，再把最后一层改成 11 分类。

例如 ResNet18：

```python
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, num_classes)
```

这叫迁移学习。

第一次运行预训练模型时，可能需要联网下载权重。

---

# 35. `main.py` 如何把模块连接起来？

导入三个模块：

```python
from model_utils.model import initialize_model
from model_utils.train import train_val
from model_utils.data import getDataLoader
```

可以读成：

```text
model.py 给我模型
data.py 给我 DataLoader
train.py 帮我训练和验证
```

设置主要参数：

```python
model_name = 'resnet18'
num_class = 11
batchSize = 32
learning_rate = 1e-4
loss = nn.CrossEntropyLoss()
epoch = 10
```

读取三类数据：

```python
train_loader = getDataLoader(filepath, 'train', batchSize)
val_loader = getDataLoader(filepath, 'val', batchSize)
no_label_Loader = getDataLoader(filepath, 'train_unl', batchSize)
```

建立模型：

```python
model, input_size = initialize_model(
    model_name,
    num_class,
    use_pretrained=False
)
```

建立优化器：

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=1e-4
)
```

最后把配置装进字典：

```python
trainpara = {...}
```

并调用：

```python
train_val(trainpara)
```

模块化版本的训练原理没有变化，只是参数通过字典一次传给训练函数。

---

# 36. 半监督学习和伪标签

## 为什么需要半监督？

有标签数据需要人工标注，数量通常较少。

无标签图片容易获得，但没有正确答案，不能直接计算 CrossEntropyLoss。

伪标签方法：

```text
先用有标签数据训练模型
        ↓
模型预测无标签图片
        ↓
取得 Softmax 最大概率
        ↓
只保留置信度超过阈值的图片
        ↓
把预测类别作为临时标签
        ↓
继续训练
```

## `semiDataset`

初始化时调用：

```python
x, y = self.get_label(no_label_loder, model, device, thres)
```

x 是筛选出的原始图片。

y 是模型为这些图片预测的类别。

## 置信度阈值

代码：

```python
if prob > thres:
```

当 `thres=0.99` 时，只有最大预测概率超过 99% 的图片才会被使用。

阈值高：

```text
数量较少，但通常更可靠
```

阈值低：

```text
数量更多，但错误伪标签可能增加
```

第一次学习建议先把半监督关闭，只理解正常训练。

---

# 37. 原代码修复了什么？

这次没有更换你的模型结构，也没有新增陌生训练框架。

只进行了小范围修复。

## 37.1 路径修复

原 `simple_class.py` 写死了：

```text
F:\pycharm\beike\...
```

换电脑后一定找不到。

现在使用：

```python
base_dir = os.path.dirname(os.path.abspath(__file__))
```

再拼接：

```python
data_dir = os.path.join(base_dir, "data", "food-11_sample")
```

这和 COVID 项目修复路径的方式一致。

## 37.2 增加 main 保护

原代码只要被 import，就会马上读取数据并开始训练。

现在：

```python
if __name__ == "__main__":
    main()
```

只有直接运行时才训练。

这也是 `Python语法速补.md` 中讲过的内容。

## 37.3 Softmax 指定维度

从：

```python
nn.Softmax()
```

改为：

```python
nn.Softmax(dim=1)
```

明确沿 11 个类别计算概率。

## 37.4 半监督 loss 修复

原半监督循环错误累计了上一段的 `train_bat_loss`。

现在改为：

```python
semi_loss += semi_bat_loss.cpu().item()
```

## 37.5 半监督准确率分母修复

原代码用训练集长度当分母。

现在使用：

```python
semi_loader.dataset.__len__()
```

## 37.6 验证集不打乱

从：

```python
shuffle=True
```

改为：

```python
shuffle=False
```

## 37.7 模块化半监督参数修复

原代码写成：

```python
get_semi_loader(
    semi_loader,
    semi_loader,
    model,
    device,
    conf_thres
)
```

函数实际只接收四个参数。

现在改为：

```python
get_semi_loader(
    semi_loader,
    model,
    device,
    conf_thres
)
```

## 37.8 保留传入的 max_acc

原 `train.py` 先读取：

```python
max_acc = para['max_acc']
```

后面又立刻写：

```python
max_acc = 0
```

等于把传入值丢掉。

现在删除了第二次重置。

## 37.9 删除没有使用的 import

删除了没有实际使用的 `cv2`、`sklearn` 和 `timm` 导入。

它们原本可能导致：

```text
ModuleNotFoundError
```

这不会改变训练逻辑。

## 37.10 防止大文件上传

`.gitignore` 会忽略：

- `food-11`；
- `food-11_sample`；
- FashionMNIST；
- `.pth` 模型；
- `.pyc` 缓存；
- `.idea`。

代码、笔记、数据说明和小曲线图仍然会上传。

---

# 38. 如何运行项目？

## 第一步：安装依赖

进入目录：

```powershell
cd 03_food_classification
```

安装：

```powershell
pip install -r requirements.txt
```

## 第二步：放置 sample 数据

将原来的：

```text
food-11_sample
```

复制到：

```text
03_food_classification/data/food-11_sample
```

它只在本地使用，不会上传。

## 第三步：运行单文件版本

```powershell
python simple_class.py
```

当前单文件版本默认：

```text
模型：预训练 VGG11_bn
batch size：16
学习率：0.001
epoch：15
伪标签阈值：0.99
```

如果第一次运行太慢，可以先把：

```python
epochs = 15
```

改成：

```python
epochs = 1
```

## 第四步：运行模块化版本

```powershell
python main.py
```

它默认：

```text
模型：ResNet18
是否预训练：False
batch size：32
学习率：0.0001
epoch：10
```

---

# 39. 常见错误

## 找不到文件夹

检查数据是否放在：

```text
03_food_classification/data/food-11_sample
```

不要只复制 `training/labeled`，需要保持完整层级。

## 缺少 CUDA

输出：

```text
device: cpu
```

不代表代码错误，只是会使用 CPU，训练速度更慢。

## CUDA out of memory

显存不足时降低 batch size：

```python
batchSize = 32
```

改为：

```python
batchSize = 8
```

## 预训练模型下载失败

第一次使用：

```python
use_pretrained=True
```

可能需要联网下载权重。

无法下载时，临时改成 False。

## 矩阵 shape 不匹配

自定义 `myModel` 假设输入是 224×224。

如果改变图片尺寸，最后的：

```python
nn.Linear(25088, 1000)
```

也可能需要重新计算。

## 内存占用太高

原 Dataset 会一次读取全部图片。

第一次使用 sample 数据。

完整数据如果内存不足，后续可以单独学习“只保存路径、按需读图”的 Dataset 写法，但这不属于本次最小修复。

---

# 40. 初学者实验

每次只修改一个参数，才能知道结果变化来自哪里。

## 实验 1：只训练 1 个 epoch

```python
epochs = 1
```

目标不是得到高准确率，而是确认：

- 数据能读取；
- 模型能前向传播；
- loss 能计算；
- backward 能运行；
- 模型能保存。

## 实验 2：修改 batch size

比较：

```text
8
16
32
```

观察速度和显存。

## 实验 3：关闭随机旋转

暂时注释：

```python
transforms.RandomRotation(50)
```

观察验证准确率变化。

## 实验 4：使用自定义 CNN

把：

```python
model, _ = initialize_model("vgg", 11, use_pretrained=True)
```

换成：

```python
model = myModel(11)
```

比较训练速度和准确率。

## 实验 5：比较预训练与不预训练

分别运行：

```python
use_pretrained=True
```

和：

```python
use_pretrained=False
```

## 实验 6：修改学习率

比较：

```text
0.01
0.001
0.0001
```

学习率太大可能使 loss 波动。

学习率太小可能学习很慢。

## 实验 7：修改伪标签阈值

完全理解普通训练以后，再比较：

```text
0.90
0.95
0.99
```

观察筛选出的伪标签数量。

---

# 41. 完整知识地图

```text
图片文件
  ↓ Image.open
PIL 图片
  ↓ resize
224×224×3 NumPy 图片
  ↓ Dataset.__getitem__
随机增强 + ToTensor
  ↓
[3, 224, 224]
  ↓ DataLoader
[batch, 3, 224, 224]
  ↓ CNN
卷积提取局部特征
  ↓
BatchNorm + ReLU
  ↓
MaxPool 缩小高宽
  ↓
重复多层
  ↓
[batch, 512, 7, 7]
  ↓ view 展平
[batch, 25088]
  ↓ 全连接层
[batch, 11] logits
  ↓ CrossEntropyLoss
loss
  ↓ backward
梯度
  ↓ optimizer.step
参数更新
  ↓
重复 batch 和 epoch
  ↓
验证集 accuracy
  ↓
保存最佳模型
```

半监督部分是在普通训练之外增加：

```text
无标签图片
  ↓ 当前模型预测
Softmax 概率
  ↓ 高置信度筛选
图片 + 伪标签
  ↓
加入训练
```

---

# 42. 一句话总结

这个项目真正新增的是：

```text
如何把图片变成 Tensor
如何用 CNN 提取图片特征
如何用 CrossEntropyLoss 训练分类模型
如何用 accuracy 检查分类效果
```

但它仍然建立在前两个项目的基础上：

$$
\boxed{
数据
\rightarrow
模型预测
\rightarrow
计算误差
\rightarrow
反向传播
\rightarrow
更新参数
\rightarrow
验证效果
}
$$

第一次学习不需要同时掌握 VGG、ResNet 和半监督。

先做到：

1. 能说出图片 batch 的四个维度；
2. 能说明 Dataset 返回图片和标签；
3. 能说明 CNN 为什么把通道变多、高宽变小；
4. 能说明模型为什么输出 11 个数；
5. 能区分 logits、Softmax 和预测类别；
6. 能说明 CrossEntropyLoss 的输入；
7. 能说明准确率怎样计算；
8. 能说出训练与验证的区别。

这八点真正理解以后，再继续学习预训练模型和伪标签会容易很多。
