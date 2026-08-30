# 从线性回归到神经网络：用 PyTorch 完成一个 COVID 数据回归项目

> 对应项目：`covid_regression.py`  
> 适合读者：人工智能研0，会 Python 基础语法，考研数学二（高等数学 + 线性代数，不考概率论），概率论本科学过但已经忘得差不多。本文凡是用到统计概念的地方（均值、方差、标准差）都会**从头讲一遍**，不默认你记得。  
> 本文目标：从上一份“手写线性回归”（`mylinear.py`）继续往前走一步，看懂一个真正使用 PyTorch `Dataset`、`DataLoader`、`nn.Module`、优化器和模型保存的回归项目。  
> 说明：博主原版代码里有几处小问题（弹窗卡住导致结果不保存、从其他目录运行找不到文件），项目代码已经做了两处小修复，本文按**最新代码**讲解，并会在相应位置指出“博主原版是什么样、为什么改”。

---

## 目录

1. [这个项目在做什么？](#1-这个项目在做什么)
2. [它和上一份线性回归代码有什么区别？](#2-它和上一份线性回归代码有什么区别)
3. [先看完整流程](#3-先看完整流程)
4. [第一部分：导入工具](#4-第一部分导入工具)
5. [第二部分：读取 CSV 数据](#5-第二部分读取-csv-数据)
6. [什么是 Dataset？](#6-什么是-dataset)
7. [CovidDataset 逐行理解](#7-coviddataset-逐行理解)
8. [训练集、验证集、测试集到底是什么？](#8-训练集验证集测试集到底是什么)
9. [为什么要标准化数据？（含概率论复习）](#9-为什么要标准化数据含概率论复习)
10. [`__getitem__` 和 `__len__` 为什么要重写？](#10-__getitem__-和-__len__-为什么要重写)
11. [什么是 DataLoader？](#11-什么是-dataloader)
12. [第三部分：真正的神经网络模型](#12-第三部分真正的神经网络模型)
13. [nn.Module 是什么？](#13-nnmodule-是什么)
14. [nn.Linear 到底是什么？](#14-nnlinear-到底是什么)
15. [ReLU 是干什么的？](#15-relu-是干什么的)
16. [这个网络长什么样？](#16-这个网络长什么样)
17. [`forward()` 为什么这么写？](#17-forward-为什么这么写)
18. [`squeeze(1)` 是什么？](#18-squeeze1-是什么)
19. [第四部分：损失函数 MSE](#19-第四部分损失函数-mse)
20. [为什么还要加 L2 正则化？](#20-为什么还要加-l2-正则化)
21. [第五部分：训练和验证](#21-第五部分训练和验证)
22. [`model.train()` 和 `model.eval()`](#22-modeltrain-和-modeleval)
23. [`optimizer.step()` 和 `zero_grad()`](#23-optimizerstep-和-zero_grad)
24. [`torch.no_grad()` 又是什么？](#24-torchnograd-又是什么)
25. [训练一轮到底发生了什么？](#25-训练一轮到底发生了什么)
26. [为什么要记录 train loss 和 val loss？](#26-为什么要记录-train-loss-和-val-loss)
27. [什么是过拟合？](#27-什么是过拟合)
28. [为什么保存“最佳模型”？](#28-为什么保存最佳模型)
29. [第六部分：训练参数设置](#29-第六部分训练参数设置)
30. [SGD + momentum 是什么？](#30-sgd--momentum-是什么)
31. [CPU 和 GPU：device 是什么？](#31-cpu-和-gpu-device-是什么)
32. [第七部分：测试集预测](#32-第七部分测试集预测)
33. [`torch.load()` 在做什么？](#33-torchload-在做什么)
34. [为什么测试阶段不需要真实答案？](#34-为什么测试阶段不需要真实答案)
35. [最终输出的 predictions.csv 是什么？](#35-最终输出的-predictionscsv-是什么)
36. [从头到尾重新串一遍](#36-从头到尾重新串一遍)
37. [几个最容易搞混的概念](#37-几个最容易搞混的概念)
38. [这份代码中几个值得注意的地方（博主原版 vs 现在）](#38-这份代码中几个值得注意的地方博主原版-vs-现在)
39. [初学者建议做的实验](#39-初学者建议做的实验)
40. [完整知识地图](#40-完整知识地图)
41. [学完这个项目之后，下一步学什么？](#41-学完这个项目之后下一步学什么)

---

# 1. 这个项目在做什么？

这份代码已经从上一份“手写线性回归”进入了真正的 PyTorch 神经网络回归。

程序读取 CSV 数据，每条样本使用 **93 个输入特征**，模型最后输出一个数字。代码中直接设置：

```python
data_dim = 93
```

所以可以先把整个任务理解成：

```text
93 个输入数字
      ↓
   神经网络
      ↓
  预测一个数字
```

这就是一个**回归问题**（回归 = 输出连续数字，比如价格、温度、感染人数；分类 = 输出类别，比如猫/狗）。

整个项目真正想训练的是：

> 一个能够根据输入特征，预测目标数值的神经网络。

运行后，项目目录下会出现：

```text
02_covid_regression/
├── covid_regression.py      ← 主程序
├── data/
│   ├── covid_train.csv      ← 带答案的训练/验证数据
│   └── covid_test.csv       ← 不带答案的测试数据
├── models/
│   └── best_model.pth       ← 训练中验证集表现最好的模型
└── results/
    ├── loss_curve.png       ← train/val loss 曲线图
    └── predictions.csv      ← 对测试集的预测结果（最终产物）
```

---

# 2. 它和上一份线性回归代码有什么区别？

上一份代码的核心模型是：

```python
pred_y = torch.matmul(x, w) + b
```

数学上就是：

$$
\hat y=Xw+b
$$

它是一个简单的线性模型。

这次代码的模型是：

```python
self.fc1 = nn.Linear(in_dim, 128)
self.relu1 = nn.ReLU()
self.fc2 = nn.Linear(128, 1)
```

也就是：

```text
93个输入
   ↓
Linear
   ↓
128个神经元
   ↓
ReLU
   ↓
Linear
   ↓
1个输出
```

因此它已经是一个简单的**多层神经网络**。

不过核心训练思想没有改变：

```text
上一份：
预测 → loss → backward → 手动更新

这一份：
预测 → loss → backward → optimizer 更新
```

最大的变化是：

> 上一份很多事情自己写，这一份开始让 PyTorch 帮你封装。

---

# 3. 先看完整流程

整份程序可以压缩成：

```text
读取 CSV
   ↓
Dataset 负责整理数据（含标准化）
   ↓
DataLoader 每次取一批
   ↓
送入神经网络
   ↓
得到预测值
   ↓
计算 MSE + L2 正则
   ↓
反向传播
   ↓
优化器更新参数
   ↓
训练多个 epoch
   ↓
每轮用验证集检查效果
   ↓
保存验证集表现最好的模型
   ↓
画出 loss 曲线并保存 loss_curve.png
   ↓
加载最佳模型
   ↓
对测试集进行预测
   ↓
保存 predictions.csv
```

你之后遇到任何 PyTorch 训练代码，都可以先找这几部分。

---

# 4. 第一部分：导入工具

最新代码：

```python
import os
import time
import csv
import sys

import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

# 指定 matplotlib 绘图后端
matplotlib.use("TKAgg")
```

大致作用：

| 工具 | 作用 |
|---|---|
| `os` | 操作文件路径、创建文件夹 |
| `time` | 统计训练时间 |
| `csv` | 读写 CSV |
| `sys` | 打印 Python 版本信息 |
| `torch` | PyTorch，Tensor、自动求导、模型训练 |
| `matplotlib.pyplot` | 画图 |
| `numpy` | 数组、矩阵处理 |
| `torch.nn` | 神经网络模块 |
| `optim` | 优化器 |
| `Dataset` | 自定义数据集 |
| `DataLoader` | 一批一批加载数据 |

> ⚠️ 博主原版里还有 `import pandas`，但代码全程没有真正用到它（已删除）。以后自己写代码时，用不到的 import 删掉比较好。

---

## 4.1 `matplotlib.use("TKAgg")`

```python
matplotlib.use("TKAgg")
```

这是在指定 matplotlib 的绘图后端（backend）。

你目前不需要深入理解它，可以先记成：

> 为了让 matplotlib 在特定环境下正常显示/保存图像。

它和神经网络核心没有太大关系。

---

# 5. 第二部分：读取 CSV 数据

核心：

```python
class CovidDataset(Dataset):
    def __init__(self, file_path, mode, mean=None, std=None):
        with open(file_path, "r") as f:
            ori_data = list(csv.reader(f))

        # 去掉第一行表头、第一列 id，并转换成 float
        csv_data = np.array(ori_data)[1:, 1:].astype(float)
```

先看最普通的 Python：

```python
with open(file_path, "r") as f:
```

表示打开文件（"r" = 只读模式）。

然后：

```python
csv.reader(f)
```

把 CSV 内容读进来。

再：

```python
np.array(...)
```

变成 NumPy 数组（你可以暂时理解成"多维表格"）。

最后：

```python
.astype(float)
```

把字符串形式的数字转换成浮点数。

例如：

```text
"3.14"
```

变成：

```python
3.14
```

## 5.1 关于 file_path 的小修复

博主原版里，路径直接写相对路径：

```python
train_file = "data/covid_train.csv"
```

这有一个坑：Python 的"相对路径"是**相对于你运行命令时所在的目录**，而不是脚本所在目录。如果你在 `D:\Desktop\deep-learning` 目录下运行：

```powershell
python 02_covid_regression/covid_regression.py
```

Python 会去找 `D:\Desktop\deep-learning\data\covid_train.csv`，找不到，报错 `FileNotFoundError`。

最新代码的修复方法是，先确定"脚本自己在哪里"，再拼路径：

```python
# 以脚本所在目录为基准，避免从其他目录运行时找不到数据
base_dir = os.path.dirname(os.path.abspath(__file__))

train_file = os.path.join(base_dir, "data", "covid_train.csv")
```

- `__file__` 是 Python 内置变量：当前脚本文件的完整路径。
- `os.path.abspath(__file__)`：转成绝对路径（防止 `__file__` 是相对路径）。
- `os.path.dirname(...)`：取路径中的"目录部分"。
- `os.path.join(base_dir, "data", "covid_train.csv")`：把各部分拼成一个完整路径（Windows 下自动加反斜杠）。

这样，无论你从哪个目录运行，都能找到数据文件。

---

# 6. 什么是 Dataset？

这是这个项目相比上一份代码非常重要的新知识。

上一份代码直接拿：

```python
X, Y
```

自己管理数据。

真实项目中数据会越来越复杂，所以 PyTorch 提供了 `Dataset`。

可以把 Dataset 理解成：

> **一个知道"我有多少数据"，并且知道"第 i 条数据是什么"的对象。**

你这里自己写：

```python
class CovidDataset(Dataset):
```

就是定义了一个符合 PyTorch 规范的数据集类。

---

# 7. CovidDataset 逐行理解

## 7.1 `__init__` 的参数

```python
def __init__(self, file_path, mode, mean=None, std=None):
```

初始化时需要三个参数：

| 参数 | 含义 |
|---|---|
| `file_path` | 文件地址 |
| `mode` | 当前数据属于 `train` / `val` / `test` 中的哪一种 |
| `mean` / `std` | 训练集算好的均值/标准差（验证集、测试集要用），默认 None |

为什么验证集和测试集要传 `mean`/`std`？这是本项目的重点之一，第 9 节会详细讲。

---

## 7.2 去掉第一行和第一列

```python
csv_data = np.array(ori_data)[1:, 1:].astype(float)
```

这里：

```python
[1:, 1:]
```

表示：

> 去掉第 0 行（表头）和第 0 列（id）。

可以想象原始 CSV 是：

```text
第1行：表头
第2行：数据
第3行：数据
...

第1列：id
后面的列：真正的数值
```

所以：

```text
去掉表头
去掉 id
留下真正的数据
```

---

## 7.3 前 93 列是特征，最后一列是答案

```python
# 前 93 列作为输入特征
X = torch.tensor(csv_data[indices, :93])

# 训练集/验证集有真实标签，测试集没有
if mode != "test":
    self.Y = torch.tensor(csv_data[indices, -1])
```

- `:93`：取前 93 列，作为输入特征。
- `-1`：Python 里 `-1` 表示"最后一列"，也就是要预测的目标值（本例是 `tested_positive`，检测呈阳性的人数）。
- 测试集只有特征、没有答案（真实考试才不会提前给你答案）。

---

## 7.4 关键：train 自己算 mean/std，val/test 用训练集的

```python
# 只使用训练集的 mean/std 做标准化
# 训练集自己计算；验证集和测试集使用传入的训练集统计量
if mode == "train":
    self.mean = X.mean(dim=0, keepdim=True)
    self.std = X.std(dim=0, keepdim=True)
else:
    if mean is None or std is None:
        raise ValueError("验证集和测试集需要传入训练集的 mean 和 std")
    self.mean = mean
    self.std = std
```

- 训练集：自己计算每列的均值 `mean` 和标准差 `std`。
- 验证集/测试集：**不自己算**，用传入的训练集 `mean`/`std`。如果没传，就报错提醒你。

然后：

```python
# 防止某一列标准差为 0，导致除以 0
self.std = torch.where(self.std == 0, torch.ones_like(self.std), self.std)

self.X = (X - self.mean) / self.std
```

- 万一某一列所有值都相同（标准差为 0），除以 0 会得到无穷大。所以把 0 替换成 1，保证安全。
- 最后一步：`(X - mean) / std`，就是标准化（第 9 节详细讲）。

---

# 8. 训练集、验证集、测试集到底是什么？

机器学习里经常把数据分成三份：

```text
训练集 train
    ↓
用来学习参数（平时刷题）

验证集 val
    ↓
训练过程中检查模型表现（模拟考试）

测试集 test
    ↓
最后进行预测/评估（最终考试）
```

可以类比考试：

```text
训练集 = 平时刷题
验证集 = 模拟考试
测试集 = 最终考试
```

---

## 8.1 训练集

代码：

```python
if mode == "train":
    indices = [i for i in range(len(csv_data)) if i % 5 != 0]
```

就是：

> 编号不能被 5 整除的数据，放进训练集。

---

## 8.2 验证集

```python
elif mode == "val":
    indices = [i for i in range(len(csv_data)) if i % 5 == 0]
```

就是：

> 编号能被 5 整除的数据，放进验证集。

所以大致是：

```text
5条数据：

第0条 → 验证
第1条 → 训练
第2条 → 训练
第3条 → 训练
第4条 → 训练
```

大约：

```text
80% train
20% val
```

---

## 8.3 测试集

```python
elif mode == "test":
    indices = list(range(len(csv_data)))
```

表示所有测试数据都保留。

注意：

> 测试集不是拿来训练模型的，而是最后拿来预测的。

# 9. 为什么要标准化数据？（含概率论复习）

这一节是整个项目里"最像概率论"的地方。你考数二没考概率论、本科又忘光了，没关系，我们从零讲。

## 9.1 先复习三个统计概念：均值、方差、标准差

### ① 均值（平均数，符号 μ）

```text
μ = (所有数加起来) / (个数)
```

例如一组数：

```text
[2, 4, 4, 4, 5, 5, 7, 9]
```

均值：

```text
(2+4+4+4+5+5+7+9) / 8 = 40 / 8 = 5
```

直觉：**数据的"中心"在哪**。

### ② 方差（符号 σ²）

方差衡量的是：**数据离均值平均有多远**。

```text
σ² = 每个数减去均值，平方，加起来，再除以个数
```

还是上面那组数，均值是 5：

```text
(2-5)² = 9
(4-5)² = 1
(4-5)² = 1
(4-5)² = 1
(5-5)² = 0
(5-5)² = 0
(7-5)² = 4
(9-5)² = 16
```

加起来 = 32，除以 8：

```text
σ² = 32 / 8 = 4
```

两个问题：

**问题1：为什么减去均值？**
> 因为想知道"偏离中心多远"，不是绝对值大小。

**问题2：为什么平方？**
> 如果不平方，正偏差（5-2=3）和负偏差（2-5=-3）会互相抵消，加起来全是 0，啥也看不出。平方后正负都变成正的。这一点和后面 MSE 损失函数的思路一模一样。

直觉：**方差大 = 数据很分散；方差小 = 数据都挤在均值附近**。

### ③ 标准差（符号 σ）

方差是"平方"的单位，如果原始数据单位是"人"，方差单位是"人²"，没有直观含义。开个根号就回到原来的单位：

```text
σ = √(σ²)
```

上面例子里：

```text
σ = √4 = 2
```

直觉：**标准差 = 平均意义上，每个数据离均值有多远**（注意是"平均意义上的近似"，不是严格算术平均，不用纠结）。

### ④ 一句话总结

```text
均值 μ     → 数据的中心
方差 σ²    → 数据有多分散（平方单位）
标准差 σ   → 数据有多分散（原单位）
```

> 💡 机器学习里你还会见到一个词叫"期望"，符号 E[X]，和均值是同一个东西（严格的概率论定义是加权平均，但统计上算样本期望就是求平均）。看到"期望"别慌，就是平均。

---

## 9.2 代码里的标准化（Z-score 标准化）

代码：

```python
self.X = (X - self.mean) / self.std
```

逐列（dim=0）操作，对每个特征列做：

$$
X'=\frac{X-\mu}{\sigma}
$$

把它拆成两步看：

**第一步：减均值 `X - μ`**

> 把数据的中心挪到 0。原来均值是 5 的一列，减去 5 之后，中心变成 0。

**第二步：除以标准差 `/ σ`**

> 把数据的"尺度"统一。标准差是 2 的列，除以 2 之后，数据大致落在 [-2, 2] 之间；标准差是 500 的列，除以 500 之后也大致落在 [-2, 2]。

标准化之后，每一列大致变成：

```text
中心在 0 附近
大部分值在 -3 ~ 3 之间
```

## 9.3 为什么要这么干？

假设不同特征的范围差很多：

```text
特征A：0 ~ 1
特征B：100 ~ 500
特征C：10000 ~ 50000
```

直接送入模型时，不同特征的尺度差距很大。神经网络学习时，"大的数字"会把"小的数字"淹没——模型会误以为特征C最重要，而特征A几乎不重要。

标准化以后，所有特征都变成大致同一尺度：

```text
特征A：-1.5 ~ 1.5
特征B：-1.8 ~ 1.9
特征C：-2.0 ~ 2.1
```

这样神经网络可以平等地看待每个特征，训练通常更快、更稳。

---

## 9.4 为什么验证集/测试集必须用训练集的 mean/std？（重点）

博主原版里，每个 Dataset 都自己算 mean/std，也就是：

```text
train 用自己的均值标准差
val 用自己的
test 用自己的
```

**这是不规范的，最新代码已经修正为：只用训练集算，验证集和测试集复用训练集的那一套。**

为什么？这是机器学习里一个很重要的原则：

> **验证集、测试集的作用是"模拟没见过的新数据"，所以绝不能让它们的信息提前"泄露"给模型。**

可以这样理解：考试时，老师是根据"平时作业"定评分标准，而不是根据"考试卷子"定。如果你用测试集自己的均值去标准化，相当于你偷看了试卷再制定答题规范，分数就没意义了。

所以正确流程是：

```text
训练集算 mean/std
      ↓
训练集、验证集、测试集
全部使用这同一套 mean/std
```

这也是为什么 `CovidDataset.__init__` 要接收 `mean`/`std` 参数、为什么代码里要写那个 `raise ValueError` 提醒你——它就是在强制你遵守这个规范。

---

## 9.5 `dim=0` 是什么？

```python
X.mean(dim=0)
```

可以先理解成：

> 对每一列求平均。

因为我们的表格是：

```text
行 = 样本
列 = 特征
```

所以：

```text
dim=0 → 沿着"行"方向求平均，结果是一个"列"对应一个均值
```

---

## 9.6 `keepdim=True` 是什么？

```python
X.mean(dim=0, keepdim=True)
```

主要作用是：

> 求平均后保留维度，方便后面的减法、除法。

目前不需要深入学习 PyTorch 的广播机制，记住"保留维度，方便做减法"就行。

---

# 10. `__getitem__` 和 `__len__` 为什么要重写？

Dataset 通常至少要告诉 PyTorch 两件事：

### ① 第 i 条数据是什么？

```python
def __getitem__(self, item):
    if self.mode == "test":
        return self.X[item].float()

    return self.X[item].float(), self.Y[item].float()
```

训练和验证：

```text
返回 输入 X + 真实答案 Y
```

测试时（`mode == "test"`）：

```text
只返回输入 X，没有答案
```

### ② 一共有多少条数据？

```python
def __len__(self):
    return len(self.X)
```

这样：

```python
len(train_set)
```

就能得到数据量。

可以记：

```text
__getitem__ = 给我第 i 条
__len__     = 一共有多少条
```

顺便复习一下 Python 语法：**方法名首尾带双下划线的叫"魔法方法"（dunder method）**。`len(x)` 之所以能生效，是因为 Python 内部自动调用 `x.__len__()`；`x[3]` 能生效，是因为内部调用了 `x.__getitem__(3)`。你不需要定义 `len(train_set)` 怎么调用，Python 帮你做了。

---

# 11. 什么是 DataLoader？

代码：

```python
train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
)
```

假设：

```python
batch_size = 16
```

那么：

> 每次给模型 16 条数据。

所以：

```python
for x, y in train_loader:
```

每一次循环就是一个 batch。

你可以把它理解成：

```text
Dataset
= 数据仓库（负责"存/取单条数据"）

DataLoader
= 搬运工（负责"每次搬一批数据过来"）
```

---

## 11.1 `shuffle=True` 和 `shuffle=False`

训练时：

```python
shuffle=True
```

表示：

> 每轮训练时把数据顺序打乱。

为什么要打乱？如果数据按某种规律排序（比如前 80% 是一种、后 20% 是另一种），模型连着看同一类数据，学到的规律会不稳定。打乱后每一批都混合了各种数据，训练更稳。

验证和测试时：

```python
shuffle=False
```

> 验证/测试不需要"学习规律"，按顺序检查即可，不需要打乱。

> ⚠️ 博主原版的 `val_loader` 写的是 `shuffle=True`，最新代码已改为 `shuffle=False`（验证时打乱顺序没有意义）。

---

# 12. 第三部分：真正的神经网络模型

代码：

```python
class MyModel(nn.Module):
    def __init__(self, in_dim):
        super().__init__()

        self.fc1 = nn.Linear(in_dim, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)
```

整体就是：

```text
93
 ↓
Linear(93 → 128)
 ↓
ReLU
 ↓
Linear(128 → 1)
 ↓
1个数字
```

---

# 13. nn.Module 是什么？

```python
class MyModel(nn.Module):
```

`nn.Module` 是 PyTorch 中神经网络的基础类。

可以理解成：

> **PyTorch 提供的神经网络"父类模板"。**

继承它以后，PyTorch 才能更方便地：

- 管理模型参数
- 保存模型
- 加载模型
- 移动到 GPU
- 配合优化器
- 切换训练/验证模式

以后你会经常看到：

```python
class MyModel(nn.Module):
```

---

## 13.1 `super().__init__()`

```python
super().__init__()
```

这是 Python 类继承中的标准写法（Python 3 的简写形式）。

当前阶段记：

> 初始化父类 `nn.Module`，把父类准备好的各种功能激活。

即可。

---

# 14. nn.Linear 到底是什么？

```python
nn.Linear(in_dim, 128)
```

表示：

> 一个全连接层，把 in_dim 个输入数字变成 128 个输出数字。

其数学本质依旧是：

$$
Y=XW+b
$$

这和上一份线性回归代码其实是一个思想：每个输出都是"输入 × 权重 + 偏置"。

区别只是：

```text
上一份：
输入 → 一个线性计算 → 输出（w 是向量）

现在：
93输入 → 线性层 → 128输出（W 是矩阵）
```

上一份的 `w` 是向量，现在变成了矩阵 `W`：128 个输出，每个输出都有自己的权重向量。所以参数一下子多了很多。

---

## 14.1 第一层

```python
self.fc1 = nn.Linear(in_dim, 128)
```

这里：

```text
in_dim = 93
```

所以：

```text
93 → 128
```

---

## 14.2 第二层

```python
self.fc2 = nn.Linear(128, 1)
```

所以：

```text
128 → 1
```

最终只输出一个数字（这就是最终的预测值）。

---

# 15. ReLU 是干什么的？

```python
self.relu1 = nn.ReLU()
```

ReLU 的公式：

$$
ReLU(x)=\max(0,x)
$$

意思：

```text
x < 0 → 0
x > 0 → 原样保留
```

比如：

```text
-5 → 0
-2 → 0
 0 → 0
 3 → 3
 8 → 8
```

---

## 15.1 为什么要加 ReLU？

如果网络只有：

```text
Linear
↓
Linear
↓
Linear
```

多个线性变换组合之后，本质上仍然可以整理成一个大的线性变换（线性函数的复合还是线性函数——考研线代里"线性变换的复合"你肯定学过）。那样"多层网络"就名存实亡了，和一层没区别。

加入：

```text
ReLU
```

以后，网络拥有了**非线性能力**——因为 `max(0, x)` 这条"折线"不是线性的，它把负数一刀切掉。

你当前只需要记：

> **Linear 负责线性变换，ReLU 给网络增加非线性。**

非线性让网络能拟合复杂的曲线，而不是只能拟合直线/平面。

---

# 16. 这个网络长什么样？

```text
x1 ─┐
x2 ─┤
x3 ─┤
... │
x93 ┘
     │
     ▼
┌───────────────┐
│ Linear 93→128 │
└───────────────┘
     │
     ▼
┌───────────────┐
│     ReLU      │
└───────────────┘
     │
     ▼
┌───────────────┐
│  Linear 128→1 │
└───────────────┘
     │
     ▼
   y_pred
```

这就是一个非常基础的前馈神经网络（数据只往前流，没有回头路）。

---

# 17. `forward()` 为什么这么写？

```python
def forward(self, x):
    x = self.fc1(x)
    x = self.relu1(x)
    x = self.fc2(x)

    # [batch_size, 1] -> [batch_size]
    if len(x.size()) > 1:
        x = x.squeeze(1)

    return x
```

它描述的是：

> 输入数据进入模型之后，要依次经过哪些层。

也就是：

```text
输入
 ↓
fc1
 ↓
ReLU
 ↓
fc2
 ↓
输出
```

---

## 17.1 为什么调用 `model(x)` 就会自动执行 `forward()`？

PyTorch 的 `nn.Module` 已经帮你定义好了相关机制（`__call__` 魔法方法会去调用 `forward`）。

所以：

```python
y_pred = model(x)
```

背后最终会执行：

```python
model.forward(x)
```

因此你只需要写好 `forward()`，以后就可以用 `model(x)` 直接得到预测结果。

---

# 18. `squeeze(1)` 是什么？

假设 batch size 是 16。

第二层输出的 shape 可能是：

```text
[16, 1]
```

意思：

```text
16 条数据
每条 1 个预测值
```

但是标签是：

```text
[16]
```

也就是：

```text
16 个数字
```

为了让两者形状一致（后面算 loss 时方便），加上：

```python
if len(x.size()) > 1:
    x = x.squeeze(1)
```

把：

```text
[16, 1]
```

变成：

```text
[16]
```

`len(x.size()) > 1` 的意思是"如果维度多于 1 才处理"。为什么加这个判断？因为 `squeeze(1)` 对"只有 1 维"的 tensor 会直接报错，加个判断保证安全。

你现在记住：

> `squeeze` 的一个常见作用，就是去掉长度为 1 的维度。

# 19. 第四部分：损失函数 MSE

最新代码里损失函数是一个自定义函数：

```python
def mse_loss(pred, target, model):
    # 均方误差
    loss_fn = nn.MSELoss(reduction="mean")

    # L2 正则项
    regularization_loss = 0.0
    for param in model.parameters():
        regularization_loss += torch.sum(param ** 2)

    # 总损失 = MSE + L2 正则
    return loss_fn(pred, target) + 0.00075 * regularization_loss
```

我们拆开讲。先看 MSE 部分：

MSE = Mean Squared Error，**均方误差**。

公式：

$$
MSE=\frac{1}{n}\sum_{i=1}^{n}(\hat y_i-y_i)^2
$$

其中 `ŷ` 是预测值，`y` 是真实值。翻译成人话：

> 把每个样本的"预测值和真实值的差"平方，全部加起来，再取平均。

比如：

```text
真实值：10
预测值：8
```

误差：

```text
8 - 10 = -2
```

平方：

```text
(-2)² = 4
```

最后把所有样本的平方误差取平均。

> 💡 概率论回忆点：这和第 9 节方差的概念几乎一样——方差是"数据偏离均值"的平均平方，MSE 是"预测偏离真实值"的平均平方。看到"平方和取平均"的式子，思路都是同一套。

---

## 19.1 为什么用平方？

因为：

```text
(-2)² = 4
(+2)² = 4
```

正负误差不会互相抵消。如果只用 `预测 - 真实` 直接加，正负会抵消，加起来永远接近 0，啥也学不到。

而且平方让大误差被放大：

```text
误差 2 → 4
误差 5 → 25
```

所以更大的错误会受到更强的惩罚。这也是合理的：我们希望模型尽量少犯大错。

---

## 19.2 `nn.MSELoss(reduction="mean")` 是什么？

```python
loss_fn = nn.MSELoss(reduction="mean")
```

`nn.MSELoss` 是 PyTorch 现成的 MSE 损失函数。`reduction="mean"` 表示"取平均"（对应公式里的 1/n）。

你只需要：

```python
loss_fn(pred, target)
```

就能拿到 MSE 数值。

---

## 19.3 为什么博主把损失函数写成一个函数而不是直接用 `nn.MSELoss`？

因为后面还要加 L2 正则项，PyTorch 内置的 `nn.MSELoss` 不支持这个。所以博主包了一层自定义函数 `mse_loss`，把"MSE + L2"组合在一起。

注意这个函数除了 `pred`、`target`，还接收 `model`——因为要遍历模型的参数算正则项。

---

# 20. 为什么还要加 L2 正则化？

代码：

```python
regularization_loss = 0.0
for param in model.parameters():
    regularization_loss += torch.sum(param ** 2)

return loss_fn(pred, target) + 0.00075 * regularization_loss
```

最终：

$$
Loss=MSE+0.00075\times L2
$$

其中：

$$
L2=\sum w^2
$$

也就是：**把模型所有参数（权重）的平方加在一起**。

---

## 20.1 它是干什么的？

神经网络参数很多。如果只追求：

```text
训练集 loss 越低越好
```

模型有可能变得过于适应训练数据——参数变得很大、很极端，专门去"记"训练集里的每一个点。

但是：

```text
训练集表现很好
验证集表现很差
```

这就是常见的**过拟合**现象（第 27 节详细讲）。

L2 正则化可以理解成：

> 给参数过大加一点"惩罚"，限制模型不要太激进。

数学上看，总损失里多了一项 `0.00075 × Σw²`。如果某个参数变得非常大，这一项就会变大，总损失变大，梯度下降就会"往回拉"。所以 L2 正则会让参数整体偏小、模型更平滑，通常泛化（在新数据上的表现）更好。

---

## 20.2 代码里怎么实现？

```python
for param in model.parameters():
```

遍历模型的所有参数（fc1 的权重和偏置、fc2 的权重和偏置）。

```python
regularization_loss += torch.sum(param ** 2)
```

每个参数平方后求和，累加。

再乘：

```python
0.00075
```

控制这部分惩罚的强度（这个值太小惩罚没作用，太大模型学不动）。

---

## 20.3 `model.parameters()` 是什么？

表示：

> 模型中所有需要训练的参数（权重 W 和偏置 b）。

它返回一个生成器，你可以用 for 循环遍历，拿到每一个参数张量。

---

# 21. 第五部分：训练和验证

训练函数：

```python
def train_val(model, train_loader, val_loader, optimizer, loss_fn,
              device, epochs, save_path):
```

这些参数分别是：

| 参数 | 含义 |
|---|---|
| `model` | 模型 |
| `train_loader` | 训练数据（一批一批取） |
| `val_loader` | 验证数据 |
| `optimizer` | 优化器（负责更新参数） |
| `loss_fn` | 损失函数（MSE + L2） |
| `device` | CPU / GPU |
| `epochs` | 训练轮数 |
| `save_path` | 最佳模型保存位置 |

> ⚠️ 博主原版函数里还有一个 `lr` 参数，但函数体里根本没用到它（学习率只存在于 optimizer 创建时）。最新代码已经把这个多余的参数删掉了，改为传入 `optimizer` 和 `loss_fn`，职责更清晰。

---

## 21.1 函数开头的准备工作

```python
model = model.to(device)
```

把模型移动到 device（GPU 或 CPU）。

```python
# 确保模型保存目录存在
save_dir = os.path.dirname(save_path)
if save_dir:
    os.makedirs(save_dir, exist_ok=True)
```

`os.path.dirname(save_path)` 取出保存路径的目录部分（比如 `models`），`os.makedirs(..., exist_ok=True)` 创建目录（已存在也不会报错）。这样不用你手动建 `models` 文件夹。

```python
train_loss_history = []
val_loss_history = []

# 初始设为正无穷，保证第一个 epoch 能够保存
min_val_loss = float("inf")
```

- 两个列表记录每轮 loss，最后画图用。
- `min_val_loss` 初始为无穷大，这样第一个 epoch 的验证 loss 一定比它小，模型一定会被保存一次。

---

# 22. `model.train()` 和 `model.eval()`

训练时：

```python
model.train()
```

表示：

> 切换到训练模式。

验证时：

```python
model.eval()
```

表示：

> 切换到评估模式。

当前这个简单网络没有 Dropout、BatchNorm 等"训练和验证行为不同"的层，所以你可能感觉不到区别。但是以后会遇到，现在就养成习惯：

```text
训练：
model.train()

验证/测试：
model.eval()
```

---

# 23. `optimizer.step()` 和 `zero_grad()`

训练中的核心四连：

```python
batch_loss.backward()
optimizer.step()
optimizer.zero_grad()
```

可以拆成：

### `backward()`

```python
batch_loss.backward()
```

> 计算梯度（loss 对每个参数的偏导数）。

### `step()`

```python
optimizer.step()
```

> 根据梯度更新参数。

### `zero_grad()`

```python
optimizer.zero_grad()
```

> 清空旧梯度。

所以：

```text
backward
→ 算

step
→ 改

zero_grad
→ 清
```

**为什么必须清空梯度？** 因为 PyTorch 的梯度是"累加"的：不清空的话，下一轮的梯度会和上一轮叠加，导致更新方向错误。所以每轮都要清零。

上一份代码中你自己写：

```python
para -= para.grad * lr
para.grad.zero_()
```

现在就被：

```python
optimizer.step()
optimizer.zero_grad()
```

封装掉了。

---

# 24. `torch.no_grad()` 又是什么？

验证阶段：

```python
with torch.no_grad():
```

原因是：

```text
这里只是检查模型效果
不需要计算梯度
也不会更新参数
```

所以告诉 PyTorch：

> 这一段不需要记录梯度。

这样可以：

- 节省内存（不用保存中间结果用于反向传播）
- 减少计算
- 更明确地表示现在只是推理

**注意 `model.eval()` 和 `torch.no_grad()` 是两件事**：`eval()` 改变模型某些层的行为（如 Dropout 开关），`no_grad()` 关闭梯度记录。代码里两者都用了，各管各的。

---

# 25. 训练一轮到底发生了什么？

核心代码：

```python
for epoch in range(epochs):
    # ---------- 训练 ----------
    model.train()
    start_time = time.time()
    train_loss = 0.0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        # 1. 前向传播
        y_pred = model(x)

        # 2. 计算 loss
        batch_loss = loss_fn(y_pred, y, model)

        # 3. 反向传播，计算梯度
        batch_loss.backward()

        # 4. 更新参数
        optimizer.step()

        # 5. 清空梯度，为下一批数据准备
        optimizer.zero_grad()

        train_loss += batch_loss.item()

    train_loss /= len(train_loader)
    train_loss_history.append(train_loss)
```

逐步翻译：

### 第一步：拿一批数据

```python
for x, y in train_loader:
```

例如：

```text
x = 16条输入
y = 16个真实答案
```

### 第二步：放到设备

```python
x = x.to(device)
y = y.to(device)
```

如果使用 GPU，就把这批数据搬到 GPU 显存上（不搬的话会报错"张量不在同一个设备上"）。

### 第三步：预测

```python
y_pred = model(x)
```

进入：

```text
Linear
 ↓
ReLU
 ↓
Linear
```

得到 16 个预测值。

### 第四步：计算损失

```python
batch_loss = loss_fn(y_pred, y, model)
```

这里的 loss 是：

```text
MSE + L2 正则
```

### 第五步：反向传播

```python
batch_loss.backward()
```

PyTorch 自动计算各个参数的梯度。

### 第六步：更新参数

```python
optimizer.step()
```

优化器根据梯度修改网络参数。

### 第七步：清零梯度

```python
optimizer.zero_grad()
```

准备下一批训练。

### 累加本批 loss

```python
train_loss += batch_loss.item()
```

`.item()` 把 PyTorch 张量变成普通 Python 数字（float）。

一个 epoch 结束后：

```python
train_loss /= len(train_loader)
train_loss_history.append(train_loss)
```

把所有批的 loss 取平均，记下来。`len(train_loader)` 等于一个 epoch 里的批数。

### 验证部分

```python
model.eval()
val_loss = 0.0

with torch.no_grad():
    for val_x, val_y in val_loader:
        val_x = val_x.to(device)
        val_y = val_y.to(device)

        val_pred_y = model(val_x)
        val_batch_loss = loss_fn(val_pred_y, val_y, model)

        val_loss += val_batch_loss.item()

val_loss /= len(val_loader)
val_loss_history.append(val_loss)
```

结构一模一样，区别是：**不更新参数，只算 loss**（包在 `torch.no_grad()` 里）。

---

## 所以训练循环最重要的是：

```text
拿数据
 ↓
预测
 ↓
算 loss
 ↓
backward
 ↓
step
 ↓
zero_grad
 ↓
下一批
```

这就是以后绝大部分 PyTorch 项目的训练骨架。

---

# 26. 为什么要记录 train loss 和 val loss？

代码：

```python
train_loss_history.append(train_loss)
...
val_loss_history.append(val_loss)
```

记录了每轮：

```text
train loss
val loss
```

最后画图保存：

```python
# ---------- 绘制并保存 Loss 曲线 ----------
os.makedirs(save_dir, exist_ok=True)
plt.plot(train_loss_history)
plt.plot(val_loss_history)
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["train", "val"])
plt.savefig(os.path.join(save_dir, "loss_curve.png"))
plt.close()
```

生成一张 `results/loss_curve.png`，两条曲线：train（蓝）、val（橙）。

> ⚠️ **为什么把 `plt.show()` 改成了 `plt.savefig()`？** 这是本项目踩过的一个坑：博主原版写的是 `plt.show()`，它会弹出一个图像窗口，并且**阻塞程序——窗口不关，后面的代码永远不执行**。而保存 predictions.csv 的代码在它后面，所以很多人运行完发现"结果没保存"。改成 `plt.savefig()` 后图片自动存成文件，不再弹窗，程序跑完自然继续保存结果。如果你还是想看曲线，直接打开 `results/loss_curve.png` 即可。

---

## 26.1 为什么不能只看 train loss？

因为：

> 模型可能越来越擅长"记住训练集"，但是越来越不擅长处理新数据。

验证集就是用来检查：

> 模型有没有真正学到规律，而不是只会做见过的数据。

---

# 27. 什么是过拟合？

例如：

```text
训练集：99分
验证集：60分
```

模型对训练数据非常熟，但是遇到新数据就不行。这就是**过拟合**。

一个常见现象是：

```text
train loss
不断下降

val loss
先下降
然后开始上升
```

这说明模型可能已经开始"过度记忆"训练数据。曲线图上看到"train 一直降、val 拐头向上"就是过拟合的经典信号。

反过来，如果 train 和 val 都差，那是**欠拟合**（模型太弱，还没学会）。

---

# 28. 为什么保存"最佳模型"？

代码：

```python
# 初始设为正无穷，保证第一个 epoch 能够保存
min_val_loss = float("inf")
```

然后：

```python
if val_loss < min_val_loss:
    min_val_loss = val_loss
    torch.save(model, save_path)
```

意思：

> 如果当前验证集 loss 比历史最好结果还低，就保存当前模型。

例如：

```text
epoch 1 → val loss = 10
epoch 2 → val loss = 7
epoch 3 → val loss = 5
epoch 4 → val loss = 6
epoch 5 → val loss = 8
```

最佳模型是：

```text
epoch 3
```

所以我们保存的是 epoch 3 的模型，而不是最后的 epoch 5。

**为什么不用最后的模型？** 因为最后一个 epoch 不一定最好，可能已经过拟合了。训练过程中"验证集表现最好"的那个模型，通常对新数据最友好。

`torch.save(model, save_path)` 把整个模型存到 `models/best_model.pth`。

# 29. 第六部分：训练参数设置

代码：

```python
# 超参数
batch_size = 16
epochs = 20
lr = 0.001
```

含义：

```text
batch_size = 16
→ 每次训练16条

epochs = 20
→ 整个训练集学20遍

lr = 0.001
→ 学习率
```

这些叫**超参数**（hyper-parameter）：不是模型自己学出来的，而是我们手动定的。

- `batch_size` 太大：每步更新太"钝"；太小：每步太"碎"、震荡大。
- `epochs` 太少：没学会；太多：过拟合。
- `lr` 太大：参数震荡甚至发散（loss 变成 NaN）；太小：学得慢。

初学阶段，先记住"超参数决定训练快慢和好坏，需要调"，不用急着找最优。

---

# 30. SGD + momentum 是什么？

代码：

```python
optimizer = optim.SGD(
    params=model.parameters(),
    lr=lr,
    momentum=0.9,
)
```

这是 PyTorch 提供的 SGD（随机梯度下降）优化器。

上一份代码手动写：

```python
para -= para.grad * lr
```

现在交给：

```python
optim.SGD(...)
```

---

## 30.1 `params=model.parameters()`

表示：

> 把模型中所有需要训练的参数交给优化器管理。

也就是：

```text
fc1 的参数
fc2 的参数
```

都让优化器统一更新。

---

## 30.2 momentum（动量）

```python
momentum=0.9
```

可以直观理解成：

> 不只看当前这一步的方向，还参考之前移动的方向。

像下山：

```text
普通 SGD：
看当前坡度 → 走一步

Momentum：
看当前坡度 + 参考之前移动方向
→ 更有惯性地走
```

想象一个小球从山上滚下来：它有惯性，不会因为眼前一个小坑就完全改变方向。动量就是给参数更新加一点"惯性"，可以加速收敛、减少震荡。

现在暂时不用背公式。

---

## 30.3 复习：上一份代码的"更新参数"和这里的对比

```text
上一份（手动）：
para -= para.grad * lr
para.grad.zero_()

这一份（封装）：
optimizer.step()
optimizer.zero_grad()
```

核心思想一模一样：**沿着梯度反方向走一步**（梯度的方向是 loss 上升最快的方向，反方向就是下降最快的方向）。

---

# 31. CPU 和 GPU：device 是什么？

代码：

```python
# 选择 CPU 或 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
```

意思：

> 如果有可用的 CUDA GPU，就用 GPU（device 为 "cuda"）；否则用 CPU。

程序运行时会打印：

```text
device: cuda
```

或者：

```text
device: cpu
```

---

## 31.1 为什么深度学习喜欢 GPU？

神经网络里有很多矩阵运算（回想一下 `Y = XW + b`，每个线性层都是一次大矩阵乘法）。

GPU 非常适合：

> 大量类似计算同时进行（几千个核心同时算）。

因此神经网络训练经常使用 GPU。

但这个项目数据量不大（每轮训练只要几秒），CPU 也完全可以完成训练。

---

## 31.2 一个实际坑：装了什么版本的 PyTorch

"能不能用 GPU"取决于你安装的 PyTorch 版本：

```text
PyTorch 2.8.0+cpu        ← CPU 版，没有 CUDA，永远只能跑 CPU
PyTorch 1.11.0+cu113     ← CUDA 11.3 版，可以用 GPU
```

如果装了 `+cpu` 版本，`torch.cuda.is_available()` 返回 False，代码自动用 CPU，这是正常的，不是 bug。

另外注意：本项目代码里 `torch.load(model_path, weights_only=False)` 用到了 PyTorch 2.0 之后才有的参数。**PyTorch 1.x（比如 1.11.0）运行会报错**，建议用 PyTorch 2.x 的环境（比如本项目实际运行成功的 py39 环境，PyTorch 2.8.0）。如果你只有 1.x 环境，把 `weights_only=False` 去掉即可。

---

# 32. 第七部分：测试集预测

训练结束以后：

```python
# ---------- 训练 + 验证 ----------
train_val(...)

# ---------- 测试 ----------
evaluate(
    save_path,
    test_loader,
    result_path,
    device,
)
```

这一阶段不是学习，而是：

> **读取训练过程中保存的最佳模型，对测试数据进行预测。**

---

# 33. `torch.load()` 在做什么？

训练中保存：

```python
torch.save(model, save_path)
```

测试时：

```python
# 加载训练阶段保存的最佳模型
model = torch.load(model_path, weights_only=False).to(device)
```

表示：

> 从文件里把之前保存的模型加载回来，然后移动到 device 上。

整个流程：

```text
训练
 ↓
发现当前 val loss 最好
 ↓
保存模型
 ↓
训练结束
 ↓
加载最佳模型
 ↓
测试
```

---

## 33.1 `weights_only=False` 是什么？

这是 PyTorch 2.0+ 引入的参数：

- 默认 `weights_only=True`：只允许加载"纯参数"，安全性高（防止加载恶意文件时执行任意代码）。
- 本项目保存的是**整个模型对象**（不是纯参数），所以必须传 `weights_only=False` 才能正常加载。

如果你用的 PyTorch 2.x，正常传这个参数就行；如果是 1.x，直接写成 `torch.load(model_path)`。

---

# 34. 为什么测试阶段不需要真实答案？

测试数据：

```python
test_set = CovidDataset(
    test_file,
    "test",
    mean=train_set.mean,
    std=train_set.std,
)
```

而 `__getitem__()` 中：

```python
if self.mode == "test":
    return self.X[item].float()
```

只有：

```text
X
```

没有：

```text
Y
```

因为真实测试答案不提供给模型（`covid_test.csv` 里本来也没有答案列）。

模型只能：

```text
输入
 ↓
预测
```

这和真正考试非常像：

```text
拿到题目
不知道答案
自己作答
```

---

# 35. 最终输出的 predictions.csv 是什么？

代码：

```python
# 确保结果保存目录存在
result_dir = os.path.dirname(result_path)
if result_dir:
    os.makedirs(result_dir, exist_ok=True)

with open(result_path, "w", newline="") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(["id", "tested_positive"])

    for i, pred in enumerate(predictions):
        csv_writer.writerow([str(i), str(pred)])

print("结果保存到了 " + result_path)
```

预测结果写在 `results/predictions.csv`（博主原版叫 `pred.csv`，直接放在当前目录，最新代码统一放进 `results/` 目录）。

最终文件内容：

```csv
id,tested_positive
0,19.256027221679688
1,5.614412307739258
2,6.845026969909668
...
```

也就是：

```text
id（第几条测试样本）
+
模型预测的 tested_positive（检测呈阳性人数）
```

`csv.writer` 生成 csv 文件，`writerow` 每次写一行。

---

## 35.1 预测结果怎么来的？

```python
predictions = []

model.eval()

with torch.no_grad():
    for x in test_loader:
        x = x.to(device)
        pred = model(x)

        # batch_size=1，所以每次拿出一个预测值
        predictions.append(pred.cpu().item())
```

- 测试时 `batch_size=1`，每条样本单独预测。
- `model(x)` 返回 shape 为 `[1]` 的张量，`.cpu()` 搬到 CPU 上，`.item()` 取成普通 Python 数字。
- 注意：模型在 GPU 上计算，结果必须 `.cpu()` 才能做后续的 Python 操作。

---

## 35.2 `enumerate()` 是什么？

普通 Python：

```python
for i, value in enumerate(values):
```

会同时得到：

```text
i      → 第几个
value  → 当前值
```

所以：

```python
for i, pred in enumerate(predictions):
```

就是：

```text
i    → 第几个预测
pred → 预测值
```

# 36. 从头到尾重新串一遍

现在把整份代码完整地串起来：

## 第一步：读取数据

```text
CSV
 ↓
去掉表头
 ↓
去掉 id
 ↓
得到数字矩阵
```

## 第二步：划分数据

```text
训练 CSV
 ↓
train（编号不被5整除）
val（编号被5整除）

测试 CSV
 ↓
test（全部）
```

## 第三步：标准化（只用训练集的 mean/std）

```text
X
 ↓
减均值（训练集的 μ）
 ↓
除标准差（训练集的 σ）
 ↓
标准化 X
```

## 第四步：DataLoader

```text
Dataset
 ↓
DataLoader
 ↓
batch_size=16
 ↓
一批一批取数据
```

## 第五步：神经网络

```text
93
 ↓
Linear(93,128)
 ↓
ReLU
 ↓
Linear(128,1)
 ↓
预测值
```

## 第六步：Loss

```text
预测值 + 真实值
 ↓
MSE
+
L2正则
 ↓
总loss
```

## 第七步：反向传播

```text
loss.backward()
 ↓
算梯度
```

## 第八步：优化器

```text
optimizer.step()
 ↓
更新参数

optimizer.zero_grad()
 ↓
清空梯度
```

## 第九步：验证

```text
model.eval()
 ↓
验证集
 ↓
val loss
 ↓
如果更好
 ↓
保存模型
```

## 第十步：画 loss 曲线

```text
train_loss_history + val_loss_history
 ↓
plt.savefig("results/loss_curve.png")
```

## 第十一步：测试

```text
加载最佳模型
 ↓
测试集
 ↓
预测
 ↓
results/predictions.csv
```

---

# 37. 几个最容易搞混的概念

## 37.1 Dataset 和 DataLoader

```text
Dataset
= 数据仓库（知道每条数据是什么）

DataLoader
= 搬运工（一批一批往外搬）
```

## 37.2 Model 和 Optimizer

```text
model
= 负责预测（前向传播）

optimizer
= 负责更新模型参数
```

## 37.3 train 和 eval

```text
train
= 学习（更新参数）

eval
= 检查（不更新参数）
```

## 37.4 backward 和 step

```text
backward
= 算梯度

step
= 更新参数
```

## 37.5 train loss 和 val loss

```text
train loss
= 在训练数据上的误差

val loss
= 在验证数据上的误差
```

## 37.6 mean/std 与标准化

```text
μ（均值）
= 数据的中心

σ（标准差）
= 数据的分散程度

(X - μ) / σ
= 把数据变成"中心0、尺度统一"
```

## 37.7 GPU 和 CPU

```text
GPU
= 大量并行计算（快，但要装对 PyTorch 版本）

CPU
= 通用计算（慢一些，但一定能用）
```

---

# 38. 这份代码中几个值得注意的地方（博主原版 vs 现在）

这部分不是核心知识，但以后自己写项目时很有价值。博主原版和最新代码的差异都集中在这里，对照着看能帮你理解"什么写法更规范"。

---

## 38.1 `pandas` 已经删掉

博主原版有：

```python
import pandas
```

但后面没有真正调用。最新代码已删掉，不影响程序主流程。

## 38.2 train/val 划分方法比较简单

代码：

```python
i % 5 != 0   # 训练
i % 5 == 0   # 验证
```

用编号规律直接划分，代码自己的注释也写了：

```python
# 逢五取一作为验证集，其余作为训练集
# 这种方法适合学习练习，真实项目通常会使用更规范的划分方法。
```

真实项目一般会用随机划分或 `train_test_split` 这类工具。

## 38.3 标准化方式：只从训练集算 mean/std（已修正）

博主原版里每个 Dataset 内部都自己算 mean/std，意味着 train/val/test 各用各的统计量——这是不规范的（信息泄露问题，见第 9.4 节）。

**最新代码已修正**：只有训练集自己算 mean/std，验证集和测试集必须传入训练集的统计量，否则报错。这也是 `CovidDataset` 多出 `mean=None, std=None` 两个参数的原因。

这是本项目最新代码里**最有学习价值的一处改进**，值得记住这个原则：

> 数据预处理（标准化）的统计量，只能来自训练集。

## 38.4 `val_loader` 的 `shuffle=False`（已修正）

博主原版：

```python
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
```

最新代码：

```python
val_loader = DataLoader(
    val_set,
    batch_size=batch_size,
    shuffle=False,
)
```

验证时不需要打乱顺序，`shuffle=False` 更规范。

## 38.5 `train_val()` 的参数更干净了（已修正）

博主原版函数签名里有一个 `lr` 参数，但函数体里从来没用到（学习率只在创建 optimizer 时使用了一次）。最新代码已删掉 `lr`，改成传入 `optimizer` 和 `loss_fn`，职责清晰：训练函数只用"现成的优化器和损失函数"，不关心它们内部怎么配置。

## 38.6 路径处理（新增修复）

博主原版全部用相对路径，从其他目录运行会报 `FileNotFoundError`。最新代码用 `base_dir = os.path.dirname(os.path.abspath(__file__))` 定位脚本目录，然后：

```python
train_file  = os.path.join(base_dir, "data", "covid_train.csv")
save_path   = os.path.join(base_dir, "models", "best_model.pth")
result_path = os.path.join(base_dir, "results", "predictions.csv")
```

这样从任何目录运行都行。

## 38.7 画图从 `plt.show()` 改为 `plt.savefig()`（新增修复）

博主原版 `plt.show()` 弹窗会阻塞程序，导致后面的预测结果永远不保存。最新代码改成 `plt.savefig(os.path.join(save_dir, "loss_curve.png"))` + `plt.close()`，自动存图、不阻塞。

## 38.8 模型保存方式比较适合入门理解

当前写：

```python
torch.save(model, save_path)
```

直接保存整个模型。

以后更常见的标准写法之一是：

```python
torch.save(model.state_dict(), save_path)
```

保存"参数"（state_dict），然后重新创建模型结构，再加载参数。好处是模型结构和参数分离、跨设备兼容性好。

现在知道存在这种区别就可以。

---

# 39. 初学者建议做的实验

不要只看代码，建议动手改。每次改完一个参数，注意观察两个东西：

```text
① train/val loss 数值和曲线变化
② 运行时间
```

---

## 实验 1：修改隐藏层大小

原来：

```python
self.fc1 = nn.Linear(in_dim, 128)
```

改成：

```python
self.fc1 = nn.Linear(in_dim, 32)
```

再试：

```text
64
256
```

观察：

```text
train loss / val loss
```

思考：隐藏层越大，模型越"能装"，但会不会更容易过拟合？

## 实验 2：去掉 ReLU

临时改成：

```python
def forward(self, x):
    x = self.fc1(x)
    x = self.fc2(x)
    return x
```

然后训练，观察 loss 会怎样。

思考：

> 为什么没有 ReLU 后，网络的表达能力会发生变化？（提示：线性函数复合还是线性，见第 15.1 节）

## 实验 3：修改 batch size

试：

```python
batch_size = 1
```

再试：

```python
batch_size = 64
```

观察速度和 loss 曲线平滑程度。

思考：

```text
batch_size=1 时曲线是不是很"抖"？
batch_size=64 时是不是更稳但每步更新更慢？
```

## 实验 4：修改学习率

试：

```python
lr = 0.0001
```

再试：

```python
lr = 0.01
```

甚至：

```python
lr = 0.1
```

观察：

```text
loss 下降速度
训练是否发散（loss 变成 nan 或巨大值）
```

思考：学习率太大为什么危险？

## 实验 5：修改 L2 正则系数

当前：

```python
0.00075
```

可以试：

```python
0
```

（等于关闭正则，看会不会过拟合）

再试：

```python
0.01
```

（正则太强，看模型是不是"学不动"了）

## 实验 6：把 val 的 shuffle 改回 True

```python
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
```

运行对比一下：验证集顺序会不会影响 val loss？（理论上几乎不影响，但 shuffle=True 没有意义）

## 实验 7（挑战）：修改网络结构

把网络改成三层：

```python
self.fc1 = nn.Linear(in_dim, 64)
self.relu1 = nn.ReLU()
self.fc2 = nn.Linear(64, 32)
self.relu2 = nn.ReLU()
self.fc3 = nn.Linear(32, 1)
```

forward 对应改成三层。观察 loss 是否更低？

---

# 40. 完整知识地图

把这个项目放进深度学习知识体系，可以理解成：

```text
                 PyTorch 回归项目
                        │
        ┌───────────────┼───────────────┐
        │               │               │
       数据             模型             训练
        │               │               │
    CSV → Dataset    nn.Module        loss
        │               │               │
    DataLoader       Linear          backward
        │             ReLU               │
        │               │            optimizer
        │               │               │
        └───────────────┼───────────────┘
                        │
                      验证
                        │
                    val loss
                        │
                  保存最佳模型
                        │
                 loss_curve.png
                        │
                      测试
                        │
              predictions.csv
```

同时，这个项目里其实暗含了三个"统计/数学"知识点，值得单独记：

```text
均值 μ、标准差 σ
→ 第 9 节（标准化）

平方误差取平均
→ 第 19 节（MSE，和方差同源）

平方和惩罚大参数
→ 第 20 节（L2 正则，和"模长"有关，线代里的二范数）
```

---

# 41. 学完这个项目之后，下一步学什么？

如果你是研0，目标是从：

> "只会 Python"

慢慢走到：

> "能看懂和写 PyTorch 项目，能读懂论文里的模型代码"

推荐路线：

```text
Python
 ↓
NumPy
 ↓
PyTorch Tensor
 ↓
Dataset / DataLoader
 ↓
自动求导
 ↓
线性回归
 ↓
nn.Linear
 ↓
ReLU
 ↓
loss
 ↓
backward
 ↓
optimizer
 ↓
train / eval
 ↓
保存与加载模型
 ↓
CNN
 ↓
图像分类
 ↓
...
```

针对你"数二、概率论忘了"的情况，给三个小建议：

1. **数学不用急着重学**：深度学习入门阶段，矩阵乘法（线代你考过）、求导（高数你考过）就够用了。概率论用到最多的地方就是"均值/方差/标准差"和后面的"交叉熵"，用到的时候再补，就像本文第 9 节一样。
2. **遇到不认识的统计符号先看上下文**：深度学习里 90% 的统计符号就是均值、方差、期望、分布。看代码时遇到 `mean`、`std`、`E[]`，先想"这是不是求平均/分散程度"，大概率对。
3. **动手改参数 > 看视频**：第 39 节的实验每个都做一遍，比你再看一遍视频有用得多。

---

# 最后：把这份项目和上一份项目放在一起看

这是最重要的一件事。

上一份项目里，你自己写：

```python
pred_y = torch.matmul(x, w) + b
loss.backward()
para -= para.grad * lr
```

所以你能看到机器学习最底层的训练过程。

这一份变成：

```python
y_pred = model(x)
batch_loss.backward()
optimizer.step()
optimizer.zero_grad()
```

也就是说：

> **PyTorch 开始帮你封装那些重复的底层操作。**

但是核心思想没有变：

```text
上一份：
预测
 ↓
loss
 ↓
backward
 ↓
自己更新

这一份：
预测
 ↓
loss
 ↓
backward
 ↓
optimizer 更新
```

所以可以把这两个项目看成同一条学习路线：

```text
手写线性回归
       ↓
理解参数、loss、梯度
       ↓
理解 PyTorch 自动求导
       ↓
使用 Dataset / DataLoader
       ↓
使用 nn.Module
       ↓
使用 optimizer
       ↓
真正的神经网络训练
```

---

# 一句话总结

这份项目真正需要掌握的不是某一行 API，而是：

$$
\boxed{
\text{数据}
\rightarrow
\text{模型}
\rightarrow
\text{预测}
\rightarrow
\text{Loss}
\rightarrow
\text{Backward}
\rightarrow
\text{Optimizer}
\rightarrow
\text{更新参数}
}
$$

然后每训练一轮：

$$
\boxed{
\text{训练集训练}
\rightarrow
\text{验证集检查}
\rightarrow
\text{保存最佳模型}
}
$$

最后：

$$
\boxed{
\text{加载最佳模型}
\rightarrow
\text{测试集预测}
\rightarrow
\text{保存 predictions.csv}
}
$$

如果你已经能用自己的话解释下面三件事，那么这个项目的核心就已经掌握了：

```text
① Dataset / DataLoader
   → 数据怎么进入模型

② nn.Module
   → 模型怎么算出预测值

③ backward / optimizer
   → 模型怎么根据错误不断调整自己
```

再加上两个"统计小知识"：

```text
④ 均值/标准差（标准化）
   → 为什么数据要变成"中心0、尺度统一"

⑤ 为什么只用训练集算 mean/std
   → 防止信息泄露，保证验证/测试的公平性
```




