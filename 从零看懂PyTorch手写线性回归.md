# 从零手写一个 PyTorch 线性回归

> 对应项目：`mylinear.py`  
> 适合读者：会 Python 基础语法，知道函数、循环、列表；学过高数但不要求熟悉概率论、机器学习或 PyTorch。  
> 本文目标：不靠“背 API”，真正看懂这段代码为什么能从一堆随机数据里，把参数学出来。

---

## 目录

1. [这个项目到底在干什么？](#1-这个项目到底在干什么)
2. [先别怕数学：只需要认识一个公式](#2-先别怕数学只需要认识一个公式)
3. [程序整体流程](#3-程序整体流程)
4. [导入三个库](#4-导入三个库)
5. [第一步：人为生成一批数据](#5-第一步人为生成一批数据)
6. [补充：正态分布到底是什么？](#6-补充正态分布到底是什么)
7. [理解 X、Y 和 shape](#7-理解-xy-和-shape)
8. [第一次画图：scatter 是什么？](#8-第一次画图scatter-是什么)
9. [第二步：把数据分成一小批一小批](#9-第二步把数据分成一小批一小批)
10. [yield 是什么？](#10-yield-是什么)
11. [第三步：定义模型](#11-第三步定义模型)
12. [第四步：定义损失函数 MAE](#12-第四步定义损失函数-mae)
13. [第五步：什么是梯度下降？](#13-第五步什么是梯度下降)
14. [sgd 函数逐行解释](#14-sgd-函数逐行解释)
15. [requires_grad=True 是什么？](#15-requires_gradtrue-是什么)
16. [第六步：真正开始训练](#16-第六步真正开始训练)
17. [loss.backward() 到底做了什么？](#17-lossbackward-到底做了什么)
18. [epoch、batch、learning rate 一次讲清楚](#18-epochbatchlearning-rate-一次讲清楚)
19. [最后两个 plt 函数到底在画什么？](#19-最后两个-plt-函数到底在画什么)
20. [detach().numpy() 为什么这么长？](#20-detachnumpy-为什么这么长)
21. [为什么最后的点没有紧紧贴着直线？](#21-为什么最后的点没有紧紧贴着直线)
22. [完整训练流程图](#22-完整训练流程图)
23. [这段代码里最值得记住的 10 个东西](#23-这段代码里最值得记住的-10-个东西)
24. [初学者可以做的几个小实验](#24-初学者可以做的几个小实验)
25. [可选优化：让代码更适合继续学习](#25-可选优化让代码更适合继续学习)
26. [如何运行这个项目](#26-如何运行这个项目)
27. [学完这份代码之后该学什么？](#27-学完这份代码之后该学什么)

---

# 1. 这个项目到底在干什么？

这份代码做的事情可以用一句话概括：

> **先自己偷偷设定一个“正确答案”，根据这个正确答案制造 500 条数据，然后让程序假装不知道答案，通过不断试错，把正确答案重新学出来。**

程序提前设定：

```python
true_w = torch.tensor([8.1, 2, 2, 4])
true_b = torch.tensor(1.1)
```

也就是说，真正的数据规律是：

$$
y = 8.1x_1 + 2x_2 + 2x_3 + 4x_4 + 1.1
$$

为了让数据不像数学题那么完美，程序还会给 $y$ 加上一点很小的随机噪声。

然后模型一开始并不知道：

```text
w = [8.1, 2, 2, 4]
b = 1.1
```

它只会从一组接近 0 的随机参数开始，不断进行：

```text
预测
  ↓
看自己错了多少
  ↓
计算应该往哪个方向修改参数
  ↓
修改一点点
  ↓
重新预测
  ↓
继续修改……
```

最后，如果训练正常，学到的 `w_0` 和 `b_0` 应该会逐渐接近真正的 `true_w` 和 `true_b`。

这就是一个最基础的**机器学习训练过程**。

---

# 2. 先别怕数学：只需要认识一个公式

这份代码虽然涉及机器学习，但你暂时不需要概率论基础。

先看最普通的一元一次函数：

$$
y = wx+b
$$

比如：

$$
y=2x+1
$$

这里：

- `x`：输入
- `y`：输出
- `w`：控制斜率
- `b`：控制整体上下移动

当：

$$
x=3
$$

那么：

$$
y=2\times3+1=7
$$

---

## 2.1 这个项目只不过是把一个 x 变成了四个 x

项目中不是：

$$
y=wx+b
$$

而是：

$$
y=w_1x_1+w_2x_2+w_3x_3+w_4x_4+b
$$

具体是：

$$
y=8.1x_1+2x_2+2x_3+4x_4+1.1
$$

本质没有变。

只是一个输入：

```text
x
```

变成了四个输入：

```text
x1、x2、x3、x4
```

你可以先把它想象成“根据 4 个因素预测一个结果”。

例如未来可以是：

```text
房屋面积
房间数
楼层
距离地铁距离
        ↓
      房价
```

这个项目没有真实房价数据，只是自己随机制造数据来练习。

---

# 3. 程序整体流程

先不要陷进代码细节。

整份代码其实就是：

```text
① 设定真实参数 true_w、true_b
             ↓
② create_data()
   根据真实参数制造 500 条训练数据
             ↓
③ data_provider()
   每次随机拿 16 条数据
             ↓
④ fun()
   用当前参数进行预测
             ↓
⑤ maeLoss()
   计算预测得有多离谱
             ↓
⑥ loss.backward()
   计算参数应该往哪个方向修改
             ↓
⑦ sgd()
   更新参数
             ↓
⑧ 重复训练 50 轮
             ↓
⑨ 对比真实参数和训练出来的参数
             ↓
⑩ matplotlib 画图
```

如果后面的某一行看不懂，随时回来看这个流程。

机器学习代码看起来有很多陌生词，但主线通常就是：

> **数据 → 模型 → 损失 → 求梯度 → 更新参数。**

---

# 4. 导入三个库

代码开头：

```python
import torch
import matplotlib.pyplot as plt
import random
```

分别负责：

### `torch`

PyTorch。

主要负责：

- 创建 Tensor（张量）
- 数学计算
- 矩阵乘法
- 自动求导
- 后续真正的神经网络训练

### `matplotlib.pyplot`

负责画图。

这里把它缩写成：

```python
plt
```

以后：

```python
plt.plot(...)
plt.scatter(...)
plt.show()
```

都是在使用 `matplotlib`。

### `random`

Python 自带的随机库。

这里主要用：

```python
random.shuffle(indices)
```

把数据编号随机打乱。

---

# 5. 第一步：人为生成一批数据

代码：

```python
def create_data(w, b, data_num):
    x = torch.normal(0, 1, (data_num, len(w)))
    y = torch.matmul(x, w) + b

    noise = torch.normal(0, 0.01, y.shape)
    y += noise

    return x, y
```

这是整个程序的数据来源。

---

## 5.1 参数是什么意思？

调用：

```python
X, Y = create_data(true_w, true_b, num)
```

前面有：

```python
num = 500
true_w = torch.tensor([8.1, 2, 2, 4])
true_b = torch.tensor(1.1)
```

所以：

```python
create_data(true_w, true_b, 500)
```

相当于告诉函数：

> 按照给定的 `w` 和 `b`，帮我制造 500 条数据。

---

## 5.2 `torch.normal()` 在做什么？

```python
x = torch.normal(0, 1, (data_num, len(w)))
```

这里暂时可以理解成：

> **随机生成很多个大多集中在 0 附近的数字。**

三个参数分别可以先理解为：

```python
torch.normal(平均值, 波动大小, 数据形状)
```

这里是：

```python
torch.normal(0, 1, (500, 4))
```

所以生成：

```text
500 行 × 4 列
```

的数据。

例如看起来可能类似：

```text
[
    [ 0.34, -1.20,  0.53,  0.91],
    [-0.71,  0.18,  1.44, -0.37],
    [ 1.15,  0.62, -0.22,  0.41],
    ...
]
```

每一行是一条数据。

每一列是一个特征。

---

# 6. 补充：正态分布到底是什么？

你概率论忘了也完全不影响看这份代码。

这里你只需要知道一个直观概念。

`torch.normal(0, 1, ...)` 生成的是**正态分布随机数**。

正态分布大概长成“中间高、两边低”的钟形：

```text
数量
 ^
 |             ****
 |          **********
 |        **************
 |      ******************
 |____**********************____> 数值
              0
```

意思是：

> 生成的数据大多数在 0 附近，特别大或者特别小的数据比较少。

这里：

```python
torch.normal(0, 1, ...)
```

第一个 `0` 是均值。

你可以简单理解成：

> 数据主要围绕 0 分布。

第二个 `1` 是标准差。

现阶段可以把它粗略理解成：

> 数据散开的程度。

这篇项目里你不需要计算正态分布概率，也不用背概率密度函数。

记住一句就够：

> **这里用正态分布只是为了方便随机造训练数据。**

---

# 7. 理解 X、Y 和 shape

因为：

```python
X, Y = create_data(true_w, true_b, 500)
```

所以 `X` 有 500 条，每条有 4 个特征：

```text
X.shape = [500, 4]
```

你可以理解成一个表格：

| 数据 | x1 | x2 | x3 | x4 |
|---|---:|---:|---:|---:|
| 第1条 | ... | ... | ... | ... |
| 第2条 | ... | ... | ... | ... |
| 第3条 | ... | ... | ... | ... |
| ... | ... | ... | ... | ... |
| 第500条 | ... | ... | ... | ... |

而 `Y` 是每一条数据对应的答案：

```text
Y.shape = [500]
```

即：

```text
第1条 X → 第1个 Y
第2条 X → 第2个 Y
...
第500条 X → 第500个 Y
```

---

## 7.1 `X[:, 3]` 是什么意思？

如果学过 Python 切片，这个很好理解。

```python
X[:, 3]
```

其中：

```text
:
```

表示：

> 所有行。

而：

```text
3
```

表示第 4 列。

注意 Python 从 0 开始：

```text
0 → 第1列 → x1
1 → 第2列 → x2
2 → 第3列 → x3
3 → 第4列 → x4
```

所以：

```python
X[:, 3]
```

就是：

> 取 500 条数据中的所有 `x4`。

同理：

```python
X[:, 0]
```

就是所有 `x1`。

---

# 8. 第一次画图：scatter 是什么？

代码：

```python
plt.scatter(X[:, 3], Y, 1)
plt.show()
```

### `plt.scatter()`

作用：

> 画散点图。

这里可以理解成：

```python
plt.scatter(横坐标, 纵坐标, 点大小)
```

所以：

```python
plt.scatter(X[:, 3], Y, 1)
```

表示：

```text
横轴：每条数据的 x4
纵轴：每条数据的 y
点大小：1
```

### `plt.show()`

表示：

> 把当前画好的图真正显示出来。

---

# 9. 第二步：把数据分成一小批一小批

代码：

```python
def data_provider(data, label, batchsize):
    length = len(label)
    indices = list(range(length))

    random.shuffle(indices)

    for each in range(0, length, batchsize):
        get_indices = indices[each: each+batchsize]
        get_data = data[get_indices]
        get_label = label[get_indices]

        yield get_data, get_label
```

这个函数相当于一个非常简化版的 PyTorch `DataLoader`。

它的任务是：

> 不要一次把全部 500 条数据都拿去训练，而是每次取一小批。

代码设定：

```python
batchsize = 16
```

所以：

```text
第1批：16条
第2批：16条
第3批：16条
...
```

直到 500 条全部使用一遍。

---

## 9.1 为什么要打乱数据？

```python
indices = list(range(length))
random.shuffle(indices)
```

原来编号可能是：

```text
0 1 2 3 4 5 6 7 ...
```

打乱后可能变成：

```text
231 14 78 3 401 96 ...
```

这意味着每次拿到的数据顺序是随机的。

机器学习训练里经常会这么做。

目前你先记住：

> **随机打乱可以避免模型总按照固定顺序看数据。**

---

# 10. yield 是什么？

这里：

```python
yield get_data, get_label
```

是一个很容易卡住初学者的 Python 语法。

普通函数：

```python
def test():
    return 1
```

执行 `return` 后，函数直接结束。

但是：

```python
yield
```

可以理解成：

> “这次先返回到这里，但把当前位置记下来；下次继续从这里往后运行。”

所以：

```python
for batch_x, batch_y in data_provider(X, Y, batchsize):
```

每循环一次，就能获得下一小批数据。

你目前可以把：

```python
yield
```

理解成：

> **一个可以连续往外“吐”数据的 return。**

这并不严谨，但对当前阶段非常实用。

---

# 11. 第三步：定义模型

代码：

```python
def fun(x, w, b):
    pred_y = torch.matmul(x, w) + b
    return pred_y
```

模型非常简单：

$$
\hat y=Xw+b
$$

这里有个新符号：

$$
\hat y
$$

读作 “y hat”，一般表示：

> **模型预测出来的 y。**

而普通的：

$$
y
$$

表示真实答案。

---

## 11.1 `torch.matmul()` 是什么？

```python
torch.matmul(x, w)
```

表示矩阵乘法。

如果你暂时不想复习线性代数，可以先直接理解成：

> PyTorch 帮我们一次算完  
> `x1*w1 + x2*w2 + x3*w3 + x4*w4`。

也就是：

$$
w_1x_1+w_2x_2+w_3x_3+w_4x_4
$$

然后再：

```python
+b
```

最终得到：

$$
\hat y=w_1x_1+w_2x_2+w_3x_3+w_4x_4+b
$$

---

# 12. 第四步：定义损失函数 MAE

代码：

```python
def maeLoss(pre_y, y):
    return torch.sum(abs(pre_y-y))/len(y)
```

这叫：

> **MAE：Mean Absolute Error，平均绝对误差。**

公式是：

$$
MAE=\frac{1}{n}\sum_{i=1}^{n}|\hat y_i-y_i|
$$

看着很数学，但其实非常简单。

假设真实答案：

```text
10
```

模型预测：

```text
8
```

那么误差：

$$
|8-10|=2
$$

再比如三条数据：

```text
真实值：10   20   30
预测值： 8   23   29
```

绝对误差：

```text
2   3   1
```

平均：

$$
\frac{2+3+1}{3}=2
$$

所以：

```text
MAE = 2
```

---

## 12.1 为什么一定要有 loss？

模型一开始参数是乱猜的。

那计算机怎么知道：

> “我现在这个参数到底好不好？”

必须定义一个数字来衡量。

这个数字就是：

```text
loss
```

一般来说：

```text
loss 越大 → 模型越差
loss 越小 → 模型越好
```

因此训练的目标非常直接：

> **想办法让 loss 越来越小。**

---

# 13. 第五步：什么是梯度下降？

这是整个项目最重要的概念之一。

别急着背公式。

假设现在站在山坡上：

```text
              山顶
               /\
              /  \
             /    \
            /  你  \
___________/________\____
```

你希望走到最低处。

问题是你不知道最低点在哪。

怎么办？

可以看看：

> 当前脚下哪个方向是下坡。

然后往下走一小步。

再重新看看哪个方向下坡。

于是：

```text
看坡度
 ↓
走一步
 ↓
再看坡度
 ↓
再走一步
 ↓
……
```

最后慢慢走到谷底。

机器学习也是这个思想。

只不过：

```text
山的高度
```

变成了：

```text
loss
```

而：

```text
当前位置
```

就是当前：

```text
w 和 b
```

所以我们希望不断修改参数：

```text
w、b
```

让：

```text
loss
```

越来越小。

这就是梯度下降最直观的理解。

---

# 14. sgd 函数逐行解释

代码：

```python
def sgd(paras, lr):
    with torch.no_grad():
        for para in paras:
            para -= para.grad * lr
            para.grad.zero_()
```

SGD 的英文：

```text
Stochastic Gradient Descent
```

中文通常叫：

> 随机梯度下降。

---

## 14.1 `paras`

调用的时候：

```python
sgd([w_0, b_0], lr)
```

所以：

```python
paras
```

就是：

```python
[w_0, b_0]
```

也就是我们想训练的参数。

---

## 14.2 `lr`

```python
lr = 0.03
```

`lr` 是：

```text
learning rate
```

中文：

> 学习率。

它决定每次修改参数时走多大一步。

可以想象下山：

```text
学习率太小：
一步只挪一点点
→ 很慢
```

```text
学习率太大：
一步跨太远
→ 有可能直接跨过最低点
```

因此学习率需要选择一个合适的值。

这里：

```python
lr = 0.03
```

---

## 14.3 参数更新公式

核心：

```python
para -= para.grad * lr
```

等价于：

```python
para = para - para.grad * lr
```

从数学角度大致写成：

$$
w_{\text{new}}
=
w_{\text{old}}
-
lr\times gradient
$$

其中：

```text
gradient
```

就是梯度。

现阶段把梯度理解成：

> **告诉参数“往哪个方向改，loss 会下降”的信息。**

就足够了。

---

## 14.4 为什么有 `torch.no_grad()`？

```python
with torch.no_grad():
```

PyTorch 默认会记录很多数学运算，以便后面自动求导。

但是更新参数：

```python
para -= para.grad * lr
```

只是我们人为修改参数。

这个修改过程不需要再被 PyTorch 记录进求导过程。

所以使用：

```python
with torch.no_grad():
```

告诉 PyTorch：

> 下面这几行只是修改参数，不要追踪梯度。

---

## 14.5 为什么最后要 `zero_()`？

```python
para.grad.zero_()
```

因为 PyTorch 的梯度默认会**累加**。

例如这次梯度：

```text
2
```

如果不清空，下次又算出：

```text
3
```

可能会变成累积：

```text
5
```

但我们这里希望每次更新之后重新计算新的梯度。

所以要：

```python
para.grad.zero_()
```

意思就是：

> 这次梯度已经用完了，清零，下次重新算。

---

# 15. requires_grad=True 是什么？

代码：

```python
w_0 = torch.normal(
    0,
    0.01,
    true_w.shape,
    requires_grad=True
)

b_0 = torch.tensor(
    0.01,
    requires_grad=True
)
```

关键：

```python
requires_grad=True
```

直译：

> 需要梯度。

也就是告诉 PyTorch：

> `w_0` 和 `b_0` 是我要训练的参数，请记录和它们有关的计算过程，后面我要对它们求梯度。

如果参数不用训练，例如真实数据 `X`，通常就不需要：

```python
requires_grad=True
```

---

## 15.1 为什么参数一开始接近 0？

```python
torch.normal(0, 0.01, true_w.shape)
```

生成的是非常接近 0 的随机数。

比如可能：

```text
[-0.004, 0.008, 0.002, -0.006]
```

模型刚开始完全不知道正确答案：

```text
[8.1, 2, 2, 4]
```

于是先从一个随机位置开始。

接下来靠训练一点一点靠近正确参数。

这就是机器学习里非常常见的：

> **参数初始化。**

---

# 16. 第六步：真正开始训练

代码核心：

```python
epochs = 50

for epoch in range(epochs):
    data_loss = 0

    for batch_x, batch_y in data_provider(X, Y, batchsize):
        pred_y = fun(batch_x, w_0, b_0)
        loss = maeLoss(pred_y, batch_y)
        loss.backward()
        sgd([w_0, b_0], lr)
        data_loss += loss

    print("epoch %03d: loss: %.6f" % (epoch, data_loss))
```

这是整份程序真正“学习”的地方。

我们逐行看。

---

## 16.1 外层循环

```python
for epoch in range(epochs):
```

前面：

```python
epochs = 50
```

意味着：

> 把整套训练数据反复学习 50 遍。

---

## 16.2 每轮 loss 先从 0 开始

```python
data_loss = 0
```

用来统计这一轮训练产生的 loss。

---

## 16.3 每次取一批数据

```python
for batch_x, batch_y in data_provider(X, Y, batchsize):
```

因为：

```python
batchsize = 16
```

所以每次拿大约 16 条。

---

## 16.4 根据当前参数进行预测

```python
pred_y = fun(batch_x, w_0, b_0)
```

即：

$$
\hat y=Xw+b
$$

---

## 16.5 计算预测误差

```python
loss = maeLoss(pred_y, batch_y)
```

得到当前这批数据的 MAE。

---

## 16.6 计算梯度

```python
loss.backward()
```

这是 PyTorch 自动求导最关键的一句。

它会根据前面记录的计算过程，算出：

```python
w_0.grad
b_0.grad
```

也就是：

> 当前 loss 对参数变化有多敏感，以及参数应该往哪个方向调整。

---

## 16.7 更新参数

```python
sgd([w_0, b_0], lr)
```

利用刚刚得到的梯度：

```text
修改 w_0
修改 b_0
清空梯度
```

然后下一批数据继续训练。

---

# 17. loss.backward() 到底做了什么？

很多初学者第一次看到：

```python
loss.backward()
```

都会觉得像魔法。

其实可以暂时把它理解成：

> **PyTorch 自动帮你求导。**

你在高数中学过类似：

$$
y=x^2
$$

那么：

$$
\frac{dy}{dx}=2x
$$

机器学习中公式可能变得很长。

例如：

```text
w、b
 ↓
预测值
 ↓
误差
 ↓
loss
```

我们最终想知道：

```text
w 改一点，loss 会怎么变化？
b 改一点，loss 会怎么变化？
```

也就是类似：

$$
\frac{\partial Loss}{\partial w}
$$

以及：

$$
\frac{\partial Loss}{\partial b}
$$

手算会很麻烦。

PyTorch 的自动求导系统会帮我们完成。

因此：

```python
loss.backward()
```

之后就能访问：

```python
w_0.grad
b_0.grad
```

---

## 17.1 训练里最重要的四连

建议把下面四步记熟：

```python
pred_y = fun(...)
loss = maeLoss(...)
loss.backward()
sgd(...)
```

它们分别是：

```text
① 前向计算：模型做预测
② 计算 loss：看看错多少
③ backward：求梯度
④ 更新参数：让下一次更准
```

以后学真正的神经网络，这个结构仍然存在。

---

# 18. epoch、batch、learning rate 一次讲清楚

这是以后会不停见到的三个词。

---

## Batch

```python
batchsize = 16
```

意思：

> 每次拿 16 条数据训练。

例如 500 条数据：

```text
500 ÷ 16 ≈ 31.25
```

所以一轮大约会产生 32 个 batch。

---

## Epoch

```python
epochs = 50
```

一个 epoch：

> 全部 500 条数据都学习一遍。

50 epoch：

> 全部数据反复学习 50 遍。

---

## Learning Rate

```python
lr = 0.03
```

学习率：

> 每次根据梯度修改参数时，修改多大一步。

---

## 一个比喻

把备考一本题库想成：

- `dataset`：整本题库
- `batch`：一次做 16 道题
- `epoch`：整本题库刷完一次
- `epochs=50`：整本刷 50 遍
- `loss`：每次错了多少
- `gradient`：告诉你哪类知识最需要改
- `learning rate`：每次调整学习策略的幅度

---

# 19. 最后两个 plt 函数到底在画什么？

最后：

```python
idx = 0

plt.plot(
    X[:, idx].detach().numpy(),
    X[:, idx].detach().numpy()
    * w_0[idx].detach().numpy()
    + b_0.detach().numpy()
)

plt.scatter(X[:, idx], Y, 1)

plt.show()
```

这里最重要的是：

```python
plt.plot(...)
plt.scatter(...)
```

---

## 19.1 `plt.plot()`：画线

最基础形式：

```python
plt.plot(x, y)
```

意思：

> 根据一组 x 和 y，把点连接起来形成线。

例如：

```python
plt.plot([1, 2, 3], [2, 4, 6])
```

大概对应：

$$
y=2x
$$

---

## 19.2 当前项目的 `plt.plot()`

因为：

```python
idx = 0
```

所以：

```python
X[:, idx]
```

就是：

```python
X[:, 0]
```

也就是所有 `x1`。

而：

```python
X[:, idx] * w_0[idx] + b_0
```

相当于：

$$
y=w_1x_1+b
$$

所以这条 `plot` 画的是：

> **只看第一个特征 x1 时，根据训练得到的 w1 和 b 得到的一条直线。**

---

## 19.3 `plt.scatter()`：画散点

```python
plt.scatter(X[:, idx], Y, 1)
```

三个主要参数：

```text
X[:, idx] → 横坐标
Y         → 纵坐标
1         → 点的大小
```

所以它画的是：

> 每条真实数据对应的散点。

---

## 19.4 `plt.show()`：显示图像

前面的：

```python
plt.plot(...)
plt.scatter(...)
```

相当于不断往画布上加东西。

最后：

```python
plt.show()
```

才会真正显示窗口。

---

# 20. detach().numpy() 为什么这么长？

最后画图中有：

```python
w_0[idx].detach().numpy()
```

拆成两个部分。

---

## 20.1 `.detach()`

训练时 PyTorch 一直在记录计算过程，因为参数需要求梯度。

但是现在已经训练完了。

画图只想使用数值，不想再参与求梯度。

所以：

```python
.detach()
```

可以先理解成：

> **把这个 Tensor 从自动求导系统中暂时拿出来，只看它的数值。**

---

## 20.2 `.numpy()`

PyTorch 中的数据类型主要是：

```text
Tensor
```

NumPy 中的数据类型主要是：

```text
ndarray
```

所以：

```python
.numpy()
```

表示：

> 把 PyTorch Tensor 转换成 NumPy 数组。

---

## 20.3 连起来

```python
.detach().numpy()
```

初学阶段可以直接记：

> **训练结束后，把 Tensor 安全地拿出来转换成普通 NumPy 数据，方便画图。**

---

# 21. 为什么最后的点没有紧紧贴着直线？

这是这份代码特别容易让人误会的一点。

真实关系是：

$$
y=8.1x_1+2x_2+2x_3+4x_4+1.1+\text{noise}
$$

也就是说：

```text
Y
```

同时受到：

```text
x1
x2
x3
x4
```

四个变量影响。

但是最后画图时：

```python
idx = 0
```

只把：

```text
x1
```

放在横轴上。

画的线也是：

$$
y=w_1x_1+b
$$

它没有把：

$$
w_2x_2+w_3x_3+w_4x_4
$$

画进去。

因此你看到的散点很可能比较散。

这**不一定表示模型训练失败**。

真正判断模型有没有学好，更直接的方法是比较：

```python
print("真实的函数值是", true_w, true_b)
print("训练得到的参数值是", w_0, b_0)
```

如果结果大致像：

```text
真实：
[8.1, 2, 2, 4]
1.1

训练：
[8.0..., 2.0..., 2.0..., 4.0...]
1.0...
```

说明模型确实学到了接近的参数。

---

# 22. 完整训练流程图

把这段程序压缩成一张图：

```text
真正参数
true_w = [8.1, 2, 2, 4]
true_b = 1.1
          │
          ▼
   create_data()
          │
          ▼
      生成 X、Y
          │
          ▼
   data_provider()
   每次拿 16 条
          │
          ▼
       fun()
  pred_y = Xw + b
          │
          ▼
     maeLoss()
   计算预测误差
          │
          ▼
    loss.backward()
      自动求梯度
          │
          ▼
        sgd()
     更新 w 和 b
          │
          ▼
   下一批数据继续
          │
          ▼
  全部数据学完一次
      = 1 epoch
          │
          ▼
       重复 50 次
          │
          ▼
     得到训练参数
          │
          ▼
 与 true_w、true_b 比较
          │
          ▼
     matplotlib 画图
```

以后看复杂神经网络，也可以先找这几个模块。

---

# 23. 这段代码里最值得记住的 10 个东西

如果一次学不完，优先记下面这些。

### 1. Tensor

PyTorch 用来存数据的基本对象。

```python
torch.tensor(...)
```

---

### 2. shape

表示数据长什么样。

例如：

```text
X.shape = (500, 4)
```

表示：

```text
500 条数据，每条 4 个特征。
```

---

### 3. 模型

当前模型：

$$
\hat y=Xw+b
$$

---

### 4. 参数

```text
w、b
```

是模型需要通过训练学出来的东西。

---

### 5. loss

衡量预测有多差。

这里用：

```text
MAE
```

---

### 6. gradient

告诉参数：

> 往哪个方向调整，loss 会下降。

---

### 7. backward

```python
loss.backward()
```

让 PyTorch 自动计算梯度。

---

### 8. learning rate

```python
lr
```

决定每次参数修改多大一步。

---

### 9. batch

一次拿多少条数据训练。

---

### 10. epoch

整个训练集完整学习一次。

---

# 24. 初学者可以做的几个小实验

强烈建议不要只看代码。

改几个数字，会比背概念有效得多。

---

## 实验 1：修改真实参数

把：

```python
true_w = torch.tensor([8.1, 2, 2, 4])
```

改成：

```python
true_w = torch.tensor([3.0, 2.0, 1.0, 5.0])
```

重新运行。

看看最后：

```text
训练得到的参数
```

会不会跟着变化。

目的：

> 理解模型是在“学习参数”。

---

## 实验 2：减少 epoch

把：

```python
epochs = 50
```

改成：

```python
epochs = 1
```

再看训练结果。

然后尝试：

```python
epochs = 5
epochs = 20
epochs = 100
```

观察参数和 loss。

目的：

> 理解训练轮数对学习结果的影响。

---

## 实验 3：修改学习率

原来：

```python
lr = 0.03
```

分别试：

```python
lr = 0.001
```

和：

```python
lr = 0.3
```

观察训练速度或稳定性。

注意：

> 学习率并不是越大越好。

---

## 实验 4：修改 batch size

原来：

```python
batchsize = 16
```

可以试：

```python
batchsize = 1
batchsize = 32
batchsize = 128
batchsize = 500
```

观察训练。

以后你会经常见到：

```text
batch_size=32
batch_size=64
batch_size=128
```

---

## 实验 5：只用一个特征制造数据

如果你特别想直观看懂最后画出来的直线，可以临时把问题变简单。

例如：

```python
true_w = torch.tensor([8.1])
```

这样模型就是：

$$
y=8.1x+1.1
$$

此时二维图最容易理解。

对于第一次学习线性回归的人，这个实验非常推荐。

---

# 25. 可选优化：让代码更适合继续学习

下面不是说原代码“错了”，而是一些更适合后续学习的写法。

---

## 25.1 统计 loss 时使用 `.item()`

原代码：

```python
data_loss += loss
```

可以改成：

```python
data_loss += loss.item()
```

`.item()` 的作用：

> 从只包含一个数的 Tensor 中拿出普通 Python 数值。

例如：

```python
loss
```

可能是：

```text
tensor(0.4312, grad_fn=<...>)
```

而：

```python
loss.item()
```

得到：

```text
0.4312
```

对于只想做打印统计的值，这样更直观。

---

## 25.2 打印平均 batch loss

现在 `data_loss` 是这一轮所有 batch loss 的累计。

如果想更容易比较每个 epoch，可以统计平均值。

例如：

```python
batch_count = 0
data_loss = 0

for batch_x, batch_y in data_provider(X, Y, batchsize):
    pred_y = fun(batch_x, w_0, b_0)
    loss = maeLoss(pred_y, batch_y)

    loss.backward()
    sgd([w_0, b_0], lr)

    data_loss += loss.item()
    batch_count += 1

print(data_loss / batch_count)
```

这样每轮的数字更容易理解。

---

## 25.3 最后画线时先排序 x

原代码直接：

```python
plt.plot(X[:, idx], ...)
```

由于 `X[:, idx]` 是随机顺序，`plot` 会按照数据现有顺序把点依次连接。

有时视觉上可能出现来回折线。

如果想画得更像正常的函数曲线，可以先按 x 排序。

例如：

```python
x_plot = X[:, idx].detach().numpy()
y_plot = (
    X[:, idx] * w_0[idx] + b_0
).detach().numpy()

order = x_plot.argsort()

plt.plot(x_plot[order], y_plot[order])
plt.scatter(X[:, idx], Y, 1)
plt.show()
```

不过要注意：

> 这依然只是画 `x1` 和 `w1*x1+b`，而完整模型实际上有 4 个输入。

---

# 26. 如何运行这个项目

假设文件名：

```text
mylinear.py
```

首先确认 Python 已安装。

然后安装依赖：

```bash
pip install torch matplotlib
```

进入文件所在目录：

```bash
cd 你的代码目录
```

运行：

```bash
python mylinear.py
```

程序会：

1. 生成数据；
2. 先弹出一张散点图；
3. 关闭第一张图后开始训练；
4. 在终端打印每个 epoch 的 loss；
5. 输出真实参数和训练参数；
6. 最后弹出训练结果相关图像。

注意：

```python
plt.show()
```

通常会阻塞代码继续运行。

所以第一张图出现后，需要把图窗关闭，程序才会继续向下训练。

---

# 27. 学完这份代码之后该学什么？

如果你目前只会 Python 和基础数学，不建议马上冲复杂神经网络。

比较顺的顺序是：

```text
当前代码：手写线性回归
        ↓
理解 Tensor / shape
        ↓
理解 loss
        ↓
理解 backward 和梯度
        ↓
学 nn.Linear
        ↓
学 nn.Module
        ↓
学 torch.optim.SGD
        ↓
学 Dataset / DataLoader
        ↓
写一个完整的 PyTorch 训练模板
        ↓
再进入多层感知机 / CNN 等网络
```

---

# 最后：你真正需要理解到什么程度？

第一次学这份代码，不需要做到：

- 自己推完所有偏导公式；
- 会证明梯度下降；
- 会推正态分布；
- 熟练矩阵微积分；
- 一眼看懂 PyTorch 自动求导底层。

你当前阶段做到下面这些，就已经足够：

1. 知道 `X` 是输入数据，`Y` 是答案；
2. 知道模型根据 `w、b` 计算预测；
3. 知道 loss 衡量预测有多差；
4. 知道 `loss.backward()` 是在自动求梯度；
5. 知道 SGD 根据梯度更新参数；
6. 知道一次 batch 是一小批数据；
7. 知道一个 epoch 是所有数据学一遍；
8. 知道训练的本质是在不断调整参数，让 loss 下降；
9. 知道 `plt.scatter()` 画点；
10. 知道 `plt.plot()` 画线。

如果这十条能用自己的话解释出来，那么这份代码的核心你就已经真正入门了。

---

## 一句话总结

这份 `mylinear.py` 并不是在教一个复杂的人工智能模型。

它真正重要的是让你第一次完整看到：

$$
\boxed{
数据
\rightarrow
预测
\rightarrow
计算误差
\rightarrow
自动求梯度
\rightarrow
更新参数
\rightarrow
重复训练
}
$$

以后无论是线性回归、神经网络、CNN，还是更复杂的深度学习模型，训练代码的骨架都离不开这条主线。
