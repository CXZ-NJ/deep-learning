# Python 语法速补：从"会基础语法"到"能读懂深度学习代码"

> 适合读者：会 `if`、`for`、`def`、`list` 等基础语法，但没系统学过类、魔法方法等进阶语法。  
> 目标：把本仓库两个 PyTorch 项目（`01_linear_regression/mylinear.py`、`02_covid_regression/covid_regression.py`）里用到的**所有进阶语法点**逐个讲透。  
> 为什么需要这篇：深度学习代码大量使用"类 + 魔法方法 + 库函数"，看不懂这些，代码当然记不住。语法不是看会的，是**逐个查、用三次就记住**的。

---

## 目录

1. [一句话总览表](#1-一句话总览表)
2. [类的本质：数据 + 方法绑在一起](#2-类的本质数据--方法绑在一起)
3. [继承：在别人写好的类上加东西](#3-继承在别人写好的类上加东西)
4. [魔法方法：Python 自动调用的"回调点"](#4-魔法方法python-自动调用的回调点)
5. [super()：先执行父类的初始化](#5-super先执行父类的初始化)
6. [切片：取矩阵的一部分](#6-切片取矩阵的一部分)
7. [列表推导式：一行写"筛选列表"](#7-列表推导式一行写筛选列表)
8. [三元表达式：一行 if-else](#8-三元表达式一行-if-else)
9. [with 语句：进去做事，出来自动收尾](#9-with-语句进去做事出来自动收尾)
10. [for 循环解包 + enumerate](#10-for-循环解包--enumerate)
11. [函数也是对象：函数可以当参数传](#11-函数也是对象函数可以当参数传)
12. [默认参数：不传就用默认值](#12-默认参数不传就用默认值)
13. [生成器 yield：边算边给（mylinear.py 里的 data_provider）](#13-生成器-yield边算边给mylinearpy-里的-data_provider)
14. [`__name__ == "__main__"`：只有直接运行才执行](#14-__name__--__main__只有直接运行才执行)
15. [字符串格式化：往字符串里填数字](#15-字符串格式化往字符串里填数字)
16. [优先级：先学哪几个？](#16-优先级先学哪几个)
17. [自测题](#17-自测题)

---

# 1. 一句话总览表

先把本仓库代码里用到的进阶语法点全部列出来，心里有个地图，后面逐条讲：

| # | 语法 | 出现在 | 一句话解释 |
|---|---|---|---|
| 1 | 类 `class` | `class CovidDataset(Dataset)` | 把数据 + 操作数据的函数打包成一个东西 |
| 2 | 继承 `(Dataset)` | `class MyModel(nn.Module)` | 在别人写好的类上，加自己的功能 |
| 3 | 魔法方法 | `__init__` `__getitem__` `__len__` | 名字带双下划线，Python 自动调用 |
| 4 | `super()` | `super().__init__()` | 先执行父类的初始化代码 |
| 5 | 切片 | `[1:, 1:]` `[:, :93]` `[-1]` | 取出矩阵的一部分 |
| 6 | 列表推导式 | `[i for i in range(...) if ...]` | 一行写出"筛选列表" |
| 7 | 三元表达式 | `"cuda" if ... else "cpu"` | 一行 if-else |
| 8 | `with` 语句 | `with open(...)` `with torch.no_grad():` | 进去自动做事，出来自动收尾 |
| 9 | 循环解包 | `for x, y in train_loader:` | 每次循环一次拿两个值 |
| 10 | `enumerate` | `for i, pred in enumerate(...)` | 同时拿到"第几个"和"值" |
| 11 | 函数当参数 | `loss_fn = mse_loss` | 函数能像变量一样传来传去 |
| 12 | 默认参数 | `mean=None, std=None` | 调用时不传，就用默认值 |
| 13 | 生成器 `yield` | `data_provider`（线性回归） | 边算边给，不用一次性生成全部 |
| 14 | `__name__` | 文件最后两行 | "只有直接运行这个文件时才执行" |
| 15 | 格式化 | `"%03d" % epoch` | 往字符串里填数字 |

---

# 2. 类的本质：数据 + 方法绑在一起

**你会的基础语法**：变量存数据，函数操作数据，两者是分开的：

```python
x = [1, 2, 3]
def double(lst):
    return [i * 2 for i in lst]
```

**类（class）做的事**：把"数据"和"操作这批数据的函数"打包成一个整体，这个整体叫"对象"（或"实例"）。

看 COVID 项目：

```python
class CovidDataset(Dataset):
    def __init__(self, ...):
        # 这里把 X、Y、mean、std 都装进 self
        self.X = ...
        self.Y = ...
        self.mean = ...
```

`train_set` 一创建出来，就是一个"整理好的数据集"：里面装着数据（X、Y、mean、std），还装着取数据的方法（`__getitem__`、`__len__`）。

## 2.1 `self` 是什么？

```python
def __init__(self, file_path, mode, mean=None, std=None):
    self.X = ...
```

`self` 就是"这个对象自己"。`self.X = ...` 的意思是：**把这个 X 记在这个对象身上**，以后 `train_set.X` 就能取出来。

**类比**：`self` 相当于每个学生自己的名字——`self.X` 是"我的 X"，`self.Y` 是"我的 Y"。同一个类可以创建很多对象（train_set、val_set、test_set），每个对象通过 `self` 区分"谁的"数据。

## 2.2 创建对象时发生了什么？

```python
train_set = CovidDataset(train_file, "train")
```

这行代码：
1. 分配一块内存，准备装这个对象
2. 自动调用 `__init__`，执行里面的所有代码（读文件、划分、标准化）
3. 把初始化好的对象赋给 `train_set`

所以 `__init__` 就是"对象出生时做的事"——你只需要在类里写好它，创建对象时 Python 自动帮你调用。

---

# 3. 继承：在别人写好的类上加东西

```python
class CovidDataset(Dataset):
```

`Dataset` 是 PyTorch 写好的类。`CovidDataset(Dataset)` 表示：

> 新类 CovidDataset 是 Dataset 的"子类"，它拥有 Dataset 的所有能力，再额外加自己的内容。

**类比**：`Dataset` 是"汽车底盘"，PyTorch 造好的；`CovidDataset` 是在底盘上装车厢、方向盘，造出"救护车"。

为什么会报"TypeError: Can't instantiate abstract class"之类错误？因为 PyTorch 规定：继承 Dataset 必须实现 `__getitem__` 和 `__len__`，不实现就不能创建对象（这就是第 4 节魔法方法的作用）。

同样的：

```python
class MyModel(nn.Module):
```

继承 PyTorch 的神经网络基类，得到"自动求导、保存加载、移动到 GPU"等能力，自己只需要定义网络结构（`__init__`）和计算过程（`forward`）。

---

# 4. 魔法方法：Python 自动调用的"回调点"

方法名首尾带双下划线的叫魔法方法（dunder method）。

**核心规则：你定义它，Python 自动调用它，你永远不直接调。**

| 魔法方法 | 触发方式 | 例子 |
|---|---|---|
| `__init__` | 创建对象时 | `CovidDataset(...)` |
| `__len__` | 调用 `len(x)` 时 | `len(train_set)` |
| `__getitem__` | 用下标 `x[i]` 时 | `train_set[3]` |

所以代码里：

```python
def __len__(self):
    return len(self.X)
```

你写的其实是"Python 问我一共有多少数据时，我该回答什么"。`len(train_set)` 能工作，是因为 Python 内部帮你翻译成了 `train_set.__len__()`。

**类比**：魔法方法是给 Python 留的"电话"，Python 到点就拨过来：`len()` 拨 `__len__`，`x[3]` 拨 `__getitem__`，创建对象拨 `__init__`。

为什么深度学习代码到处是魔法方法？因为 PyTorch 的规范就是：你按约定写好这几个方法，框架就能把你的类"接进"训练流程（DataLoader 用 `__getitem__` 取数据、用 `__len__` 算总量）。

---

# 5. super()：先执行父类的初始化

```python
class MyModel(nn.Module):
    def __init__(self, in_dim):
        super().__init__()

        self.fc1 = nn.Linear(in_dim, 128)
        ...
```

`super().__init__()` 意思是：

> 先调用父类（nn.Module）的 `__init__`，把父类准备好的功能激活，然后再做自己的事。

为什么要这样？因为 `nn.Module` 内部要注册参数、记录子层等，你不先激活它，后面的功能（`model(x)`、`.to(device)`、`parameters()`）全部不工作。

**规则**：凡是继承别人写的类，且自己重写了 `__init__`，第一行基本都要写 `super().__init__()`。

---

# 6. 切片：取矩阵的一部分

Python 列表和 NumPy/PyTorch 张量都支持切片，格式是 `[起始:结束]`（含起始，不含结束）：

```python
data[1:, 1:]      # 去掉第0行和第0列
data[:, :93]      # 所有行，前93列
data[:, -1]       # 所有行，最后一列（-1 表示倒数第一）
indices[each : each + batchsize]   # 从 each 到 each+batchsize
```

| 写法 | 意思 |
|---|---|
| `a[1:]` | 从第 1 个到末尾 |
| `a[:5]` | 从开头到第 4 个（不含 5） |
| `a[1:3]` | 第 1、2 个 |
| `a[-1]` | 最后一个 |
| `a[:, :93]` | 二维里：所有行 + 前 93 列 |

COVID 数据是"行 = 样本、列 = 特征"，所以：

```python
csv_data[indices, :93]   # 挑出 indices 这些行，取前 93 列作特征
csv_data[indices, -1]    # 挑出这些行，取最后一列作答案
```

**类比**：切片就是"从表格里划一块出来"。

---

# 7. 列表推导式：一行写"筛选列表"

```python
indices = [i for i in range(len(csv_data)) if i % 5 != 0]
```

拆开看就是：

```python
indices = []
for i in range(len(csv_data)):
    if i % 5 != 0:          # i 不能被 5 整除
        indices.append(i)
```

公式：

```text
[要存的东西  for 变量 in 序列  if 条件]
```

三种变体：

| 写法 | 结果 |
|---|---|
| `[i for i in range(5)]` | `[0, 1, 2, 3, 4]` |
| `[i * 2 for i in range(5)]` | `[0, 2, 4, 6, 8]` |
| `[i for i in range(5) if i % 2 == 0]` | `[0, 2, 4]` |

**类比**：一条流水线——把原料（range）扫一遍，满足条件（if）的加工（`要存的东西`）后丢进成品箱。

---

# 8. 三元表达式：一行 if-else

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

等价于：

```python
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
```

公式：

```text
A if 条件 else B
```

条件成立取 A，否则取 B。适合"二选一赋值"这种简单场景，逻辑复杂时别用，普通 if 更清楚。

---

# 9. with 语句：进去做事，出来自动收尾

```python
with open(file_path, "r") as f:
    ori_data = list(csv.reader(f))
```

等价于：

```python
f = open(file_path, "r")
try:
    ori_data = list(csv.reader(f))
finally:
    f.close()          # 无论中间有没有报错，都关闭文件
```

`with` 帮你保证：**用完自动关闭文件**，即使中间报错也会关，不会泄漏资源。

PyTorch 里的典型用法：

```python
with torch.no_grad():
    for val_x, val_y in val_loader:
        ...
```

进入 `with` 时：关闭梯度记录。退出 `with` 时：自动恢复梯度记录。你只管"这段不记梯度"，进出都交给 `with`。

**类比**：`with` 像一个"限时自习室"——进门（with）登记"不计算梯度"，出门自动恢复原状。Python 保证你一定"出得了这个门"。

---

# 10. for 循环解包 + enumerate

**解包（unpacking）**：把"一包"值拆开分别赋值：

```python
x, y = (1, 2)          # x=1, y=2
a, b = [3, 4]          # a=3, b=4
```

在 for 里配合使用：

```python
for x, y in train_loader:
```

每次循环，DataLoader 给出一批数据 `(输入, 答案)`，解包后 `x` 拿输入、`y` 拿答案。

**enumerate**：同时拿到"第几个"和"值"：

```python
for i, pred in enumerate(predictions):
```

等价于：

```python
i = 0
for pred in predictions:
    ...
    i += 1
```

`enumerate(列表)` 每次给出一对 `(序号, 值)`，配合解包就是 `for i, pred in ...`。

---

# 11. 函数也是对象：函数可以当参数传

```python
loss_fn = mse_loss
```

注意 `mse_loss` **后面没有括号**——没有括号就不是"调用"，而是"把函数存进变量"。

然后：

```python
def train_val(model, ..., loss_fn, ...):
    ...
    batch_loss = loss_fn(y_pred, y, model)   # 这里才真正调用
```

`train_val` 收到 `loss_fn` 后，在内部调用它。效果是：**训练函数不关心"损失函数具体怎么算"，你传什么它就用什么**。以后你想换损失函数，只需改 `loss_fn = xxx`，训练代码一行不用动。

**类比**：函数名 = 手机号码。`loss_fn = mse_loss` 是把号码存进通讯录；`loss_fn(...)` 是拨号。存号码不花话费，拨号才花。

Python 里函数、类都可以这样传来传去（它们都是"对象"）。这是函数式风格的基石，深度学习代码里到处都是。

---

# 12. 默认参数：不传就用默认值

```python
def __init__(self, file_path, mode, mean=None, std=None):
```

调用时可以只传前两个：

```python
train_set = CovidDataset(train_file, "train")
# mean、std 自动用默认值 None
```

也可以传全部：

```python
val_set = CovidDataset(train_file, "val", mean=train_set.mean, std=train_set.std)
```

**规则**：带默认值的参数必须放在没有默认值的参数**后面**，否则报错。

本项目里这个设计很巧妙：训练集自己算 mean/std（默认 None 就行）；验证/测试集必须传入（代码里检查了 None 就报错）——**用默认参数 + 报错检查，强制你遵守"只用训练集统计量"的规范**。

---

# 13. 生成器 yield：边算边给（mylinear.py 里的 data_provider）

线性回归项目里有：

```python
def data_provider(data, label, batchsize):
    length = len(label)
    indices = list(range(length))
    random.shuffle(indices)

    for each in range(0, length, batchsize):
        get_indices = indices[each: each+batchsize]
        get_data = data[get_indices]
        get_label = label[get_indices]

        yield get_data, get_label    # 关键：yield 而不是 return
```

`yield` 和 `return` 的区别：

| | return | yield |
|---|---|---|
| 调用后 | 返回一个值，函数结束 | 返回一个值，**函数暂停在这里** |
| 再次调用 | 从开头重新执行 | **从暂停处继续** |
| 结果 | 普通值 | 生成器对象 |

```python
for batch_x, batch_y in data_provider(X, Y, batchsize):
    # 每轮循环，函数从上次的 yield 处继续，吐出下一批
```

**类比**：自动售货机。`return` 是一次性把仓库搬给你；`yield` 是投一次币掉一罐，掉完再投，仓库里的货都是"边算边给"。

好处：数据有 10 万条时，不用一次性全部放进内存，而是一次一批。PyTorch 的 DataLoader 内部也是这个思路。

---

# 14. `__name__ == "__main__"`：只有直接运行才执行

```python
if __name__ == "__main__":
    main()
```

每份 Python 文件被运行时，Python 都会设置一个隐藏变量 `__name__`：

| 场景 | `__name__` 的值 |
|---|---|
| `python covid_regression.py` 直接运行 | `"__main__"` |
| 被别的文件 `import` | 文件名（如 `"covid_regression"`） |

所以这行的意思是：

> 只有"直接运行这个文件"时才执行 main()；如果这个文件是被 import 的，就不执行。

**作用**：以后你写工具函数文件时，别人 import 你的函数不会把你的测试代码跑一遍。

---

# 15. 字符串格式化：往字符串里填数字

```python
print("[%03d/%03d] %.2f sec(s)  train_loss: %.6f  val_loss: %.6f" % (epoch + 1, epochs, time.time() - start_time, train_loss, val_loss))
```

`%` 是"占位符"，后面括号里按顺序填值：

| 占位符 | 含义 | 例子 |
|---|---|---|
| `%d` | 整数 | `%03d` = 至少3位，不够补0（`5` → `005`） |
| `%f` | 小数 | `%.2f` = 保留2位小数；`%.6f` = 保留6位 |
| `%s` | 字符串 | `"结果保存到了 %s" % path` |

现在更常见的写法是 f-string（Python 3.6+）：

```python
print(f"[{epoch+1:03d}/{epochs:03d}] {time.time()-start_time:.2f} sec(s)  train_loss: {train_loss:.6f}")
```

两者效果一样，博主代码用了 `%` 风格，你能看懂即可；自己写建议用 f-string，更直观。

---

# 16. 优先级：先学哪几个？

| 优先级 | 语法 | 理由 |
|---|---|---|
| ⭐⭐⭐ 必学 | 类 + `self`、魔法方法、切片、for 解包、`with` | 本项目出现频率最高，深度学习通用 |
| ⭐⭐ 要懂 | 继承、`super()`、默认参数、三元表达式、函数当参数 | 看懂代码必须，写的时候可查 |
| ⭐ 了解即可 | `yield`、`__name__`、格式化 | 知道存在，遇到再深究 |

**方法建议**：不要先花一个月学完 Python 再来——深度学习用到什么查什么，同一个语法在代码里出现 3 次以上自然就记住了。这 15 个点，本项目已经帮你"复现"了一遍，对照着代码看就是最好的练习。

---

# 17. 自测题

不用写代码，用大白话回答：

1. `len(train_set)` 能工作，Python 背后实际调用了哪个方法？
2. `self.X = X` 里的 `self` 是什么？换成 `train_set.X` 能取出什么？
3. `with torch.no_grad():` 里那段代码结束后，梯度记录会怎样？
4. `loss_fn = mse_loss` 为什么不加括号？`loss_fn(...)` 和 `mse_loss(...)` 是什么关系？
5. `"cuda" if torch.cuda.is_available() else "cpu"` 等价于几行 if-else？
6. `csv_data[:, -1]` 取的是哪部分数据？
7. `yield` 和 `return` 最大的区别是什么？
8. 自己写的文件被别的代码 import 时，`__name__` 的值是什么？

答案都在上面各节里，答不出来的回到对应小节再看一遍。
