# 从零看懂 PyTorch：BERT 中文酒店评论二分类

> 本项目是在前面回归与图像分类项目基础上的进一步学习。
>
> 前三个项目主要处理：
>
> ```text
> 数值数据
> 图像数据
> ```
>
> 第四个项目开始处理：
>
> ```text
> 自然语言文本
> ```
>
> 项目使用 `bert-base-chinese` 对酒店评论进行二分类。  
> 重点学习 **文本数据读取、Tokenizer、BERT 输入格式、预训练模型、BERT Encoder、pooled output、文本分类、AdamW、学习率调度器、梯度裁剪** 等知识。
>
> 本文完全按照当前项目代码结构展开：
>
> ```text
> main.py
> model_utils/
> ├── data.py
> ├── model.py
> └── train.py
> ```
>
> 并结合 `jiudian.txt` 数据集解释整个数据流和训练流程。

---

## 目录

- [1. 项目任务：酒店评论二分类](#sec-1)
- [2. 第四个项目相对于前三个项目新增了什么](#sec-2)
- [3. 第四个项目完整结构](#sec-3)
- [4. 整个项目运行流程](#sec-4)
- [5. `main.py`：项目总入口](#sec-5)
- [6. 固定随机种子](#sec-6)
- [7. 项目主要超参数](#sec-7)
- [8. `jiudian.txt` 数据集](#sec-8)
- [9. 为什么要对数据进行抽样](#sec-9)
- [10. `data.py`：读取文本数据](#sec-10)
- [11. `split(",", 1)` 为什么要写 1](#sec-11)
- [12. 自定义 `JdDataset`](#sec-12)
- [13. 文本 Dataset 与图像 Dataset 的区别](#sec-13)
- [14. `train_test_split` 划分训练集和验证集](#sec-14)
- [15. `stratify=label` 是什么](#sec-15)
- [16. BERT 项目中的 DataLoader](#sec-16)
- [17. `model.py`：BERT 分类模型](#sec-17)
- [18. BERT 是什么](#sec-18)
- [19. BERT 与 Transformer 的关系](#sec-19)
- [20. `BertModel.from_pretrained()`](#sec-20)
- [21. 什么是预训练模型](#sec-21)
- [22. `BertTokenizer`](#sec-22)
- [23. Tokenizer 做了什么](#sec-23)
- [24. `input_ids`](#sec-24)
- [25. `[CLS]` 和 `[SEP]`](#sec-25)
- [26. `attention_mask`](#sec-26)
- [27. `token_type_ids`](#sec-27)
- [28. `padding='max_length'`](#sec-28)
- [29. `truncation=True`](#sec-29)
- [30. `max_length=128`](#sec-30)
- [31. BERT 输入 Tensor 的形状](#sec-31)
- [32. `build_bert_input()` 完整流程](#sec-32)
- [33. BERT 前向传播](#sec-33)
- [34. `sequence_out` 是什么](#sec-34)
- [35. `pooled_output` 是什么](#sec-35)
- [36. 为什么分类使用 `pooled_output`](#sec-36)
- [37. `Linear(768, 2)`](#sec-37)
- [38. BERT 分类模型完整数据流](#sec-38)
- [39. 为什么模型最后不写 Softmax](#sec-39)
- [40. `CrossEntropyLoss`](#sec-40)
- [41. `train.py`：训练函数](#sec-41)
- [42. 一个 Batch 的训练过程](#sec-42)
- [43. `model.zero_grad()`](#sec-43)
- [44. `loss.backward()`](#sec-44)
- [45. 梯度裁剪 `clip_grad_norm_`](#sec-45)
- [46. `optimizer.step()`](#sec-46)
- [47. AdamW 优化器](#sec-47)
- [48. Weight Decay](#sec-48)
- [49. 学习率调度器 Scheduler](#sec-49)
- [50. `CosineAnnealingWarmRestarts`](#sec-50)
- [51. 训练 Loss 的统计](#sec-51)
- [52. 训练 Accuracy 的统计](#sec-52)
- [53. 验证阶段](#sec-53)
- [54. `model.eval()` 与 `torch.no_grad()`](#sec-54)
- [55. 保存最佳模型](#sec-55)
- [56. 为什么设置 `max_acc=0.85`](#sec-56)
- [57. 训练曲线](#sec-57)
- [58. `model_Datapara()` 与多 GPU](#sec-58)
- [59. 当前项目中没有实际使用的配置](#sec-59)
- [60. 当前代码几个需要特别理解的地方](#sec-60)
- [61. 第四个项目完整调用关系](#sec-61)
- [62. 第四个项目完整数据流](#sec-62)
- [63. 第四个项目与第三个项目对比](#sec-63)
- [64. 本项目必须掌握的问题](#sec-64)
- [65. 推荐学习顺序](#sec-65)
- [66. 项目总结](#sec-66)

---

<a id="sec-1"></a>

# 1. 项目任务：酒店评论二分类

这个项目的任务是：

> **输入一条中文酒店评论，判断它属于两个类别中的哪一类。**

数据文件：

```text
jiudian.txt
```

格式：

```text
label,review
```

例如：

```text
1,酒店位置不错，服务很好
```

可以理解为：

```text
label  → 类别标签
review → 中文评论文本
```

所以模型需要学习：

```text
中文评论
   ↓
BERT
   ↓
文本特征
   ↓
分类层
   ↓
类别 0 / 类别 1
```

这是一个：

> **二分类文本分类任务**

---

<a id="sec-2"></a>

# 2. 第四个项目相对于前三个项目新增了什么

前三个项目已经逐步学习了：

```text
项目一：
Tensor
Loss
Backward
Optimizer

项目二：
Dataset
DataLoader
Batch
Train / Validation

项目三：
图像 Tensor
CNN
BatchNorm
CrossEntropyLoss
Accuracy
预训练视觉模型
```

第四个项目最重要的新知识是：

```text
自然语言文本
↓
Tokenizer
↓
Token ID
↓
BERT
↓
Transformer Encoder
↓
文本特征
↓
分类
```

整体学习路线：

```text
原始中文字符串
    ↓
BertTokenizer
    ↓
input_ids
attention_mask
token_type_ids
    ↓
BertModel
    ↓
pooled_output
    ↓
Linear
    ↓
2 个 Logits
    ↓
CrossEntropyLoss
```

也就是说：

> 前三个项目重点在“如何训练神经网络”，第四个项目开始进入“如何让神经网络理解自然语言”。

---

<a id="sec-3"></a>

# 3. 第四个项目完整结构

项目结构：

```text
04_bert_text_classification/
│
├── main.py
├── jiudian.txt
│
├── model_save/
│   └── 训练过程中保存的模型
│
└── model_utils/
    ├── data.py
    ├── model.py
    └── train.py
```

各文件职责：

| 文件 | 作用 |
|---|---|
| `main.py` | 项目总入口、设置参数、创建模型、创建优化器、开始训练 |
| `jiudian.txt` | 酒店评论数据集 |
| `data.py` | 读取文本、构建 Dataset、DataLoader |
| `model.py` | 定义 BERT 分类模型 |
| `train.py` | 训练、验证、保存模型、画曲线 |
| `model_save/` | 保存训练得到的模型 |

可以理解为：

```text
data.py
→ 数据

model.py
→ 模型

train.py
→ 怎么训练

main.py
→ 把它们全部组织起来
```

这与第三个项目的多文件版本思路是一致的。

---

<a id="sec-4"></a>

# 4. 整个项目运行流程

运行：

```bash
python main.py
```

程序整体执行：

```text
main.py
│
├── 固定随机种子
│
├── 设置超参数
│
├── get_dataloader()
│      ↓
│   data.py
│      ├── 读取 jiudian.txt
│      ├── 抽样平衡类别
│      ├── train_test_split
│      ├── JdDataset
│      └── DataLoader
│
├── myBertModel()
│      ↓
│   model.py
│      ├── 加载 bert-base-chinese
│      ├── 加载 BertTokenizer
│      └── Linear(768, 2)
│
├── CrossEntropyLoss
├── AdamW
├── Scheduler
│
└── train_val()
       ↓
    train.py
       ├── Train
       ├── Validation
       ├── 保存最佳模型
       └── 绘制曲线
```

---

<a id="sec-5"></a>

# 5. `main.py`：项目总入口

`main.py` 导入：

```python
from model_utils.data import get_dataloader
from model_utils.model import myBertModel
from model_utils.train import train_val
```

这三句非常重要。

它们正好对应：

```text
get_dataloader
→ 数据

myBertModel
→ 模型

train_val
→ 训练
```

因此 `main.py` 不负责具体实现细节，而负责：

> **组织整个项目。**

---

<a id="sec-6"></a>

# 6. 固定随机种子

代码：

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
seed_everything(1)
```

作用和第三个项目一样：

> **尽量提高实验的可复现性。**

因为训练过程可能包含：

```text
Dataset 划分
DataLoader shuffle
参数更新
CUDA 运算
```

等随机因素。

---

<a id="sec-7"></a>

# 7. 项目主要超参数

代码：

```python
model_name = 'MyModel'
num_class = 2
batchSize = 4
learning_rate = 0.0001
loss = nn.CrossEntropyLoss()
epoch = 3
```

含义：

```text
num_class = 2
→ 二分类

batchSize = 4
→ 每次训练 4 条评论

learning_rate = 0.0001
→ 学习率 1e-4

epoch = 3
→ 整个训练集训练 3 遍
```

设备：

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

如果 GPU 可用：

```text
cuda
```

否则：

```text
cpu
```

---

<a id="sec-8"></a>

# 8. `jiudian.txt` 数据集

数据第一行：

```text
label,review
```

所以代码：

```python
if i == 0:
    continue
```

跳过表头。

项目当前数据共有两个类别：

```text
0
1
```

当前文件本身类别并不平衡。

代码注释记录：

```text
标签 1：5322 条
标签 0：2444 条
```

并且数据在文件中不是随机混合，而是大体按照标签连续排列。

因此项目没有直接使用所有数据，而是进行了人工抽样。

---

<a id="sec-9"></a>

# 9. 为什么要对数据进行抽样

代码：

```python
if i > 200 and i < 7500:
    continue
```

它的效果是：

```text
前 200 条
→ 保留

201 ~ 7499
→ 跳过

7500 之后
→ 保留
```

由于原始文件前部主要是标签 `1`，后部是标签 `0`，所以这段代码本质上是在：

```text
从数量多的类别中取少量样本
+
从数量少的类别尾部取一部分样本
```

使两个类别数量更加接近。

这是一种非常直接的：

> **欠采样（Under-sampling）**

思路。

如果类别极度不平衡：

```text
类别 1：5000
类别 0：200
```

即使模型永远预测：

```text
1
```

也可能得到很高的 Accuracy。

所以分类任务中必须注意：

> **类别平衡问题。**

---

<a id="sec-10"></a>

# 10. `data.py`：读取文本数据

核心函数：

```python
def read_txt_data(path):
```

创建：

```python
label = []
data = []
```

分别保存：

```text
label
→ 标签

data
→ 评论文本
```

读取文件：

```python
with open(path, "r", encoding="utf-8") as f:
```

使用：

```text
UTF-8
```

读取中文文本。

---

<a id="sec-11"></a>

# 11. `split(",", 1)` 为什么要写 1

代码：

```python
line = line.split(",", 1)
```

非常值得理解。

评论文本本身可能包含很多逗号，例如：

```text
1,位置不错,服务很好,早餐也可以
```

如果直接：

```python
split(",")
```

会得到：

```text
["1", "位置不错", "服务很好", "早餐也可以"]
```

但是项目真正想要的是：

```text
标签：
1

文本：
位置不错,服务很好,早餐也可以
```

所以：

```python
split(",", 1)
```

中的：

```text
1
```

表示：

> **最多只分割一次。**

因此：

```text
"1,位置不错,服务很好"
```

得到：

```python
[
    "1",
    "位置不错,服务很好"
]
```

然后：

```python
label.append(line[0])
data.append(line[1])
```

---

<a id="sec-12"></a>

# 12. 自定义 `JdDataset`

定义：

```python
class JdDataset(Dataset):
```

和前面项目一样，需要实现：

```text
__init__
__getitem__
__len__
```

初始化：

```python
self.X = x
```

此时：

```text
self.X
```

不是 Tensor。

而是：

> **Python 中文字符串列表。**

标签：

```python
label = [int(i) for i in label]
self.Y = torch.LongTensor(label)
```

把：

```text
"0"
"1"
```

从字符串变成整数：

```text
0
1
```

然后转成：

```python
torch.LongTensor
```

这是为了后面的：

```python
CrossEntropyLoss
```

---

<a id="sec-13"></a>

# 13. 文本 Dataset 与图像 Dataset 的区别

第三个项目中：

```text
Dataset
↓
直接返回图片 Tensor
```

例如：

```text
[3, 224, 224]
```

但本项目：

```python
return self.X[item], self.Y[item]
```

返回：

```text
中文字符串
+
LongTensor 标签
```

例如：

```text
"酒店位置不错，服务很好"
+
tensor(1)
```

为什么不在 Dataset 里直接转 Tensor？

因为文本不能像图片一样：

```python
ToTensor()
```

文本必须先经过：

> **Tokenizer**

Tokenizer 的工作放在：

```text
model.py
```

中的：

```python
build_bert_input()
```

完成。

---

<a id="sec-14"></a>

# 14. `train_test_split` 划分训练集和验证集

代码：

```python
train_x, val_x, train_y, val_y = train_test_split(
    x,
    label,
    test_size=valSize,
    shuffle=True,
    stratify=label
)
```

默认：

```python
valSize = 0.2
```

也就是：

```text
80%
→ 训练集

20%
→ 验证集
```

所以：

```text
所有抽样后的数据
      ↓
train_test_split
      ↓
80% Train
20% Validation
```

---

<a id="sec-15"></a>

# 15. `stratify=label` 是什么

这一项很重要：

```python
stratify=label
```

作用：

> **让训练集和验证集尽量保持与原数据相同的类别比例。**

例如抽样后的数据：

```text
类别 1：200
类别 0：267
```

如果完全随机切分，有可能验证集中：

```text
类别 1 太少
类别 0 太多
```

加上：

```python
stratify=label
```

会尽量保持：

```text
Train
和
Validation
```

中的类别比例一致。

---

<a id="sec-16"></a>

# 16. BERT 项目中的 DataLoader

代码：

```python
train_loader = DataLoader(
    train_set,
    batch_size=batchsize,
    shuffle=True
)
```

如果：

```text
batch_size = 4
```

那么一个 Batch 大致是：

```text
4 条中文评论
+
4 个标签
```

例如：

```text
text:
[
 "酒店很好",
 "房间太小",
 "服务不错",
 "早餐很差"
]

labels:
tensor([1, 0, 1, 0])
```

这里和图像项目最大的区别是：

```text
图片项目：
batch_x 已经是 Tensor

文本项目：
batch 中的 text 仍然是字符串
```

真正的 Tensor 化在 BERT Tokenizer 中完成。

---

<a id="sec-17"></a>

# 17. `model.py`：BERT 分类模型

定义：

```python
class myBertModel(nn.Module):
```

核心结构其实很简单：

```text
文本
↓
Tokenizer
↓
BERT
↓
768 维文本特征
↓
Linear
↓
2 类
```

初始化：

```python
self.bert = BertModel.from_pretrained(bert_path)
```

Tokenizer：

```python
self.tokenizer = BertTokenizer.from_pretrained(bert_path)
```

分类器：

```python
self.out = nn.Sequential(
    nn.Linear(768, num_class)
)
```

---

<a id="sec-18"></a>

# 18. BERT 是什么

BERT：

> **Bidirectional Encoder Representations from Transformers**

本项目不需要一开始记住全部细节。

先理解成：

> BERT 是一个已经经过大规模文本预训练的语言模型，可以把一句中文文本转换成具有上下文信息的向量表示。

例如：

```text
酒店服务很好
```

经过 BERT 后，不再是简单字符串，而会变成：

```text
一组高维语义特征
```

然后这些特征可以用于：

```text
文本分类
情感分析
命名实体识别
问答
```

本项目使用的是：

```text
文本分类
```

---

<a id="sec-19"></a>

# 19. BERT 与 Transformer 的关系

BERT 是基于 Transformer 的。

可以先记：

```text
Transformer
├── Encoder
└── Decoder

BERT
└── 主要使用 Transformer Encoder
```

因此 BERT 的核心能力之一来自：

> **Self-Attention**

它可以让一个词在计算表示时参考句子中的其他词。

例如：

```text
这个酒店房间不大，但是服务很好
```

模型在理解：

```text
好
```

时，可以结合：

```text
服务
```

以及整个上下文。

---

<a id="sec-20"></a>

# 20. `BertModel.from_pretrained()`

代码：

```python
self.bert = BertModel.from_pretrained(bert_path)
```

其中：

```python
bert_path = 'bert-base-chinese'
```

这里不是：

```text
随机创建一个 BERT
```

而是：

> **加载已经训练好的 `bert-base-chinese` 参数。**

这和第三个项目：

```text
预训练 ResNet
```

的思想非常相似。

第三个项目：

```text
ImageNet
↓
预训练 CNN
↓
Food-11
```

第四个项目：

```text
大规模中文文本
↓
预训练 BERT
↓
酒店评论分类
```

---

<a id="sec-21"></a>

# 21. 什么是预训练模型

可以把预训练理解成：

```text
第一阶段：
先在非常大的通用数据集上学习

第二阶段：
再拿到具体任务上继续训练
```

本项目：

```text
bert-base-chinese
已经提前学过大量中文文本
        ↓
酒店评论二分类
        ↓
Fine-tuning
```

所以我们不是从零开始让模型学习：

```text
什么是中文
什么是词语
什么是上下文
```

而是在已有语言能力基础上学习：

```text
酒店评论的类别判断
```

---

<a id="sec-22"></a>

# 22. `BertTokenizer`

代码：

```python
self.tokenizer = BertTokenizer.from_pretrained(bert_path)
```

Tokenizer 可以理解成：

> **BERT 的文本预处理器。**

因为神经网络不能直接计算：

```text
"酒店服务很好"
```

必须先把文字转换成数字。

所以：

```text
字符串
↓
Tokenizer
↓
Token
↓
Token ID
↓
Tensor
```

---

<a id="sec-23"></a>

# 23. Tokenizer 做了什么

代码：

```python
Input = self.tokenizer(
    text,
    return_tensors='pt',
    padding='max_length',
    truncation=True,
    max_length=128
)
```

这里 Tokenizer 一次完成多个任务：

```text
文本切分
↓
转换 Token ID
↓
加入特殊 Token
↓
补齐长度
↓
截断过长文本
↓
生成 Attention Mask
↓
生成 Token Type IDs
↓
转换 PyTorch Tensor
```

最终得到一个字典：

```python
Input
```

其中最重要的三个字段：

```text
input_ids
attention_mask
token_type_ids
```

---

<a id="sec-24"></a>

# 24. `input_ids`

代码：

```python
input_ids = Input["input_ids"]
```

它表示：

> **每个 Token 在 BERT 词表中的编号。**

例如，仅作为理解：

```text
酒店
服务
很好
```

最终会被转换成类似：

```text
[101, 6983, 2421, ..., 102]
```

这里真正具体的 ID 由：

```text
bert-base-chinese 的词表
```

决定。

所以：

```text
文字
↓
Token ID
```

以后模型计算的其实是：

```text
数字 ID
```

而不是原始字符本身。

---

<a id="sec-25"></a>

# 25. `[CLS]` 和 `[SEP]`

BERT Tokenizer 会加入特殊 Token。

分类任务中最重要的两个：

```text
[CLS]
[SEP]
```

可以把一句话理解成：

```text
[CLS] 酒店 服务 很好 [SEP]
```

其中：

```text
[CLS]
```

放在整句话最前面。

BERT 会得到 `[CLS]` 对应的表示。

这个表示会用于本项目的句子分类。

---

<a id="sec-26"></a>

# 26. `attention_mask`

代码：

```python
attention_mask = Input["attention_mask"]
```

为什么需要它？

因为：

```python
padding='max_length'
```

会把所有文本补到统一长度：

```text
128
```

例如一句话只有 8 个有效 Token：

```text
真实 Token：
8 个

最大长度：
128
```

剩余位置需要 Padding。

那么 Attention Mask 可以帮助模型区分：

```text
真实 Token
和
Padding
```

一般可以粗略理解：

```text
1
→ 有效 Token

0
→ Padding
```

例如：

```text
input_ids:
[101, ..., 102, 0, 0, 0]

attention_mask:
[1,   ..., 1,   0, 0, 0]
```

---

<a id="sec-27"></a>

# 27. `token_type_ids`

代码：

```python
token_type_ids = Input["token_type_ids"]
```

它用于区分不同文本片段。

BERT 原始设计可以同时输入：

```text
句子 A
+
句子 B
```

例如：

```text
[CLS]
句子 A
[SEP]
句子 B
[SEP]
```

这时：

```text
token_type_ids
```

可以告诉模型：

```text
哪些 Token 属于句子 A
哪些 Token 属于句子 B
```

本项目每个样本只有一条酒店评论，因此通常都是同一段文本。

---

<a id="sec-28"></a>

# 28. `padding='max_length'`

代码：

```python
padding='max_length'
```

表示：

> 所有文本统一补到 `max_length`。

本项目：

```python
max_length=128
```

因此：

```text
短文本
↓
Padding
↓
128

长文本
↓
Truncation
↓
128
```

这样才能组成一个规则的 Batch Tensor。

---

<a id="sec-29"></a>

# 29. `truncation=True`

酒店评论可能非常长。

如果超过：

```text
128 Token
```

代码：

```python
truncation=True
```

会截断超出的部分。

否则一个 Batch 中不同句子长度不同，会给模型输入带来问题。

---

<a id="sec-30"></a>

# 30. `max_length=128`

项目使用：

```python
max_length=128
```

因此每一条评论经过 Tokenizer 后都统一为：

```text
长度 128
```

例如 Batch Size：

```text
4
```

那么：

```text
input_ids.shape
=
[4, 128]
```

其他两个：

```text
attention_mask.shape
=
[4, 128]

token_type_ids.shape
=
[4, 128]
```

---

<a id="sec-31"></a>

# 31. BERT 输入 Tensor 的形状

假设：

```text
batch_size = 4
max_length = 128
```

则：

```text
input_ids
[4, 128]

attention_mask
[4, 128]

token_type_ids
[4, 128]
```

可以理解：

```text
4
→ 4 条评论

128
→ 每条评论最多 128 个 Token 位置
```

这与 CNN：

```text
[B, C, H, W]
```

不同。

BERT 常见输入：

```text
[B, L]
```

其中：

```text
B = Batch Size
L = Sequence Length
```

---

<a id="sec-32"></a>

# 32. `build_bert_input()` 完整流程

代码：

```python
def build_bert_input(self, text):
```

处理：

```python
Input = self.tokenizer(...)
```

然后：

```python
input_ids = Input["input_ids"].to(self.device)
attention_mask = Input["attention_mask"].to(self.device)
token_type_ids = Input["token_type_ids"].to(self.device)
```

最终：

```python
return input_ids, attention_mask, token_type_ids
```

所以：

```text
DataLoader 返回字符串
      ↓
build_bert_input()
      ↓
Tokenizer
      ↓
PyTorch Tensor
      ↓
移动到 GPU
      ↓
BERT
```

---

<a id="sec-33"></a>

# 33. BERT 前向传播

代码：

```python
sequence_out, pooled_output = self.bert(
    input_ids=input_ids,
    attention_mask=attention_mask,
    token_type_ids=token_type_ids,
    return_dict=False
)
```

BERT 输入三个主要 Tensor：

```text
input_ids
attention_mask
token_type_ids
```

输出：

```text
sequence_out
pooled_output
```

---

<a id="sec-34"></a>

# 34. `sequence_out` 是什么

`sequence_out` 可以理解为：

> **每一个 Token 最终得到的上下文特征。**

假设：

```text
batch_size = 4
sequence_length = 128
hidden_size = 768
```

那么大致：

```text
sequence_out.shape
=
[4, 128, 768]
```

也就是说：

```text
4 条评论
×
128 个 Token
×
每个 Token 768 维特征
```

例如一句话：

```text
酒店 服务 很 好
```

不是只得到一个向量。

而是：

```text
酒店 → 768 维
服务 → 768 维
很   → 768 维
好   → 768 维
...
```

---

<a id="sec-35"></a>

# 35. `pooled_output` 是什么

本项目真正使用：

```python
pooled_output
```

它可以理解成：

> **BERT 为整条输入文本生成的一个固定长度表示。**

对于：

```text
bert-base-chinese
```

隐藏维度：

```text
768
```

假设：

```text
batch_size = 4
```

则：

```text
pooled_output.shape
≈
[4, 768]
```

即：

```text
每条评论
↓
一个 768 维向量
```

---

<a id="sec-36"></a>

# 36. 为什么分类使用 `pooled_output`

本项目要判断的是：

> **整条评论属于类别 0 还是类别 1。**

而不是给每一个字单独分类。

所以最终需要的是：

```text
整句话的表示
```

因此：

```python
out = self.out(pooled_output)
```

把：

```text
768 维文本特征
```

送到：

```text
分类层
```

---

<a id="sec-37"></a>

# 37. `Linear(768, 2)`

代码：

```python
nn.Linear(768, num_class)
```

本项目：

```python
num_class = 2
```

所以：

```text
768
↓
2
```

假设 Batch Size = 4：

```text
pooled_output:
[4, 768]

↓ Linear

out:
[4, 2]
```

每一条评论输出两个数。

例如：

```text
评论 1：
[-1.2, 2.6]

评论 2：
[3.1, -0.8]
```

这两个值是：

> **Logits**

不是概率。

---

<a id="sec-38"></a>

# 38. BERT 分类模型完整数据流

把 `model.py` 串起来：

```text
4 条中文评论
│
↓
BertTokenizer
│
├── input_ids        [4,128]
├── attention_mask   [4,128]
└── token_type_ids   [4,128]
│
↓
BertModel
│
├── sequence_out     [4,128,768]
└── pooled_output    [4,768]
│
↓
Linear(768,2)
│
↓
Logits
[4,2]
```

这就是整个模型最核心的数据流。

---

<a id="sec-39"></a>

# 39. 为什么模型最后不写 Softmax

模型：

```python
out = self.out(pooled_output)
return out
```

没有：

```python
Softmax
```

这是正确的。

因为训练中：

```python
loss = nn.CrossEntropyLoss()
```

然后：

```python
bat_loss = loss(pred, labels)
```

和第三个 Food-11 项目一样：

> `CrossEntropyLoss` 应该直接接收 Logits。

因此不要改成：

```python
pred = torch.softmax(model(text), dim=1)
loss(pred, labels)
```

---

<a id="sec-40"></a>

# 40. `CrossEntropyLoss`

假设：

```text
pred.shape = [4,2]
```

标签：

```text
labels.shape = [4]
```

例如：

```text
pred:
[
 [2.1, -0.5],
 [-0.7, 1.8],
 [1.3, 0.2],
 [-1.0, 2.4]
]

labels:
[0,1,0,1]
```

CrossEntropyLoss 会根据：

```text
正确类别对应的 Logit
```

计算损失。

目标：

```text
正确类别得分更高
错误类别得分更低
```

---

<a id="sec-41"></a>

# 41. `train.py`：训练函数

训练入口：

```python
def train_val(para):
```

这里不是把很多参数一个一个传进去，而是：

```python
para
```

一个字典。

例如：

```python
trainpara = {
    'model': model,
    'train_loader': train_loader,
    'optimizer': optimizer,
    ...
}
```

然后在函数中：

```python
model = para['model']
train_loader = para['train_loader']
...
```

这样可以把大量训练配置集中到一个字典中传入。

---

<a id="sec-42"></a>

# 42. 一个 Batch 的训练过程

核心代码：

```python
for batch in tqdm(train_loader):

    model.zero_grad()

    text, labels = batch[0], batch[1].to(device)

    pred = model(text)

    bat_loss = loss(pred, labels)

    bat_loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        1.0
    )

    optimizer.step()

    scheduler.step()
```

完整流程：

```text
一个 Batch 的文本和标签
        ↓
清空梯度
        ↓
BERT forward
        ↓
得到 [B,2] Logits
        ↓
CrossEntropyLoss
        ↓
backward
        ↓
梯度裁剪
        ↓
AdamW 更新参数
        ↓
Scheduler 调整学习率
```

---

<a id="sec-43"></a>

# 43. `model.zero_grad()`

代码：

```python
model.zero_grad()
```

作用：

> **清空上一轮留下来的梯度。**

因为 PyTorch 默认会累加梯度。

如果不清：

```text
Batch 1 梯度
+
Batch 2 梯度
+
Batch 3 梯度
```

会不断累积。

这和前面的：

```python
optimizer.zero_grad()
```

目的相同。

当前项目保留原结构，使用：

```python
model.zero_grad()
```

---

<a id="sec-44"></a>

# 44. `loss.backward()`

代码：

```python
bat_loss.backward()
```

PyTorch 会从 Loss 开始：

```text
Loss
↓
Linear
↓
BERT
↓
所有可训练参数
```

自动计算梯度。

即：

$$
\frac{\partial L}
{\partial \theta}
$$

这里的参数不仅包括最后：

```text
Linear(768,2)
```

还包括：

```text
BERT 中大量参数
```

因为本项目没有冻结 BERT。

所以这是：

> **对预训练 BERT 整体进行 Fine-tuning。**

---

<a id="sec-45"></a>

# 45. 梯度裁剪 `clip_grad_norm_`

代码：

```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    1.0
)
```

含义：

> 如果整体梯度过大，就把梯度范数限制在一定范围内。

这里：

```text
max_norm = 1.0
```

为什么使用梯度裁剪？

主要为了：

```text
避免梯度过大
↓
提高训练稳定性
```

顺序必须是：

```text
backward
↓
clip_grad_norm_
↓
optimizer.step
```

不能：

```text
optimizer.step
↓
再 clip
```

因为参数已经更新完了。

---

<a id="sec-46"></a>

# 46. `optimizer.step()`

代码：

```python
optimizer.step()
```

作用：

> 根据当前参数梯度更新模型参数。

也就是：

```text
Gradient
↓
Optimizer
↓
修改 BERT + 分类层参数
```

---

<a id="sec-47"></a>

# 47. AdamW 优化器

`main.py`：

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=0.0001
)
```

AdamW 是训练 Transformer / BERT 时常见的优化器之一。

第一遍学习可以把它理解为：

> 在 Adam 的基础上，更明确地处理 Weight Decay。

本项目：

```text
learning_rate = 1e-4
weight_decay = 1e-4
```

---

<a id="sec-48"></a>

# 48. Weight Decay

代码：

```python
weight_decay=0.0001
```

Weight Decay 可以粗略理解为：

> 对模型参数增加一定约束，不希望参数无限制变大。

作用之一：

```text
缓解过拟合
```

需要注意：

```text
Weight Decay
```

和：

```text
Learning Rate
```

不是一回事。

---

<a id="sec-49"></a>

# 49. 学习率调度器 Scheduler

除了优化器：

```python
optimizer
```

项目还有：

```python
scheduler
```

定义：

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(...)
```

Scheduler 不直接更新模型参数。

它负责：

> **训练过程中调整 Learning Rate。**

流程：

```text
optimizer
→ 更新参数

scheduler
→ 调整 optimizer 后续使用的学习率
```

---

<a id="sec-50"></a>

# 50. `CosineAnnealingWarmRestarts`

代码：

```python
torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=20,
    eta_min=1e-9
)
```

核心思想可以先理解为：

```text
学习率
从较高值
↓
按照余弦形状逐渐降低
↓
到一定周期后重新升高
↓
再次下降
```

大概类似：

```text
高
╲
 ╲
  ╲
   低
   ↑重新开始
高
╲
 ╲
...
```

其中：

```text
T_0 = 20
```

控制第一次重启的周期。

当前代码是在：

```python
每个 Batch
```

后调用：

```python
scheduler.step()
```

所以它是随着 Batch 更新次数推进，而不是只在每个 Epoch 结束时推进。

---

<a id="sec-51"></a>

# 51. 训练 Loss 的统计

代码：

```python
train_loss += bat_loss.item() * labels.size(0)
```

为什么乘：

```python
labels.size(0)
```

因为：

```python
CrossEntropyLoss()
```

默认返回的是：

> 当前 Batch 的平均 Loss。

假设：

```text
Batch A：
4 个样本
平均 Loss = 0.8

Batch B：
3 个样本
平均 Loss = 0.5
```

不能直接：

```text
(0.8 + 0.5) / 总样本数
```

更准确做法：

```text
0.8 × 4
+
0.5 × 3
```

最后：

```python
train_loss / len(train_loader.dataset)
```

得到样本平均 Loss。

---

<a id="sec-52"></a>

# 52. 训练 Accuracy 的统计

代码：

```python
np.argmax(
    pred.detach().cpu().numpy(),
    axis=1
)
```

如果：

```text
pred.shape = [4,2]
```

则：

```text
axis=1
```

表示：

> 对每个样本的两个类别 Logit 找最大值所在位置。

例如：

```text
pred:
[
 [2.0, 0.1],
 [0.3, 1.5]
]
```

得到：

```text
预测类别：
[0,1]
```

再：

```python
== labels.cpu().numpy()
```

比较预测和真实标签。

累计：

```python
train_acc
```

最终：

```python
train_acc / len(train_loader.dataset)
```

得到：

$$
Accuracy
=
\frac{预测正确样本数}
{总样本数}
$$

---

<a id="sec-53"></a>

# 53. 验证阶段

代码：

```python
if i % val_epoch == 0:
```

当前：

```python
val_epoch = 1
```

所以：

```text
每个 Epoch 都验证
```

验证：

```python
model.eval()

with torch.no_grad():
```

然后遍历：

```python
val_loader
```

计算：

```text
Val Loss
Val Accuracy
```

---

<a id="sec-54"></a>

# 54. `model.eval()` 与 `torch.no_grad()`

和图像项目一样，两者不要混淆。

```python
model.eval()
```

表示：

> 模型切换到验证 / 推理模式。

```python
torch.no_grad()
```

表示：

> 不构建反向传播需要的梯度计算图。

验证阶段：

```text
只 forward
不 backward
不 optimizer.step
```

所以两者经常一起使用。

---

<a id="sec-55"></a>

# 55. 保存最佳模型

代码先算：

```python
current_val_acc = (
    val_acc / val_loader.dataset.__len__()
)
```

然后：

```python
if current_val_acc > max_acc:
    torch.save(
        model,
        save_path+'best_model.pth'
    )

    max_acc = current_val_acc
```

也就是：

```text
当前验证 Accuracy
       ↓
超过历史最好
       ↓
保存模型
```

最终：

```text
model_save/best_model.pth
```

代表：

> 训练过程中验证准确率达到新高时保存的模型。

---

<a id="sec-56"></a>

# 56. 为什么设置 `max_acc=0.85`

`main.py`：

```python
'max_acc': 0.85
```

意思是：

```text
初始最佳准确率门槛
=
85%
```

因此只有：

```text
current_val_acc > 0.85
```

才会保存：

```text
best_model.pth
```

例如：

```text
Epoch 1:
Accuracy = 0.80
→ 不保存

Epoch 2:
Accuracy = 0.88
→ 保存

Epoch 3:
Accuracy = 0.86
→ 不覆盖
```

---

<a id="sec-57"></a>

# 57. 训练曲线

训练结束后：

```python
plt.plot(plt_train_loss)
plt.plot(plt_val_loss)
```

画：

```text
Train Loss
Val Loss
```

然后：

```python
plt.plot(plt_train_acc)
plt.plot(plt_val_acc)
```

画：

```text
Train Accuracy
Val Accuracy
```

并：

```python
plt.savefig('acc.png')
```

保存准确率曲线。

通过曲线可以观察：

```text
是否收敛
是否过拟合
训练是否稳定
```

---

<a id="sec-58"></a>

# 58. `model_Datapara()` 与多 GPU

`model.py` 还有：

```python
def model_Datapara(model, device, pre_path=None):
```

其中：

```python
torch.nn.DataParallel(model)
```

表示：

> 使用 PyTorch DataParallel 尝试让多个 GPU 并行处理 Batch。

当前 `main.py` 并没有调用：

```python
model_Datapara()
```

所以：

> **当前实际训练流程没有使用这一函数。**

因此第一次学习本项目时不用把它当主线。

---

<a id="sec-59"></a>

# 59. 当前项目中没有实际使用的配置

`trainpara` 中还有：

```python
'learning_rate'
'warmup_ratio'
'weight_decay'
'use_lookahead'
```

但是当前 `train_val()` 真正读取的主要是：

```text
model
train_loader
val_loader
scheduler
optimizer
loss
epoch
device
save_path
max_acc
val_epoch
```

也就是说：

```text
warmup_ratio
use_lookahead
```

当前并没有在训练函数中真正使用。

这一点阅读代码时要注意：

> 字典里出现某个参数，不代表训练逻辑一定实际使用了它。

---

<a id="sec-60"></a>

# 60. 当前代码几个需要特别理解的地方

这一节专门总结当前项目容易误解的位置。

## 60.1 `model_name = 'MyModel'`

当前：

```python
model_name = 'MyModel'
```

实际上后续没有利用：

```text
model_name
```

选择不同模型。

真正创建的是：

```python
model = myBertModel(...)
```

所以它目前更像保留的配置变量。

---

## 60.2 `param_optimizer`

代码：

```python
param_optimizer = list(model.parameters())
```

当前后面没有实际使用：

```text
param_optimizer
```

真正给 AdamW 的仍然是：

```python
model.parameters()
```

---

## 60.3 BERT 没有被冻结

项目没有：

```python
param.requires_grad = False
```

因此：

```text
BERT
+
最后 Linear
```

都会被反向传播更新。

这属于：

> **全参数 Fine-tuning**

---

## 60.4 `sequence_out` 得到了但没有使用

代码：

```python
sequence_out, pooled_output = self.bert(...)
```

但是后面只：

```python
self.out(pooled_output)
```

所以：

```text
sequence_out
```

虽然 BERT 返回了，但当前分类任务没有继续使用。

---

## 60.5 当前项目不是自己从零实现 Transformer

代码：

```python
BertModel.from_pretrained(...)
```

说明：

> 项目重点是“使用 BERT 完成文本分类”，而不是手写 Transformer / Self-Attention。

因此学习时可以分层：

```text
第一层：
先看懂 BERT 怎么用

第二层：
再深入 Transformer 结构
```

---

<a id="sec-61"></a>

# 61. 第四个项目完整调用关系

可以把四个 Python 文件理解成：

```text
main.py
│
├── from model_utils.data import get_dataloader
│        │
│        └── data.py
│             ├── read_txt_data()
│             ├── JdDataset
│             └── get_dataloader()
│
├── from model_utils.model import myBertModel
│        │
│        └── model.py
│             ├── BertTokenizer
│             ├── BertModel
│             ├── myBertModel
│             └── model_Datapara()
│
└── from model_utils.train import train_val
         │
         └── train.py
              ├── train
              ├── validation
              ├── save model
              └── plot
```

其中：

```text
main.py
```

就像总指挥。

---

<a id="sec-62"></a>

# 62. 第四个项目完整数据流

从最原始的文件开始：

```text
jiudian.txt
│
├── label
└── review
     ↓
read_txt_data()
     ↓
Python 字符串
+
LongTensor 标签
     ↓
JdDataset
     ↓
DataLoader
     ↓
一个 Batch
│
├── 4 条中文字符串
└── 4 个标签
     ↓
myBertModel
     ↓
BertTokenizer
│
├── input_ids
├── attention_mask
└── token_type_ids
     ↓
BertModel
     ↓
pooled_output
[B, 768]
     ↓
Linear(768, 2)
     ↓
Logits
[B, 2]
     ↓
CrossEntropyLoss
     ↓
backward()
     ↓
gradient clipping
     ↓
AdamW
     ↓
参数更新
```

这条流程是整个第四个项目最需要记住的内容。

---

<a id="sec-63"></a>

# 63. 第四个项目与第三个项目对比

第三个项目：

```text
图片
↓
Transform
↓
Tensor
↓
CNN / ResNet
↓
图像特征
↓
Linear
↓
类别
```

第四个项目：

```text
文本
↓
Tokenizer
↓
Token IDs
↓
BERT
↓
文本特征
↓
Linear
↓
类别
```

二者的对应关系非常清楚：

| 图像分类 | 文本分类 |
|---|---|
| 图片 | 中文文本 |
| Transform | Tokenizer |
| `[B,C,H,W]` | `[B,L]` |
| CNN / ResNet | BERT |
| 图像特征 | 文本特征 |
| Linear | Linear |
| CrossEntropyLoss | CrossEntropyLoss |
| Accuracy | Accuracy |

所以第四个项目并不是重新学习一套完全不同的 PyTorch。

核心训练框架仍然是：

```text
Dataset
↓
DataLoader
↓
Model
↓
Loss
↓
Backward
↓
Optimizer
↓
Validation
```

只是：

> **数据表示方式和模型结构变了。**

---

<a id="sec-64"></a>

# 64. 本项目必须掌握的问题

学完这个项目以后，至少应该可以回答下面这些问题。

## 数据部分

1. `jiudian.txt` 每一行是什么格式？

2. 为什么：

```python
split(",", 1)
```

不能简单写成：

```python
split(",")
```

3. 为什么标签要转：

```python
LongTensor
```

4. 为什么项目要对样本做抽样？

5. 什么是类别不平衡？

6. `stratify=label` 有什么作用？

7. 为什么训练 DataLoader：

```python
shuffle=True
```

而验证：

```python
shuffle=False
```

---

## Tokenizer 部分

8. 为什么中文字符串不能直接输入神经网络？

9. Tokenizer 的作用是什么？

10. 什么是 Token？

11. 什么是 Token ID？

12. `input_ids` 是什么？

13. `attention_mask` 是什么？

14. `token_type_ids` 是什么？

15. 为什么需要 Padding？

16. 为什么需要 Truncation？

17. `max_length=128` 是什么意思？

18. 如果：

```text
batch_size=4
max_length=128
```

那么：

```text
input_ids.shape
```

大概是多少？

答案：

```text
[4,128]
```

---

## BERT 部分

19. BERT 是基于什么结构？

20. BERT 与 Transformer 的关系是什么？

21. 为什么说 BERT 主要是 Encoder-only？

22. `BertModel.from_pretrained()` 是什么意思？

23. 什么叫预训练？

24. 什么叫 Fine-tuning？

25. 本项目有没有冻结 BERT？

26. `sequence_out` 是什么？

27. `pooled_output` 是什么？

28. 为什么分类使用 `pooled_output`？

29. 为什么：

```text
pooled_output
```

是 768 维？

30. `Linear(768,2)` 表示什么？

---

## 分类部分

31. 模型最终为什么输出 2 个数？

32. 这两个数是不是概率？

33. 什么是 Logits？

34. 为什么训练前不需要手动 Softmax？

35. `CrossEntropyLoss` 接收什么？

36. `argmax(axis=1)` 在这里做什么？

---

## 训练部分

37. `model.zero_grad()` 是什么？

38. `loss.backward()` 做什么？

39. 为什么需要梯度裁剪？

40. 梯度裁剪为什么一定要放：

```text
backward
之后
optimizer.step
之前
```

41. `optimizer.step()` 做什么？

42. AdamW 是什么？

43. Weight Decay 是什么？

44. Scheduler 是什么？

45. Optimizer 和 Scheduler 有什么区别？

46. `model.train()` / `model.eval()` 有什么区别？

47. `torch.no_grad()` 有什么作用？

48. Train Loss 怎么统计？

49. Accuracy 怎么计算？

50. 为什么 `max_acc=0.85` 表示 85%？

---

<a id="sec-65"></a>

# 65. 推荐学习顺序

建议不要直接从：

```text
Transformer 数学推导
```

开始。

对于当前项目，按照下面顺序更容易建立整体理解。

## 第一步：先看数据

先掌握：

```text
jiudian.txt
↓
read_txt_data
↓
JdDataset
↓
DataLoader
```

确保能回答：

```text
一个 Batch 到底长什么样？
```

---

## 第二步：学习 Tokenizer

重点：

```text
中文字符串
↓
Tokenizer
↓
input_ids
attention_mask
token_type_ids
```

这一部分是：

> **从普通 Python 文本进入 BERT 世界的入口。**

---

## 第三步：只理解 BERT 输入输出

先不要急着研究 Self-Attention 公式。

只先理解：

```text
输入：
[B,128]

↓ BERT

sequence_out：
[B,128,768]

pooled_output：
[B,768]
```

然后：

```text
[B,768]
↓
Linear
↓
[B,2]
```

---

## 第四步：理解训练

重点：

```text
pred
↓
CrossEntropyLoss
↓
backward
↓
clip_grad_norm
↓
AdamW
↓
scheduler
```

---

## 第五步：再学习 Transformer

当整个项目已经能讲通以后，再系统学习：

```text
Embedding
Position Encoding
Self-Attention
Q / K / V
Multi-Head Attention
Feed Forward
Residual Connection
LayerNorm
Transformer Encoder
```

这样再回来看 BERT，会容易很多。

---

<a id="sec-66"></a>

# 66. 项目总结

第四个项目可以浓缩为一句话：

> **使用预训练中文 BERT 将酒店评论编码成 768 维文本表示，再通过线性分类层完成二分类，并使用 CrossEntropyLoss、AdamW、梯度裁剪和学习率调度器完成 Fine-tuning。**

完整知识关系：

```text
酒店评论
│
↓
Tokenizer
│
├── input_ids
├── attention_mask
└── token_type_ids
│
↓
BERT
│
├── sequence_out
└── pooled_output
│
↓
Linear(768,2)
│
↓
Logits
│
↓
CrossEntropyLoss
│
↓
Backward
│
↓
Gradient Clipping
│
↓
AdamW
│
↓
Scheduler
│
↓
Validation
│
↓
Best Model
```

从四个项目整体来看，可以形成这样一条 PyTorch 学习路线：

```text
项目一：线性回归
↓
理解最基础训练循环

项目二：完整回归项目
↓
理解 Dataset / DataLoader / Validation

项目三：图像分类
↓
理解 CNN / 图像分类 / 迁移学习

项目四：BERT 文本分类
↓
理解 NLP / Tokenizer / Transformer 预训练模型
```

真正重要的变化不是 PyTorch 的训练框架变了，而是：

```text
数据类型
+
特征提取模型
```

在变化。

共同框架一直是：

```text
Data
↓
Dataset
↓
DataLoader
↓
Model
↓
Loss
↓
Backward
↓
Optimizer
↓
Validation
```

只要这条主线建立起来，后面无论学习：

```text
CNN
ResNet
BERT
Transformer
VAE
GAN
Diffusion
```

都会更容易理解它们在整个深度学习训练系统中的位置。
