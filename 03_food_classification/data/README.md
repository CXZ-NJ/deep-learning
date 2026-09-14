# Food-11 数据放置说明

数据集图片不会提交到 GitHub。完整数据量较大，请在本地下载并解压。

课程数据地址：

- <https://www.kaggle.com/datasets/zhaopang/ml2021springhw3>

Food-11 的 11 个类别编号：

| 编号 | 英文类别 | 中文理解 |
|---|---|---|
| `00` | Bread | 面包 |
| `01` | Dairy product | 乳制品 |
| `02` | Dessert | 甜点 |
| `03` | Egg | 鸡蛋 |
| `04` | Fried food | 油炸食品 |
| `05` | Meat | 肉类 |
| `06` | Noodles/Pasta | 面条/意大利面 |
| `07` | Rice | 米饭 |
| `08` | Seafood | 海鲜 |
| `09` | Soup | 汤 |
| `10` | Vegetable/Fruit | 蔬菜/水果 |

下载后，把数据整理为以下结构：

```text
03_food_classification/
└── data/
    └── food-11/
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

仓库代码使用的是 `00–10` 共 11 类。原来的 `food-11_sample` 中还存在一个 `11` 目录，但 Food-11 模型没有第 12 个输出，因此修复后的代码会明确提示并忽略该目录。

完成放置后，可以先运行单文件版本：

```bash
python simple_class.py --epochs 1
```

也可以直接指定电脑上现有的数据位置，不必复制完整数据集：

```bash
python simple_class.py --data-root "D:/你的目录/food-11" --epochs 1
```

模块化版本：

```bash
python main.py --data-root "D:/你的目录/food-11" --epochs 10
```

数据目录已经写入仓库根目录的 `.gitignore`，不会被普通的 `git add` 加入 GitHub。
