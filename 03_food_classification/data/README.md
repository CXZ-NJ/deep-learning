# 数据放置说明

图片数据量较大，不上传到 GitHub，只在自己的电脑中保存。

数据集地址：<https://www.kaggle.com/datasets/zhaopang/ml2021springhw3>

## 运行 `simple_class.py`

将小型示例数据放到：

```text
03_food_classification/data/food-11_sample/
```

目录结构：

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
└── validation/
    ├── 00/
    ├── 01/
    ├── ...
    └── 10/
```

注意：你当前的 sample 数据中还有一个 `11` 文件夹，但代码设置的是 11 分类，只循环 `00–10`，所以 `11` 不会参与训练。建议以后制作 sample 时只保留 `00–10`。

## 运行 `main.py`

`main.py` 默认同样读取：

```text
03_food_classification/data/food-11_sample/
```

如果要改成完整数据，把完整数据放到：

```text
03_food_classification/data/food-11/
```

然后在 `main.py` 中注释 sample 路径、启用下一行完整数据路径即可。
