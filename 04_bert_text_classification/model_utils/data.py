from torch.utils.data import Dataset,DataLoader,ConcatDataset
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np



def read_txt_data(path):
    label = []
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in tqdm(enumerate(f)):
            if i == 0:
                continue  # continue 表示立即执行下一次循环

            # jiudian.txt 原始样本不平衡：
            # 前 5322 条为标签 1，后 2444 条为标签 0。
            # 这里保留原项目的抽样方式：
            # 保留前 200 条正类，以及第 7500 行之后的负类，使训练样本更接近平衡。
            if i > 200 and i < 7500:
                continue

            line = line.strip('\n')
            line = line.split(",", 1)  # 1表示分割次数，只按第一个逗号分开标签和评论
            label.append(line[0])
            data.append(line[1])
    print(len(label))
    print(len(data))
    return data, label


class JdDataset(Dataset):
    def __init__(self, x, label):
        self.X = x
        label = [int(i) for i in label]
        self.Y = torch.LongTensor(label)

    def __getitem__(self, item):
        return self.X[item], self.Y[item]            # 文本可以保持为 str，DataLoader 会按 batch 收集文本

    def __len__(self):
        return len(self.Y)


def get_dataloader(path, batchsize=1, valSize=0.2):
    x, label = read_txt_data(path)
    train_x, val_x, train_y, val_y = train_test_split(
        x, label,
        test_size=valSize,
        shuffle=True,
        stratify=label
    )

    train_set = JdDataset(train_x, train_y)
    val_set = JdDataset(val_x, val_y)

    # 修正：训练集每个 epoch 打乱样本顺序；验证集不需要打乱
    train_loader = DataLoader(train_set, batch_size=batchsize, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batchsize, shuffle=False)
    return train_loader, val_loader




if __name__ == '__main__':

    get_dataloader("../jiudian.txt",batchsize=4)
