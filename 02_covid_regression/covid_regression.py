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


print("Python:", sys.version)
print("PyTorch:", torch.__version__)

# ==================== 1. 数据集 ====================

class CovidDataset(Dataset):
    def __init__(self, file_path, mode, mean=None, std=None):
        with open(file_path, "r") as f:
            ori_data = list(csv.reader(f))

        # 去掉第一行表头、第一列 id，并转换成 float
        csv_data = np.array(ori_data)[1:, 1:].astype(float)

        # 逢五取一作为验证集，其余作为训练集
        # 这种方法适合学习练习，真实项目通常会使用更规范的划分方法。
        if mode == "train":
            indices = [i for i in range(len(csv_data)) if i % 5 != 0]

        elif mode == "val":
            indices = [i for i in range(len(csv_data)) if i % 5 == 0]

        elif mode == "test":
            indices = list(range(len(csv_data)))

        else:
            raise ValueError("mode 必须是 'train'、'val' 或 'test'")

        # 前 93 列作为输入特征
        X = torch.tensor(csv_data[indices, :93])

        # 训练集/验证集有真实标签，测试集没有
        if mode != "test":
            self.Y = torch.tensor(csv_data[indices, -1])

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

        # 防止某一列标准差为 0，导致除以 0
        self.std = torch.where(self.std == 0, torch.ones_like(self.std), self.std)

        self.X = (X - self.mean) / self.std
        self.mode = mode

    def __getitem__(self, item):
        if self.mode == "test":
            return self.X[item].float()

        return self.X[item].float(), self.Y[item].float()

    def __len__(self):
        return len(self.X)


# ==================== 2. 神经网络模型 ====================

class MyModel(nn.Module):
    def __init__(self, in_dim):
        super().__init__()

        self.fc1 = nn.Linear(in_dim, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)

        # [batch_size, 1] -> [batch_size]
        if len(x.size()) > 1:
            x = x.squeeze(1)

        return x


# ==================== 3. 损失函数 ====================

def mse_loss(pred, target, model):
    # 均方误差
    loss_fn = nn.MSELoss(reduction="mean")

    # L2 正则项
    regularization_loss = 0.0
    for param in model.parameters():
        regularization_loss += torch.sum(param ** 2)

    # 总损失 = MSE + L2 正则
    return loss_fn(pred, target) + 0.00075 * regularization_loss


# ==================== 4. 训练 + 验证 ====================

def train_val(model, train_loader, val_loader, optimizer, loss_fn,
              device, epochs, save_path):
    model = model.to(device)

    # 确保模型保存目录存在
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    train_loss_history = []
    val_loss_history = []

    # 初始设为正无穷，保证第一个 epoch 能够保存
    min_val_loss = float("inf")

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

        # ---------- 验证 ----------
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

        # ---------- 保存验证集上表现最好的模型 ----------
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model, save_path)

        print(
            "[%03d/%03d] %.2f sec(s)  train_loss: %.6f  val_loss: %.6f"
            % (
                epoch + 1,
                epochs,
                time.time() - start_time,
                train_loss,
                val_loss,
            )
        )

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


# ==================== 5. 测试集预测 ====================

def evaluate(model_path, test_loader, result_path, device):
    # 加载训练阶段保存的最佳模型
    model = torch.load(model_path, weights_only=False).to(device)

    predictions = []

    model.eval()

    with torch.no_grad():
        for x in test_loader:
            x = x.to(device)
            pred = model(x)

            # batch_size=1，所以每次拿出一个预测值
            predictions.append(pred.cpu().item())

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


# ==================== 6. 主程序 ====================

def main():
    # 以脚本所在目录为基准，避免从其他目录运行时找不到数据
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 数据路径
    train_file = os.path.join(base_dir, "data", "covid_train.csv")
    test_file = os.path.join(base_dir, "data", "covid_test.csv")

    # 超参数
    batch_size = 16
    epochs = 20
    lr = 0.001

    # 选择 CPU 或 GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # ---------- 数据集 ----------
    # 先建立训练集，因为验证集和测试集要使用训练集的 mean/std
    train_set = CovidDataset(train_file, "train")
    val_set = CovidDataset(
        train_file,
        "val",
        mean=train_set.mean,
        std=train_set.std,
    )
    test_set = CovidDataset(
        test_file,
        "test",
        mean=train_set.mean,
        std=train_set.std,
    )

    # ---------- DataLoader ----------
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
    )

    # ---------- 模型 ----------
    data_dim = 93
    model = MyModel(data_dim).to(device)

    # ---------- 保存路径 ----------
    save_path = os.path.join(base_dir, "models", "best_model.pth")
    result_path = os.path.join(base_dir, "results", "predictions.csv")

    # ---------- 损失函数和优化器 ----------
    loss_fn = mse_loss

    optimizer = optim.SGD(
        params=model.parameters(),
        lr=lr,
        momentum=0.9,
    )

    # ---------- 训练 + 验证 ----------
    train_val(
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        device,
        epochs,
        save_path,
    )

    # ---------- 测试 ----------
    evaluate(
        save_path,
        test_loader,
        result_path,
        device,
    )


if __name__ == "__main__":
    main()
