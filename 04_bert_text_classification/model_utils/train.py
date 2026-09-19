import torch
import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm









def train_val(para):

########################################################
    model = para['model']
    train_loader =para['train_loader']
    val_loader = para['val_loader']
    scheduler = para['scheduler']
    optimizer = para['optimizer']
    loss = para['loss']
    epoch = para['epoch']
    device = para['device']
    save_path = para['save_path']
    max_acc = para['max_acc']
    val_epoch = para['val_epoch']


#################################################
    plt_train_loss = []
    plt_train_acc = []
    plt_val_loss = []
    plt_val_acc = []

    for i in range(epoch):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        val_acc = 0.0
        val_loss = 0.0

        for batch in tqdm(train_loader):
            # 保留原项目在每个 batch 开始时清空梯度的写法
            model.zero_grad()

            text, labels = batch[0], batch[1].to(device)
            pred = model(text)
            bat_loss = loss(pred, labels)
            bat_loss.backward()

            # 修正：梯度裁剪必须发生在 backward() 之后、optimizer.step() 之前，
            # 否则参数已经更新后再裁剪梯度就没有作用了
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            # 修正：CrossEntropyLoss 默认是当前 batch 的平均 loss。
            # 乘上 batch 样本数后，最后再除以整个数据集样本数，得到正确的样本平均 loss
            train_loss += bat_loss.item() * labels.size(0)

            # 保留原来的准确率统计逻辑
            train_acc += np.sum(
                np.argmax(pred.detach().cpu().numpy(),axis=1) == labels.cpu().numpy()
            )

        plt_train_loss.append(train_loss/train_loader.dataset.__len__())
        plt_train_acc.append(train_acc/train_loader.dataset.__len__())

        if i % val_epoch == 0:
            model.eval()
            with torch.no_grad():
                for batch in tqdm(val_loader):
                    val_text, val_labels = batch[0], batch[1].to(device)
                    val_pred = model(val_text)
                    val_bat_loss = loss(val_pred, val_labels)

                    # 修正：与训练 loss 一样，先乘当前 batch 的样本数
                    val_loss += val_bat_loss.item() * val_labels.size(0)

                    val_acc += np.sum(
                        np.argmax(val_pred.detach().cpu().numpy(), axis=1) == val_labels.cpu().numpy()
                    )

            plt_val_loss.append(val_loss/val_loader.dataset.__len__())

            # 修正：val_acc 上面累计的是“预测正确的样本数量”，
            # 先除以验证集大小得到真正的 0~1 准确率，再与 max_acc 比较
            current_val_acc = val_acc/val_loader.dataset.__len__()
            plt_val_acc.append(current_val_acc)

            if current_val_acc > max_acc:
                # 修正：保存验证准确率最好的模型
                torch.save(model, save_path+'best_model.pth')
                max_acc = current_val_acc

            print('[%03d/%03d] %2.2f sec(s) TrainAcc : %3.6f TrainLoss : %3.6f | valAcc: %3.6f valLoss: %3.6f  ' % \
                  (i, epoch, time.time()-start_time, plt_train_acc[-1], plt_train_loss[-1], plt_val_acc[-1], plt_val_loss[-1])
                  )

            if i % 50 == 0:
                # 修正：Windows 文件名不能包含冒号 ":"，改成下划线，并补充 .pth 后缀
                torch.save(model, save_path+'epoch_'+str(i)+ '-%.2f.pth'%plt_val_acc[-1])
        else:
            plt_val_loss.append(plt_val_loss[-1])
            plt_val_acc.append(plt_val_acc[-1])
            print('[%03d/%03d] %2.2f sec(s) TrainAcc : %3.6f TrainLoss : %3.6f   ' % \
                  (i, epoch, time.time()-start_time, plt_train_acc[-1], plt_train_loss[-1])
                  )

    plt.plot(plt_train_loss)
    plt.plot(plt_val_loss)
    plt.title('loss')
    plt.legend(['train', 'val'])
    plt.show()

    plt.plot(plt_train_acc)
    plt.plot(plt_val_acc)
    plt.title('Accuracy')
    plt.legend(['train', 'val'])
    plt.savefig('acc.png')
    plt.show()
