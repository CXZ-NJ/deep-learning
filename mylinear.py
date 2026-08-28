import torch
import matplotlib.pyplot as plt #  画图的

import random  #随机


def create_data(w, b, data_num):  #生成数据
    x = torch.normal(0, 1, (data_num, len(w)))
    y = torch.matmul(x, w) + b    #matmul表示矩阵相乘

    noise = torch.normal(0, 0.01, y.shape)  #噪声要加到y上
    y += noise

    return x, y

num = 500

true_w = torch.tensor([8.1,2,2,4])
true_b = torch.tensor(1.1)


X, Y = create_data(true_w, true_b, num)


plt.scatter(X[:, 3], Y, 1)
plt.show()


def data_provider(data, label, batchsize):       #每次访问这个函数， 就能提供一批数据
    length = len(label)
    indices = list(range(length))
    #我不能按顺序取  把数据打乱
    random.shuffle(indices)

    for each in range(0, length, batchsize):
        get_indices = indices[each: each+batchsize]
        get_data = data[get_indices]
        get_label = label[get_indices]

        yield get_data,get_label  #有存档点的return

batchsize = 16
# for batch_x, batch_y in data_provider(X, Y, batchsize):
#     print(batch_x, batch_y)
#     break


def fun(x, w, b):
    pred_y = torch.matmul(x, w) + b
    return pred_y

def maeLoss(pre_y, y):
    return torch.sum(abs(pre_y-y))/len(y)

def sgd(paras, lr):          #随机梯度下降，更新参数
    with torch.no_grad():  #属于这句代码的部分，不计算梯度
        for para in paras:
            para -= para.grad * lr      #不能写成   para = para - para.grad*lr
            para.grad.zero_()      #使用过的梯度，归0


lr = 0.03
w_0 = torch.normal(0, 0.01, true_w.shape, requires_grad=True)   #这个w需要计算梯度
b_0 = torch.tensor(0.01, requires_grad=True)
print(w_0, b_0)


epochs = 50

for epoch in range(epochs):
    data_loss = 0
    for batch_x, batch_y in data_provider(X, Y, batchsize):
        pred_y = fun(batch_x,w_0, b_0)
        loss = maeLoss(pred_y, batch_y)
        loss.backward()
        sgd([w_0, b_0], lr)
        data_loss += loss

    print("epoch %03d: loss: %.6f"%(epoch, data_loss))


print("真实的函数值是", true_w, true_b)
print("训练得到的参数值是", w_0, b_0)

idx = 0
plt.plot(X[:, idx].detach().numpy(), X[:, idx].detach().numpy()*w_0[idx].detach().numpy()+b_0.detach().numpy())
plt.scatter(X[:, idx], Y, 1)
plt.show()

