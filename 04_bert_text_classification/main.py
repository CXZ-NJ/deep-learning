import torch.nn as nn
import torch
import random
import numpy  as np
import os

from model_utils.data import get_dataloader
from model_utils.model import myBertModel
from model_utils.train import train_val


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


model_name = 'MyModel'
num_class = 2
batchSize = 4
learning_rate = 0.0001
loss = nn.CrossEntropyLoss()
epoch = 3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_path = "jiudian.txt"
bert_path = 'bert-base-chinese'
save_path = 'model_save/'

# 修正：如果模型保存文件夹不存在，则先创建，避免 torch.save() 保存时报错
os.makedirs(save_path, exist_ok=True)

seed_everything(1)
##########################################

train_loader, val_loader = get_dataloader(data_path, batchsize=batchSize)

model = myBertModel(bert_path, num_class, device).to(device)
# tokenizer, model = build_model_and_tokenizer(bert_path)

param_optimizer = list(model.parameters())
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=20,eta_min=1e-9)




trainpara = {'model': model,
             'train_loader': train_loader,
             'val_loader': val_loader,
             'scheduler': scheduler,
             'optimizer': optimizer,
             'learning_rate': learning_rate,
             'warmup_ratio' : 0.1,
             'weight_decay' : 0.0001,
             'use_lookahead' : True,
             'loss': loss,
             'epoch': epoch,
             'device': device,
             'save_path': save_path,
             # max_acc 表示 0~1 范围内的验证准确率，例如 0.85 表示 85%
             'max_acc': 0.85,
             'val_epoch' : 1
             }
train_val(trainpara)
