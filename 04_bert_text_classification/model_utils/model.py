import torch
import torch.nn as nn
import numpy as np
# from timm.models.vision_transformer import PatchEmbed, Block
import torchvision.models as models
import transformers
from transformers import (BertPreTrainedModel, BertConfig,
                          BertForSequenceClassification, BertTokenizer,BertModel,
                          )




class myBertModel(nn.Module):
    def __init__(self, bert_path, num_class, device):
        super(myBertModel, self).__init__()
        self.device = device

        # 修正：类别数已经由外部传入，不再固定写死为 2
        self.num_class = num_class

        # bert_config = BertConfig.from_pretrained(bert_path)
        # self.bert = BertModel(bert_config)
        self.bert = BertModel.from_pretrained(bert_path)
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)

        self.out = nn.Sequential(
            # bert-base-chinese 的隐藏层维度为 768，本项目保持原来的分类头结构
            nn.Linear(768,num_class)
        )



    def build_bert_input(self, text):
        Input = self.tokenizer(text,return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        input_ids = Input["input_ids"].to(self.device)
        attention_mask =  Input["attention_mask"].to(self.device)
        token_type_ids = Input["token_type_ids"].to(self.device)
        return input_ids, attention_mask, token_type_ids

    def forward(self, text):
        input_ids, attention_mask, token_type_ids = self.build_bert_input(text)
        sequence_out, pooled_output = self.bert(input_ids=input_ids,
                                                attention_mask=attention_mask,
                                                token_type_ids=token_type_ids,
                                                return_dict=False)

        out = self.out(pooled_output)
        return out




def model_Datapara(model, device,  pre_path=None):
    model = torch.nn.DataParallel(model).to(device)
    if pre_path != None:
        model_dict = torch.load(pre_path).module.state_dict()
        model.module.load_state_dict(model_dict)
    return model
