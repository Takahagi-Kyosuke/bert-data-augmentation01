#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertModel, BertConfig

import numpy as np
import pickle
import sys

#コマンドラインから引数(モデル)を受け取る
argvs = sys.argv
argc = len(argvs)

#学習済みの日本語BERTモデルを読み込む
#東北大版
config = BertConfig.from_pretrained('cl-tohoku/bert-base-japanese')
bert = BertModel(config=config)
#ストックマーク版
#config = BertConfig.from_json_file('ストックマーク版BERT/bert_config.json')
#bert = BertModel.from_pretrained('ストックマーク版BERT/pytorch_model.bin',config=config)



#GPUを利用する
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data Setting
with open('xtest.pkl','br') as fr:
    xtest = pickle.load(fr)
with open('ytest.pkl','br') as fr:
    ytest = pickle.load(fr)

# Define model
class DocCls(nn.Module):
    def __init__(self,bert):
        super(DocCls, self).__init__()
        self.bert = bert
        self.cls=nn.Linear(768,9)
    def forward(self,x):
        bout = self.bert(x)
        bs = len(bout[0])
        h0 = [ bout[0][i][0] for i in range(bs)]
        h0 = torch.stack(h0,dim=0)
        return self.cls(h0)

# model generate
net = DocCls(bert).to(device)
#保存したモデルを呼び出して使う
net.load_state_dict(torch.load(argvs[1]))

# Test

real_data_num, ok = 0, 0 #テストデータの数，正解数

#テスト時は以下の2行を追加(微分値の計算を行わないようにする)
net.eval()
with torch.no_grad():
    for i in range(len(xtest)):
        x = torch.LongTensor(xtest[i]).unsqueeze(0).to(device)
        ans = net(x)
        #print(ans.size())
        #ansの行の，最大要素のインデックスを返す
        #item()で、Python組み込み型（この場合は整数int）として要素の値を取得できる
        ans1 = torch.argmax(ans,dim=1).item()

        if (ans1 == ytest[i]): #予測結果と正解ラベルが一致したら
            ok += 1
        real_data_num += 1
print(ok, real_data_num, ok/real_data_num)
