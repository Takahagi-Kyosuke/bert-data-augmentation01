#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertModel,BertConfig

import numpy as np
import pickle
import sys
#from sklearn.model_selection import train_test_split

#コマンドラインから引数を受け取る
argvs = sys.argv
argc = len(argvs)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

##################
#   DataLoader   #
##################

class MyDataset(Dataset):
    def __init__(self, xdata, ydata):
        self.data = xdata
        self.label = ydata
    def __len__(self): #labelの長さを返す
        return len(self.label)
    def __getitem__(self, idx): #dataとlabelのidx番目の要素を返す
        x = self.data[idx]
        y = self.label[idx]
        return (x,y)

#バッチの形をデフォルトから変更
def my_collate_fn(batch): 
    images, targets= list(zip(*batch))
    xs = list(images)
    ys = list(targets)
    return xs, ys   

with open(argvs[1],'br') as fr:
    xtrain = pickle.load(fr)
with open(argvs[2],'br') as fr:
    ytrain = pickle.load(fr)

with open('xval.pkl','br') as fr:
    xval = pickle.load(fr)
with open('yval.pkl','br') as fr:
    yval = pickle.load(fr)

#dataloaderを作成
batch_size = 1
train_dataset = MyDataset(xtrain,ytrain) #MyDatasetのインスタンスを生成
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,   #バッチサイズ(一度に取り出すデータの数)
    shuffle=False,            #データの順序がシャッフルされる
    collate_fn=my_collate_fn #自作のcollate_fnを使用
    )

#学習用のモデルの定義
#学習済みの日本語BERTモデルを読み込む
#東北大版
bert = BertModel.from_pretrained('cl-tohoku/bert-base-japanese')
#ストックマーク版
#config = BertConfig.from_json_file('ストックマーク版BERT/bert_config.json')
#bert = BertModel.from_pretrained('ストックマーク版BERT/pytorch_model.bin',config=config)

class DocCls(nn.Module):
    def __init__(self,bert):
        super(DocCls, self).__init__()
        self.bert = bert
        self.cls=nn.Linear(768,9)
    def forward(self,x1,x2):
        #print(x1.size(),x2.size())
        #attention_mask→モデルが注意を払うべきトークンの判別に利用
        #1が注意を払うべきトークン、0が埋め込み
        bout = self.bert(input_ids=x1, attention_mask=x2)
        bs = len(bout[0])
        h0 = [ bout[0][i][0] for i in range(bs)]
        h0 = torch.stack(h0,dim=0)
        output = self.cls(h0)
        return output

#検証用のモデルの定義
class DocCls2(nn.Module):
    def __init__(self,bert):
        super(DocCls2, self).__init__()
        self.bert = bert
        self.cls=nn.Linear(768,9)
    def forward(self,x):
        bout = self.bert(x)
        bs = len(bout[0])
        h0 = [ bout[0][i][0] for i in range(bs)]
        h0 = torch.stack(h0,dim=0)
        return self.cls(h0)

# model generate, optimizer and criterion setting
net = DocCls(bert).to(device)
optimizer = optim.SGD(net.parameters(),lr=0.001)
criterion = nn.CrossEntropyLoss()

net2 = DocCls2(bert).to(device)

#earystopping
class EarlyStopping:
    """earlystoppingクラス"""

    def __init__(self, patience, verbose, path):
        """引数：最小値の非更新数カウンタ、表示設定、モデル格納path"""

        self.patience = patience    #設定ストップカウンタ
        self.verbose = verbose      #表示の有無
        self.counter = 0            #現在のカウンタ値
        self.best_score = None      #ベストスコア
        self.early_stop = False     #ストップフラグ
        self.val_acc_max = 0   #前回のベストスコア記憶用
        self.path = path             #ベストモデル格納path

    def __call__(self, val_acc, model):
        """
        特殊(call)メソッド
        実際に学習ループ内で最小lossを更新したか否かを計算させる部分
        """
        score = val_acc

        if self.best_score is None:  #1Epoch目の処理
            self.best_score = score   #1Epoch目はそのままベストスコアとして記録する
            self.checkpoint(val_acc, model)  #記録後にモデルを保存してスコア表示する
        elif score < self.best_score:  # ベストスコアを更新できなかった場合
            self.counter += 1   #ストップカウンタを+1
            if self.verbose:  #表示を有効にした場合は経過を表示
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')  #現在のカウンタを表示する 
            if self.counter >= self.patience:  #設定カウントを上回ったらストップフラグをTrueに変更
                self.early_stop = True
        else:  #ベストスコアを更新した場合
            self.best_score = score  #ベストスコアを上書き
            self.checkpoint(val_acc, model)  #モデルを保存してスコア表示
            self.counter = 0  #ストップカウンタリセット

    def checkpoint(self, val_acc, model):
        '''ベストスコア更新時に実行されるチェックポイント関数'''
        if self.verbose:  #表示を有効にした場合は、前回のベストスコアからどれだけ更新したか？を表示
            print(f'Validation accuracy increased ({self.val_acc_max:.6f} --> {val_acc:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)  #ベストモデルを指定したpathに保存
        self.val_acc_max = val_acc  #その時のlossを記録する

#学習
def train_net(dataloader, net, optimizer, criterion, device):
    
    i, lossK = 0, 0.0 

    net.train()
    for xs, ys in dataloader: #xsは要素が単語id列のリスト，ysはラベルのリスト(要素数=バッチサイズ)
        xs1, xmsk = [], [] #要素がtensorのリスト→[tensor([]),tensor([]),...tensor([])]
        for k in range(len(xs)): 
            tid = xs[k]
            xs1.append(torch.LongTensor(tid))
            xmsk.append(torch.LongTensor([1] * len(tid)))

        #pad_sequece:長さの違うテンソルを与えると, 短いものの末尾にゼロ埋めを施して次元を揃えてくれる関数
        #例...xs1[0]の長さ480，xs1[1]の長さ403だったら，両方長さ480に揃えてくれる
        #[tensor([]),tensor([]),...tensor([])]をtensor([[],[],...[]])に直してくれる
        xs1 = pad_sequence(
            xs1,             
            batch_first=True #(seq_len, batch, input_size)→(batch, seq_len, input_size)
            ).to(device)
        xmsk = pad_sequence(
            xmsk, 
            batch_first=True
            ).to(device)
        ys = torch.LongTensor(ys).to(device)

        outputs = net(xs1,xmsk)
    
        loss = criterion(outputs, ys)
        lossK += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        i += 1
    return net

#検証
def val_net(xval, yval, net, device):#, earlystopping):

    real_data_num, ok = 0, 0 #テストデータの数，正解数

    #テスト時は以下の2行を追加(微分値の計算を行わないようにする)
    net.eval()
    with torch.no_grad():
        for i in range(len(xval)):
            x = torch.LongTensor(xval[i]).unsqueeze(0).to(device)
            ans = net(x)
            #ansの行の，最大要素のインデックスを返す
            #item()で、Python組み込み型（この場合は整数int）として要素の値を取得できる
            ans1 = torch.argmax(ans,dim=1).item()

            if (ans1 == yval[i]): #予測結果と正解ラベルが一致したら
                ok += 1
            real_data_num += 1

    return ok/real_data_num

earlystopping = EarlyStopping(patience=5, verbose=True, path=argvs[3])
epochs = 30
flag = 0
for ep in range(epochs):
    #学習
    net = train_net(
        dataloader=train_dataloader,
        net=net,
        optimizer=optimizer,
        criterion=criterion,
        device = device
        )
    print("{} epoch finished".format(ep))

    #学習したモデルを検証用モデルに読み込む
    outfile = "tmp.model"
    torch.save(net.state_dict(),outfile)
    net2.load_state_dict(torch.load("tmp.model"))

    #検証
    accuracy = val_net(
        xval=xval,
        yval=yval,
        net=net2,
        device = device,
        #earlystopping=earlystopping
    )

    #early stopping
    earlystopping(accuracy, net) #callメソッド呼び出し
    if earlystopping.early_stop: #ストップフラグがTrueの場合、breakでforループを抜ける
        print("Early Stopping!")
        break