

##  假設輸入維度是(batch,embedding,length)的張量.
##  使用C1D進行特徵萃取,並保證length不變.
##  輸出與輸入維度一致.
import torch
import torch.nn as nn
batch, embedding, length = 4, 256, 7
x = torch.randn((batch, embedding, length))
print(x.shape)
c1 = nn.Conv1d(embedding, embedding, 1)
y1 = c1(x)
print(y1.shape)
c2 = nn.Conv1d(embedding, embedding, 3, padding=1)
y2 = c2(x)
print(y2.shape)
c3 = nn.Conv1d(embedding, embedding, 5, padding=2)
y3 = c3(x)
print(y3.shape)
c4 = nn.Conv1d(embedding, embedding, 7, padding=3)
y4 = c4(x)
print(y4.shape)
pass


##  假設輸入維度是(length, batch,embedding)的張量.
##  使用GRU進行特徵萃取,並保證length不變.
##  輸出結果有兩個,第一個是最後layer所有length的萃取特徵.
##  第二個是所有layer最後的length的萃取特徵,layer數量體現在第一個維度.
##  所以GRU是可以設置多層,
import torch
import torch.nn as nn
batch, embedding, length = 1, 4, 3
layer = 1
x = torch.randn((length, batch, embedding))
print(x.shape)
g1 = nn.GRU(embedding, embedding, layer)
h = torch.randn(layer, batch, embedding)
y1, h1 = g1(x, h)
y1.shape
h1.shape
