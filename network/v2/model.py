
import torch
import torchvision
from torch import nn

class constant:

    version = '1.0.0'
    article = 150
    pass

class record(nn.Module):

    def __init__(self):

        super(record, self).__init__()
        layer = dict()
        layer['e01'] = nn.Embedding(47226, constant.article)
        layer['e02'] = nn.Embedding(45877, constant.article)
        layer['e03'] = nn.Embedding(134, constant.article)
        layer['e04'] = nn.Embedding(133, constant.article)
        layer['e05'] = nn.Embedding(21, constant.article)
        layer['e06'] = nn.Embedding(32, constant.article)
        layer['e07'] = nn.Embedding(32, constant.article)
        layer['e08'] = nn.Embedding(52, constant.article)
        layer['e09'] = nn.Embedding(52, constant.article)
        layer['e10'] = nn.Embedding(10, constant.article)
        layer['e11'] = nn.Embedding(10, constant.article)
        layer['e12'] = nn.Embedding(22, constant.article)
        layer['e13'] = nn.Embedding(22, constant.article)
        layer['e14'] = nn.Embedding(301, constant.article)
        layer['e15'] = nn.Embedding(252, constant.article)
        layer['e16'] = nn.Embedding(12, constant.article)
        layer['e17'] = nn.Embedding(12, constant.article)
        layer['e18'] = nn.Embedding(7, constant.article)
        layer['e19'] = nn.Embedding(7, constant.article)
        layer['e20'] = nn.Embedding(59, constant.article)
        layer['e21'] = nn.Embedding(58, constant.article)
        layer['e22'] = nn.Embedding(23, constant.article)
        layer['e23'] = nn.Embedding(23, constant.article)
        layer['e24'] = nn.Embedding(43407, constant.article)
        layer['e25'] = nn.Embedding(105544, constant.article)
        layer['a1'] = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(25*constant.article, 5),
            num_layers=2
        )
        layer['f1'] = nn.Sequential(nn.Linear(25*constant.article, 512), nn.Tanh(), nn.Dropout(0.2))
        self.layer = nn.ModuleDict(layer)
        return

    def forward(self, x="(length, batch, index)"):

        group = []
        for iteration, index in enumerate(range(x.shape[2]), 1):

            group = group + [self.layer["e"+str(iteration).zfill(2)](x[:,:,index])]
            pass

        group = torch.stack(group, 3).flatten(2,3)
        group = self.layer['a1'](group)
        ##  (batch, hidden)
        y = self.layer['f1'](group[0,:,:])
        return(y)

class model(nn.Module):

    def __init__(self):

        super(model, self).__init__()
        layer = dict()
        layer['record'] = record()
        layer['f1']     = nn.Sequential(nn.Linear(604, 512), nn.Tanh(), nn.Dropout(0.2))
        layer['f2']     = nn.Sequential(nn.Linear(512+512, 512), nn.Tanh(), nn.Dropout(0.2))
        layer['f3']     = nn.Sequential(nn.Linear(512, 105544), nn.Sigmoid())
        layer['f4']     = nn.Sequential(nn.Linear(512, constant.article), nn.Sigmoid())
        self.layer = nn.ModuleDict(layer)
        return

    def forward(self, x='[row, sequence, target]'):

        group = torch.cat([self.layer['f1'](x[0]), self.layer['record'](x[1])], 1)
        group = self.layer['f2'](group)
        y = self.layer['f3'](group), self.layer['f4'](group), self.layer['record'].layer['e25'](x[2])
        return(y)

    pass
