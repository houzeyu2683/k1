
import torch
import torchvision
from torch import nn

class constant:

    version = '3.0.0'
    embedding = 50
    pass

class x1(nn.Module):

    def __init__(self):

        super(x1, self).__init__()
        layer = dict()
        layer['f01'] = nn.Sequential(
            nn.Linear(5, 64), nn.Sigmoid(), nn.Dropout(0.2), 
            nn.Linear(64, 128), nn.LeakyReLU(), nn.Dropout(0.2), 
            nn.Linear(128, 256), nn.LeakyReLU(), nn.Dropout(0.2),
            nn.Linear(256, 512), nn.LeakyReLU(), nn.Dropout(0.2),
            nn.Linear(512, 1024), nn.LeakyReLU(), nn.Dropout(0.2)
        )
        self.layer = nn.ModuleDict(layer)
        return

    def forward(self, x):

        y = self.layer['f01'](x)
        return(y)

    pass

class x2(nn.Module):

    def __init__(self):

        super(x2, self).__init__()
        layer = dict()
        layer['e01'] = nn.Embedding(352899, constant.embedding)
        layer['f01'] = nn.Sequential(nn.Linear(constant.embedding, 1024))
        self.layer = nn.ModuleDict(layer)
        return

    def forward(self, x):

        v = self.layer['e01'](x.transpose(0,1))
        y = self.layer['f01'](v).squeeze()
        return(y)

    pass

class x3(nn.Module):

    def __init__(self):

        super(x3, self).__init__()
        layer = dict()
        layer['e01'] = nn.Embedding(47226, constant.embedding)
        layer['e02'] = nn.Embedding(45877, constant.embedding)
        layer['e03'] = nn.Embedding(134, constant.embedding)
        layer['e04'] = nn.Embedding(133, constant.embedding)
        layer['e05'] = nn.Embedding(21, constant.embedding)
        layer['e06'] = nn.Embedding(32, constant.embedding)
        layer['e07'] = nn.Embedding(32, constant.embedding)
        layer['e08'] = nn.Embedding(52, constant.embedding)
        layer['e09'] = nn.Embedding(52, constant.embedding)
        layer['e10'] = nn.Embedding(10, constant.embedding)
        layer['e11'] = nn.Embedding(10, constant.embedding)
        layer['e12'] = nn.Embedding(22, constant.embedding)
        layer['e13'] = nn.Embedding(22, constant.embedding)
        layer['e14'] = nn.Embedding(301, constant.embedding)
        layer['e15'] = nn.Embedding(252, constant.embedding)
        layer['e16'] = nn.Embedding(12, constant.embedding)
        layer['e17'] = nn.Embedding(12, constant.embedding)
        layer['e18'] = nn.Embedding(7, constant.embedding)
        layer['e19'] = nn.Embedding(7, constant.embedding)
        layer['e20'] = nn.Embedding(59, constant.embedding)
        layer['e21'] = nn.Embedding(58, constant.embedding)
        layer['e22'] = nn.Embedding(23, constant.embedding)
        layer['e23'] = nn.Embedding(23, constant.embedding)
        layer['e24'] = nn.Embedding(43407, constant.embedding)
        layer['e25'] = nn.Embedding(105544, 512)
        layer['p01'] = nn.Embedding(2000, constant.embedding)
        layer['p02'] = nn.Embedding(2000, constant.embedding)
        layer['p03'] = nn.Embedding(2000, constant.embedding)
        layer['p04'] = nn.Embedding(2000, constant.embedding)
        layer['p05'] = nn.Embedding(2000, constant.embedding)
        layer['p06'] = nn.Embedding(2000, constant.embedding)
        layer['p07'] = nn.Embedding(2000, constant.embedding)
        layer['p08'] = nn.Embedding(2000, constant.embedding)
        layer['p09'] = nn.Embedding(2000, constant.embedding)
        layer['p10'] = nn.Embedding(2000, constant.embedding)
        layer['p11'] = nn.Embedding(2000, constant.embedding)
        layer['p12'] = nn.Embedding(2000, constant.embedding)
        layer['p13'] = nn.Embedding(2000, constant.embedding)
        layer['p14'] = nn.Embedding(2000, constant.embedding)
        layer['p15'] = nn.Embedding(2000, constant.embedding)
        layer['p16'] = nn.Embedding(2000, constant.embedding)
        layer['p17'] = nn.Embedding(2000, constant.embedding)
        layer['p18'] = nn.Embedding(2000, constant.embedding)
        layer['p19'] = nn.Embedding(2000, constant.embedding)
        layer['p20'] = nn.Embedding(2000, constant.embedding)
        layer['p21'] = nn.Embedding(2000, constant.embedding)
        layer['p22'] = nn.Embedding(2000, constant.embedding)
        layer['p23'] = nn.Embedding(2000, constant.embedding)
        layer['p24'] = nn.Embedding(2000, constant.embedding)
        layer['p25'] = nn.Embedding(2000, 512)
        layer['f01'] = nn.Sequential(nn.Linear((constant.embedding * 24) + 512, 128), nn.ReLU())
        layer['f02'] = nn.Sequential(nn.Linear((constant.embedding * 24) + 512, 256*4), nn.ReLU())
        layer['f03'] = nn.Sequential(nn.Linear((constant.embedding * 24) + 512, 256*4), nn.ReLU())
        layer['r01'] = nn.LSTM(128, 256, 4)
        self.layer = nn.ModuleDict(layer)
        return

    def forward(self, x):

        loop = range(x.shape[2])
        v = []
        for l in loop:

            e = x[:,:,l]
            p = x[:,:,l].clone()
            for r in range(len(p)): p[r,:] = r
            e = self.layer["e" + str(l+1).zfill(2)](e)
            p = self.layer["p" + str(l+1).zfill(2)](p)
            v =  v + [e + p]
            pass
        
        v = torch.cat(v, 2)
        s = self.layer['f01'](v)
        h = self.layer['f02'](v.sum(0))
        c = self.layer['f03'](v.sum(0))
        h = torch.stack(torch.split(h, int(h.shape[1]/4), 1), 0)
        c = torch.stack(torch.split(c, int(c.shape[1]/4), 1), 0)
        _,(m,_) = self.layer['r01'](s, (h, c))
        y = m.permute(1,0,2).flatten(1,-1)
        return(y)

    pass

class x4(nn.Module):

    def __init__(self):

        super(x4, self).__init__()
        layer = dict()
        layer['f01'] = nn.Sequential(nn.Linear(1,10), nn.ReLU())
        layer['f02'] = nn.Sequential(nn.Linear(10,256*4), nn.ReLU())
        layer['f03'] = nn.Sequential(nn.Linear(10,256*4), nn.ReLU())
        layer['r01'] = nn.LSTM(10, 256, 4)        
        self.layer = nn.ModuleDict(layer)
        return

    def forward(self, x):

        loop = range(len(x))
        v = []
        for l in loop:
            
            v = v + [self.layer['f01'](x[l:l+1,:].permute(1,0))]
            pass

        s = torch.stack(v, 0)
        h = self.layer['f02'](s.sum(0))
        c = self.layer['f03'](s.sum(0))
        h = torch.stack(torch.split(h, int(h.shape[1]/4), 1), 0)
        c = torch.stack(torch.split(c, int(c.shape[1]/4), 1), 0)
        _,(m,_) = self.layer['r01'](s, (h, c))
        y = m.permute(1,0,2).flatten(1,-1)
        return(y)

    pass

class x5(nn.Module):

    def __init__(self):

        super(x5, self).__init__()
        layer = dict()
        layer['e01'] = nn.Embedding(4, 10)
        layer['p01'] = nn.Embedding(2000, 10)
        layer['f01'] = nn.Sequential(nn.Linear(10, 10), nn.ReLU())
        layer['f02'] = nn.Sequential(nn.Linear(10,256*4), nn.ReLU())
        layer['f03'] = nn.Sequential(nn.Linear(10,256*4), nn.ReLU())
        layer['r01'] = nn.LSTM(10, 256, 4) 
        self.layer =nn.ModuleDict(layer)

    def forward(self, x):

        p = x.clone()
        for r in range(len(p)): p[r,:] = r
        v = self.layer['e01'](x) + self.layer['p01'](p)
        s = self.layer['f01'](v)
        h = self.layer['f02'](s.sum(0))
        c = self.layer['f03'](s.sum(0))
        h = torch.stack(torch.split(h, int(h.shape[1]/4), 1), 0)
        c = torch.stack(torch.split(c, int(c.shape[1]/4), 1), 0)
        _,(m,_) = self.layer['r01'](s, (h, c))
        y = m.permute(1,0,2).flatten(1,-1)
        return(y)

    pass

class model(nn.Module):

    def __init__(self):

        super(model, self).__init__()
        layer = dict()
        layer['x1'] = x1()
        layer['x2'] = x2()
        layer['x3'] = x3()
        layer['x4'] = x4()
        layer['x5'] = x5()
        layer['a1'] = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(1024, 4, 2048), 
            num_layers=2, 
            norm=None
        )
        layer['f1'] = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        layer['f2'] = nn.Sequential(nn.Linear(512, 512), nn.Sigmoid())
        self.layer = nn.ModuleDict(layer)
        return

    def forward(self, x):

        v = [
            self.layer['x1'](x[0]),
            self.layer['x2'](x[1]),
            self.layer['x3'](x[2]),
            self.layer['x4'](x[3]),
            self.layer['x5'](x[4])
        ]
        v = torch.stack(v, 0)
        v = self.layer['a1'](v)
        v = self.layer['f1'](v).sum(0)
        v = self.layer['f2'](v)    
        e = self.layer['x3'].layer['e25'](x[5][:,0:1].permute(1,0)).squeeze()
        e = e + self.layer['x3'].layer['p25'](x[5][:,1:2].permute(1,0)).squeeze()
        t = x[5][:,2:3].squeeze()
        y = (v, e, t)
        return(y)

    pass

        