
import torch
import torchvision
from torch import nn

class constant:

    version = '4.0.0'
    sequence = {
        "article_code":105545, 
        "sales_channel_id":5,
        "product_code":47227, 
        "prod_name":45878, 
        "product_type_no":135, 
        "product_type_name":134,
        "product_group_name":22, 
        "graphical_appearance_no":33, 
        "graphical_appearance_name":33, 
        "colour_group_code":53, 
        "colour_group_name":53, 
        "perceived_colour_value_id":11, 
        "perceived_colour_value_name":11, 
        "perceived_colour_master_id":23, 
        "perceived_colour_master_name":23, 
        "department_no":302, 
        "department_name":253, 
        "index_code":13, 
        "index_name":13, 
        "index_group_no":8, 
        "index_group_name":8, 
        "section_no":60, 
        "section_name":59, 
        "garment_group_no":24, 
        "garment_group_name":24, 
        "detail_desc":43408,
        'price':1
    }
    pass

##  用戶基本訊息模組.
class row(nn.Module):

    def __init__(self):

        super(row, self).__init__()
        layer = dict()
        layer['f1'] = nn.Sequential(
            nn.Linear(5, 64), nn.Sigmoid(), nn.Dropout(0.2), 
            nn.Linear(64, 128), nn.LeakyReLU(), nn.Dropout(0.2), 
            nn.Linear(128, 256), nn.LeakyReLU(), nn.Dropout(0.2)
        )
        layer['e1'] = nn.Embedding(352899, 256)
        layer['f2'] = nn.Sequential(nn.Linear(256, 128), nn.LeakyReLU(), nn.Dropout(0.2))
        layer['f3'] = nn.Sequential(nn.Linear(256+128, 256), nn.LeakyReLU(), nn.Dropout(0.2))
        layer['f4'] = nn.Sequential(nn.Linear(256, 512), nn.LeakyReLU(), nn.Dropout(0.2))
        layer['f5'] = nn.Sequential(nn.Linear(256, 512), nn.LeakyReLU(), nn.Dropout(0.2))
        self.layer = nn.ModuleDict(layer)
        return

    def forward(self, x='[n, c]'):

        v = [
            self.layer['f1'](x[0]),
            self.layer['f2'](self.layer['e1'](x[1]).squeeze())
        ]
        v = torch.cat(v, dim=1)
        v = self.layer['f3'](v)
        h, c = self.layer['f4'](v), self.layer['f5'](v)
        h = torch.stack(torch.split(h, 128, 1))
        c = torch.stack(torch.split(c, 128, 1))
        y = (h, c)        
        return(y)

    def status(self):

        x = torch.randn((7, 5)), torch.randint(0, 10, (1, 7))
        y = self.forward(x)
        status = '1'
        return(status, y)

    pass

##  序列模組.
class sequence(nn.Module):

    def __init__(self, level=20):

        super(sequence, self).__init__()
        self.limit = 2000
        self.level = level
        pass

        if(self.level>1):

            layer = dict()
            layer['e1'] = nn.Embedding(self.level, 256)
            layer['p1'] = nn.Embedding(self.limit, 256)
            layer['r1'] = nn.LSTM(256, 128, 4)
            layer['f1'] = nn.Sequential(nn.Linear(512,512), nn.Sigmoid(), nn.Dropout(0.1))
            pass
        
        else:

            layer = dict()
            layer['f1'] = nn.Sequential(nn.Linear(1,256))
            layer['r1'] = nn.LSTM(256, 128, 4)
            layer['f2'] = nn.Sequential(nn.Linear(512,512), nn.Sigmoid(), nn.Dropout(0.1))
            pass
        
        layer['o1'] = nn.Sequential(nn.Linear(512, self.level))
        self.layer = nn.ModuleDict(layer)
        return

    def forward(self, x='[s, (h, c)]'):

        if(self.level>1):

            p = x[0].clone()
            for i in range(len(p)): p[i,:] = i 
            v = self.layer['e1'](x[0]) + self.layer['p1'](p)
            _, (h, _) = self.layer['r1'](v, x[1])
            h = h.permute(1,0,2).flatten(1,-1)
            y = self.layer['f1'](h)
            pass
        
        else:

            v = self.layer['f1'](x[0].unsqueeze(-1))
            _, (h, _) = self.layer['r1'](v, x[1])
            h = h.permute(1,0,2).flatten(1,-1)
            y = self.layer['f2'](h)              
            pass

        return(y)

    def status(self):

        if(self.level>1):

            x = [torch.randint(0, 20, (3, 7)), (torch.randn((4, 7, 128)), torch.randn((4, 7, 128)))]
            y = self.forward(x)
            status = '1'
            pass

        else:

            x = [torch.randn((3, 7)), (torch.randn((4, 7, 128)), torch.randn((4, 7, 128)))]
            y = self.forward(x)
            status = '1'
            pass

        return(status, y)

    pass

class model(nn.Module):

    def __init__(self):

        super(model, self).__init__()
        layer = dict()
        layer['r1'] = row()
        for i, (_,v) in enumerate(constant.sequence.items(), 1): layer["s"+str(i)] = sequence(level=v)
        layer['a1'] = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(512, 4), num_layers=4, norm=None
        )
        self.layer = nn.ModuleDict(layer)
        return

    def forward(self, x='batch'):

        h, c = self.layer['r1']([x[0],x[1]])
        s = []
        for i, (_,_) in enumerate(constant.sequence.items(), 1): 
            
            s += [self.layer["s"+str(i)]([x[2][i-1][0], (h,c)])]
            pass

        a = self.layer['a1'](torch.stack(s))
        o = []
        for i, (_,_) in enumerate(constant.sequence.items(), 1): 
            
            o += [self.layer["s"+str(i)].layer['o1'](a[i-1,:,:]).squeeze()]
            pass        
        
        y = o
        return(y)

    pass

