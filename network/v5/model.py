
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

class mask:

    def padding(x="(length, batch)", value=0):

        y = (x==value).transpose(0,1)
        y = y.cuda() if(x.is_cuda) else y.cpu()
        return(y)

    def sequence(x="(length, batch)", recourse=False):

        if(not recourse):

            length = len(x)
            y = torch.full((length,length), bool(False))
            y = y.cuda() if(x.is_cuda) else y.cpu()
            pass

        else:

            length = len(x)
            y = torch.triu(torch.full((length,length), float('-inf')), diagonal=1)
            y = y.cuda() if(x.is_cuda) else y.cpu()
            pass

        return(y)

    pass

'''
batch['row(numeric)']
tensor([[1.0000, 1.0000, 0.4300],
        [1.0000, 1.0000, 0.5000]])
batch['row(numeric)'].shape
torch.Size([2, 3])

batch['row(category)']
tensor([[     0,      0],
        [     4,      4],
        [178400,  90870]])
batch['row(category)'].shape
torch.Size([3, 2])

batch['sequence(price)']['history']
tensor([[[0.0000],
         [1.1905]]])
batch['sequence(price)']['history'].shape
torch.Size([1, 2, 1])

batch['sequence(price)']['future']
tensor([[[1.0412],
         [1.1003]],
        [[1.0458],
         [0.0000]]])
batch['sequence(price)']['future'].shape
torch.Size([2, 2, 1])

batch['sequence(article_code)']['history']
tensor([[    1, 42792]])
batch['sequence(article_code)']['history'].shape
torch.Size([1, 2])

batch['sequence(article_code)']['future']
tensor([[23890, 42615],
        [23890,     0]])
batch['sequence(article_code)']['future'].shape
torch.Size([2, 2])
>>> 
'''

class row(nn.Module):

    def __init__(self):

        super(row, self).__init__()
        layer = dict()
        layer['f1'] = nn.Sequential(
            nn.Linear(3, 64), nn.LeakyReLU(), nn.Dropout(0.2), 
            nn.Linear(64, 128), nn.LeakyReLU(), nn.Dropout(0.2)
        )
        layer['e1'] = nn.Embedding(4, 16)
        layer['e2'] = nn.Embedding(5, 25)
        layer['e3'] = nn.Embedding(352899, 256)
        layer['f2'] = nn.Sequential(
            nn.Linear(128+256+16+25, 128), 
            nn.LeakyReLU(), 
            nn.Dropout(0.2)
        )
        self.layer = nn.ModuleDict(layer)
        return

    def forward(self, x='batch'):

        v = self.layer['f1'](x["row(numeric)"])
        l = x["row(category)"].split(1)
        e = [
            v,
            self.layer['e1'](l[0]).squeeze(0),
            self.layer['e2'](l[1]).squeeze(0),
            self.layer['e3'](l[2]).squeeze(0)
        ]
        v = torch.cat(e, 1)
        y = self.layer['f2'](v)
        return(y)

    pass

class sequence(nn.Module):

    def __init__(self):

        super(sequence, self).__init__()
        layer = dict()

        ##  Sequence(price)
        layer['f1'] = nn.Sequential(
            nn.Linear(1,8), nn.LeakyReLU(), nn.Dropout(0.2),
            nn.Linear(8,32), nn.LeakyReLU(), nn.Dropout(0.2)
        )
        layer['f2'] = nn.Sequential(
            nn.Linear(1,8), nn.LeakyReLU(), nn.Dropout(0.2),
            nn.Linear(8,32), nn.LeakyReLU(), nn.Dropout(0.2)
        )
        layer['f3'] = nn.Sequential(
            nn.Linear(1,8), nn.LeakyReLU(), nn.Dropout(0.2),
            nn.Linear(8,32), nn.LeakyReLU(), nn.Dropout(0.2)
        )
        layer['r1'] = nn.LSTM(32, 32, 1)
        pass

        ##  Sequence(article_code)
        layer['e1'] = nn.Embedding(105542, 64)
        layer['p1'] = nn.Embedding(2000, 64)
        layer['a1'] = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(64, 4), num_layers=2, norm=None
        )
        layer['f4'] = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.1))
        pass

        self.layer = nn.ModuleDict(layer)
        return

    def forward(self, x='batch'):

        cache = {}
        pass
        s = self.layer['f1'](x['sequence(price)']["history"])
        h = self.layer['f2'](x['sequence(price)']["history"][0,:,:]).unsqueeze(0)
        c = self.layer['f3'](x['sequence(price)']["history"][0,:,:]).unsqueeze(0)
        _, (v, _)= self.layer['r1'](s, (h, c))
        v = v.permute(1,0,2).flatten(1,-1)
        cache['sequence(price)'] = v
        pass

        e = self.layer['e1'](x["sequence(article_code)"]['history'])
        p = self.layer['p1'](self.position(x["sequence(article_code)"]['history']))
        v = e + p
        cache['mask(padding)'] = mask.padding(x["sequence(article_code)"]['history'], 0)
        v = self.layer['a1'](src=v, mask=None, src_key_padding_mask=cache['mask(padding)']) + v 
        v = self.layer['f4'](v)
        cache['sequence(article_code)'] = v
        pass

        y = cache
        return(y)
    
    def position(self, e):

        p = e.clone()
        for i in range(len(p)): p[i,:] = i
        return(p)

    pass

class suggestion(nn.Module):

    def __init__(self):

        super(suggestion, self).__init__()
        layer = dict()
        layer['row'] = row()
        layer['sequence'] = sequence()
        layer['next(price)'] = nn.Sequential(
            nn.Linear(128 + 32 + 64, 1)
        )
        layer['next(article_code)'] = nn.Sequential(
            nn.Linear(128 + 32 + 64, 105542)
        )
        layer['embedding(article_code)'] = nn.Sequential(nn.Linear(128+32+64, 64), nn.Sigmoid())
        self.layer = nn.ModuleDict(layer)
        return

    def forward(self, x='batch'):

        cache = dict()
        row = self.layer['row'](x)
        history = self.layer['sequence'](x)
        v = torch.cat(
            [
                row,
                history['sequence(price)'],
                history['sequence(article_code)'].sum(0)
            ], 
            dim = 1
        )
        cache['next(price)'] = self.layer['next(price)'](v)
        cache['next(article_code)'] = self.layer['next(article_code)'](v)
        cache['embedding(article_code)'] = self.layer['embedding(article_code)'](v)
        y = cache
        return(y)

    pass

class model(nn.Module):

    def __init__(self):

        super(model, self).__init__()
        layer = dict()
        layer['suggestion'] = suggestion()
        self.layer = nn.ModuleDict(layer)
        return

    def forward(self, x='batch'):

        cache = dict()
        pass
    
        cache['day(1)'] = self.layer['suggestion'](x)
        u = [x['sequence(price)']['history'], cache['day(1)']['next(price)'].unsqueeze(0)]
        x['sequence(price)']['history'] = torch.cat(u, 0)
        u = [x['sequence(article_code)']['history'] , cache['day(1)']['next(article_code)'].argmax(1).unsqueeze(0)]
        x['sequence(article_code)']['history'] = torch.cat(u, 0)
        pass

        cache['day(2)'] = self.layer['suggestion'](x)
        u = [x['sequence(price)']['history'], cache['day(1)']['next(price)'].unsqueeze(0)]
        x['sequence(price)']['history'] = torch.cat(u, 0)
        u = [x['sequence(article_code)']['history'] , cache['day(1)']['next(article_code)'].argmax(1).unsqueeze(0)]
        x['sequence(article_code)']['history'] = torch.cat(u, 0)
        pass

        cache['day(3)'] = self.layer['suggestion'](x)
        u = [x['sequence(price)']['history'], cache['day(1)']['next(price)'].unsqueeze(0)]
        x['sequence(price)']['history'] = torch.cat(u, 0)
        u = [x['sequence(article_code)']['history'] , cache['day(1)']['next(article_code)'].argmax(1).unsqueeze(0)]
        x['sequence(article_code)']['history'] = torch.cat(u, 0)
        pass

        cache['day(4)'] = self.layer['suggestion'](x)
        u = [x['sequence(price)']['history'], cache['day(1)']['next(price)'].unsqueeze(0)]
        x['sequence(price)']['history'] = torch.cat(u, 0)
        u = [x['sequence(article_code)']['history'] , cache['day(1)']['next(article_code)'].argmax(1).unsqueeze(0)]
        x['sequence(article_code)']['history'] = torch.cat(u, 0)
        pass

        cache['day(5)'] = self.layer['suggestion'](x)
        u = [x['sequence(price)']['history'], cache['day(1)']['next(price)'].unsqueeze(0)]
        x['sequence(price)']['history'] = torch.cat(u, 0)
        u = [x['sequence(article_code)']['history'] , cache['day(1)']['next(article_code)'].argmax(1).unsqueeze(0)]
        x['sequence(article_code)']['history'] = torch.cat(u, 0)
        pass

        cache['day(6)'] = self.layer['suggestion'](x)
        u = [x['sequence(price)']['history'], cache['day(1)']['next(price)'].unsqueeze(0)]
        x['sequence(price)']['history'] = torch.cat(u, 0)
        u = [x['sequence(article_code)']['history'] , cache['day(1)']['next(article_code)'].argmax(1).unsqueeze(0)]
        x['sequence(article_code)']['history'] = torch.cat(u, 0)
        pass

        cache['day(7)'] = self.layer['suggestion'](x)
        u = [x['sequence(price)']['history'], cache['day(1)']['next(price)'].unsqueeze(0)]
        x['sequence(price)']['history'] = torch.cat(u, 0)
        u = [x['sequence(article_code)']['history'] , cache['day(1)']['next(article_code)'].argmax(1).unsqueeze(0)]
        x['sequence(article_code)']['history'] = torch.cat(u, 0)
        pass

        cache['day(8)'] = self.layer['suggestion'](x)
        u = [x['sequence(price)']['history'], cache['day(1)']['next(price)'].unsqueeze(0)]
        x['sequence(price)']['history'] = torch.cat(u, 0)
        u = [x['sequence(article_code)']['history'] , cache['day(1)']['next(article_code)'].argmax(1).unsqueeze(0)]
        x['sequence(article_code)']['history'] = torch.cat(u, 0)
        pass

        cache['day(9)'] = self.layer['suggestion'](x)
        u = [x['sequence(price)']['history'], cache['day(1)']['next(price)'].unsqueeze(0)]
        x['sequence(price)']['history'] = torch.cat(u, 0)
        u = [x['sequence(article_code)']['history'] , cache['day(1)']['next(article_code)'].argmax(1).unsqueeze(0)]
        x['sequence(article_code)']['history'] = torch.cat(u, 0)
        pass

        cache['day(10)'] = self.layer['suggestion'](x)
        u = [x['sequence(price)']['history'], cache['day(1)']['next(price)'].unsqueeze(0)]
        x['sequence(price)']['history'] = torch.cat(u, 0)
        u = [x['sequence(article_code)']['history'] , cache['day(1)']['next(article_code)'].argmax(1).unsqueeze(0)]
        x['sequence(article_code)']['history'] = torch.cat(u, 0)
        pass

        cache['day(11)'] = self.layer['suggestion'](x)
        u = [x['sequence(price)']['history'], cache['day(1)']['next(price)'].unsqueeze(0)]
        x['sequence(price)']['history'] = torch.cat(u, 0)
        u = [x['sequence(article_code)']['history'] , cache['day(1)']['next(article_code)'].argmax(1).unsqueeze(0)]
        x['sequence(article_code)']['history'] = torch.cat(u, 0)
        pass
    
        cache['day(12)'] = self.layer['suggestion'](x)
        u = [x['sequence(price)']['history'], cache['day(1)']['next(price)'].unsqueeze(0)]
        x['sequence(price)']['history'] = torch.cat(u, 0)
        u = [x['sequence(article_code)']['history'] , cache['day(1)']['next(article_code)'].argmax(1).unsqueeze(0)]
        x['sequence(article_code)']['history'] = torch.cat(u, 0)
        pass

        ##  
        for s in ['next(price)', 'next(article_code)', 'embedding(article_code)']:

            l = [
                cache['day(1)'][s],
                cache['day(2)'][s],
                cache['day(3)'][s],
                cache['day(4)'][s],
                cache['day(5)'][s],
                cache['day(6)'][s],
                cache['day(7)'][s],
                cache['day(8)'][s],
                cache['day(9)'][s],
                cache['day(10)'][s],
                cache['day(11)'][s],
                cache['day(12)'][s],                
            ]
            cache[s] = torch.stack(l)
            pass

        return(cache)

    pass

# 'day(1)', 'day(2)', 'day(3)', 'day(4)', 'day(5)', 'day(6)', 'day(7)', 'day(8)', 'day(9)', 'day(10)', 'day(11)', 'day(12)'
#         v = self.layer['f1'](x[0].unsqueeze(-1))
#         _, (h, _) = self.layer['r1'](v, x[1])
#         h = h.permute(1,0,2).flatten(1,-1)
#         y = self.layer['f2'](h)  
#         return(y)

#     def status(self):

#         x = [torch.randn((3, 7)), (torch.randn((4, 7, 128)), torch.randn((4, 7, 128)))]
#         y = self.forward(x)
#         status = '1'
#         return(status, y)

#     pass

##  序列模組.
# class sequence(nn.Module):

#     def __init__(self, level=20):

#         super(sequence, self).__init__()
#         self.limit = 2000
#         self.level = level
#         pass

#         if(self.level>1):

#             layer = dict()
#             layer['e1'] = nn.Embedding(self.level, 256)
#             layer['p1'] = nn.Embedding(self.limit, 256)
#             layer['r1'] = nn.LSTM(256, 128, 4)
#             layer['f1'] = nn.Sequential(nn.Linear(512,512), nn.Sigmoid(), nn.Dropout(0.1))
#             pass
        
#         else:

#             layer = dict()
#             layer['f1'] = nn.Sequential(nn.Linear(1,256))
#             layer['r1'] = nn.LSTM(256, 128, 4)
#             layer['f2'] = nn.Sequential(nn.Linear(512,512), nn.Sigmoid(), nn.Dropout(0.1))
#             pass
        
#         layer['o1'] = nn.Sequential(nn.Linear(512, self.level))
#         self.layer = nn.ModuleDict(layer)
#         return

#     def forward(self, x='[s, (h, c)]'):

#         if(self.level>1):

#             p = x[0].clone()
#             for i in range(len(p)): p[i,:] = i 
#             v = self.layer['e1'](x[0]) + self.layer['p1'](p)
#             _, (h, _) = self.layer['r1'](v, x[1])
#             h = h.permute(1,0,2).flatten(1,-1)
#             y = self.layer['f1'](h)
#             pass
        
#         else:

#             v = self.layer['f1'](x[0].unsqueeze(-1))
#             _, (h, _) = self.layer['r1'](v, x[1])
#             h = h.permute(1,0,2).flatten(1,-1)
#             y = self.layer['f2'](h)              
#             pass

#         return(y)

#     def status(self):

#         if(self.level>1):

#             x = [torch.randint(0, 20, (3, 7)), (torch.randn((4, 7, 128)), torch.randn((4, 7, 128)))]
#             y = self.forward(x)
#             status = '1'
#             pass

#         else:

#             x = [torch.randn((3, 7)), (torch.randn((4, 7, 128)), torch.randn((4, 7, 128)))]
#             y = self.forward(x)
#             status = '1'
#             pass

#         return(status, y)

#     pass

# class model(nn.Module):

#     def __init__(self):

#         super(model, self).__init__()
#         layer = dict()
#         layer['r1'] = row()
#         for i, (_,v) in enumerate(constant.sequence.items(), 1): layer["s"+str(i)] = sequence(level=v)
#         layer['a1'] = nn.TransformerEncoder(
#             encoder_layer=nn.TransformerEncoderLayer(512, 4), num_layers=4, norm=None
#         )
#         self.layer = nn.ModuleDict(layer)
#         return

#     def forward(self, x='batch'):

#         h, c = self.layer['r1']([x[0],x[1]])
#         s = []
#         for i, (_,_) in enumerate(constant.sequence.items(), 1): 
            
#             s += [self.layer["s"+str(i)]([x[2][i-1][0], (h,c)])]
#             pass

#         a = self.layer['a1'](torch.stack(s))
#         o = []
#         for i, (_,_) in enumerate(constant.sequence.items(), 1): 
            
#             o += [self.layer["s"+str(i)].layer['o1'](a[i-1,:,:]).squeeze()]
#             pass        
        
#         y = o
#         return(y)
