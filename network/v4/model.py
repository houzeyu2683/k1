
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

# ##  數值序列模組.
# class price(nn.Module):

#     def __init__(self):

#         super(price, self).__init__()
#         layer = dict()
#         layer['f1'] = nn.Sequential(nn.Linear(1,256))
#         layer['r1'] = nn.LSTM(256, 128, 4)
#         layer['f2'] = nn.Sequential(nn.Linear(512,512), nn.Sigmoid(), nn.Dropout(0.1))
#         self.layer = nn.ModuleDict(layer)
#         return

#     def forward(self, x='[s, (h, c)]'):

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

    # def status(self):

    #     [torch.randn(6, 7)] + [torch.randint(0, 10, (6,7)) for _ in range(26)]
    # pass
# class m2(nn.Module):

#     def __init__(self, level=20):

#         super(m2, self).__init__()
#         self.limit = 2000
#         self.level = level
#         layer = dict()
#         if(self.level>1):

#             layer['e1'] = nn.Embedding(, 256)
#             layer['p1'] = nn.Embedding(self.limit, 256)
#             layer['r1'] = nn.LSTM(256, 128, 4)
#             layer['f1'] = nn.Sequential(nn.Linear(512,512), nn.Sigmoid(), nn.Dropout(0.1))
#             pass

#         else:

#             layer['f1'] = nn.Sequential(nn.Linear(1,256))
#             layer['r1'] = nn.LSTM(256, 128, 4)
#             layer['f2'] = nn.Sequential(nn.Linear(512,512), nn.Sigmoid(), nn.Dropout(0.1))
#             pass

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

#         if(self.level>0):

#             x = torch.randint(0, 20, (7, 2)), (torch.randn((4, 2, 128)), torch.randn((4, 2, 128)))
#             y = self.forward(x)
#             status = '1'
#             pass

#         else:

#             x = torch.randn((7, 2)), (torch.randn((4, 2, 128)), torch.randn((4, 2, 128)))
#             y = self.forward(x)
#             status = '1'
#             pass

#         return(status, y)

#     pass

##  序列特徵進行注意力編碼機制.
# class m3(nn.Module):

#     def __init__(self, embedding=512):

#         super(m3, self).__init__()
#         self.embedding = embedding
#         layer = dict()
#         layer['a1'] = nn.TransformerEncoder(
#             encoder_layer=nn.TransformerEncoderLayer(self.embedding, 4), num_layers=4, norm=None
#         )
#         self.layer = nn.ModuleDict(layer)
#         return

#     def forward(self, x=''):

#         y = self.layer['a1'](x)
#         return(y)

#     def status(self):

#         x = torch.randn((27, 2, self.embedding))
#         y = self.forward(x)
#         status = '1'
#         return(status, y)

#     pass

# ##  預測模組.
# class m4(nn.Module):

#     def __init__(self, level=20):

#         super(m4, self).__init__()
#         self.level = level
#         layer = dict()
#         layer['f1'] = nn.Sequential(nn.Linear(512, level))
#         self.layer = nn.ModuleDict(layer)
#         return

#     def forward(self, x=''):

#         y = self.layer['f1'](x)
#         return(y)

#     def status(self):

#         x = torch.randn((16, 512))
#         y = self.forward(x)
#         status = '1'
#         return(status, y)

#     pass

# ##  主架構.
# class model(nn.Module):

#     def __init__(self):

#         super(model, self).__init__()
#         layer = dict()
#         layer['c1'] = m1()
#         layer['s1'] = m2(level=105545)
#         layer['s2'] = m2(level=47227)
#         layer['s3'] = m2(level=45878)
#         layer['s4'] = m2(level=135)
#         layer['s5'] = m2(level=134)
#         layer['s6'] = m2(level=22)
#         layer['s7'] = m2(level=33)
#         layer['s8'] = m2(level=33)
#         layer['s9'] = m2(level=53)
#         layer['s10'] = m2(level=53)
#         layer['s11'] = m2(level=11)
#         layer['s12'] = m2(level=11)
#         layer['s13'] = m2(level=23)
#         layer['s14'] = m2(level=23)
#         layer['s15'] = m2(level=302)
#         layer['s16'] = m2(level=253)
#         layer['s17'] = m2(level=13)
#         layer['s18'] = m2(level=13)
#         layer['s19'] = m2(level=8)
#         layer['s20'] = m2(level=8)
#         layer['s21'] = m2(level=60)
#         layer['s22'] = m2(level=59)
#         layer['s23'] = m2(level=24)
#         layer['s24'] = m2(level=24)
#         layer['s25'] = m2(level=43408)
#         layer['s26'] = m2(level=1)
#         layer['s27'] = m2(level=5)
#         layer['a1'] = m3(embedding=512)
#         layer['t1'] = m4(level=105545)
#         layer['t2'] = m4(level=47227)
#         layer['t3'] = m4(level=45878)
#         layer['t4'] = m4(level=135)
#         layer['t5'] = m4(level=134)
#         layer['t6'] = m4(level=22)
#         layer['t7'] = m4(level=33)
#         layer['t8'] = m4(level=33)
#         layer['t9'] = m4(level=53)
#         layer['t10'] = m4(level=53)
#         layer['t11'] = m4(level=11)
#         layer['t12'] = m4(level=11)
#         layer['t13'] = m4(level=23)
#         layer['t14'] = m4(level=23)
#         layer['t15'] = m4(level=302)
#         layer['t16'] = m4(level=253)
#         layer['t17'] = m4(level=13)
#         layer['t18'] = m4(level=13)
#         layer['t19'] = m4(level=8)
#         layer['t20'] = m4(level=8)
#         layer['t21'] = m4(level=60)
#         layer['t22'] = m4(level=59)
#         layer['t23'] = m4(level=24)
#         layer['t24'] = m4(level=24)
#         layer['t25'] = m4(level=43408)
#         layer['t26'] = m4(level=1)
#         layer['t27'] = m4(level=5)
#         self.layer = nn.ModuleDict(layer)
#         return

#     def forward(self, b='dict()'):

#         h, c = layer['c1']([b['x1'], b['x2']])
#         s = torch.stack(
#             [
#                 layer['s1']([b['s1'], (h,c)]),
#                 layer['s2']([b['s2'], (h, c)]),
#                 layer['s3']([b['s3'], (h, c)]),
#                 layer['s4']([b['s4'], (h, c)]),
#                 layer['s5']([b['s5'], (h, c)]),
#                 layer['s6']([b['s6'], (h, c)]),
#                 layer['s7']([b['s7'], (h, c)]),
#                 layer['s8']([b['s8'], (h, c)]),
#                 layer['s9']([b['s9'], (h, c)]),
#                 layer['s10']([b['s10'], (h, c)]),
#                 layer['s11']([b['s11'], (h, c)]),
#                 layer['s12']([b['s12'], (h, c)]),
#                 layer['s13']([b['s13'], (h, c)]),
#                 layer['s14']([b['s14'], (h, c)]),
#                 layer['s15']([b['s15'], (h, c)]),
#                 layer['s16']([b['s16'], (h, c)]),
#                 layer['s17']([b['s17'], (h, c)]),
#                 layer['s18']([b['s18'], (h, c)]),
#                 layer['s19']([b['s19'], (h, c)]),
#                 layer['s20']([b['s20'], (h, c)]),
#                 layer['s21']([b['s21'], (h, c)]),
#                 layer['s22']([b['s22'], (h, c)]),
#                 layer['s23']([b['s23'], (h, c)]),
#                 layer['s24']([b['s24'], (h, c)]),
#                 layer['s25']([b['s25'], (h, c)]),
#                 layer['s26']([b['s26'], (h, c)]),
#                 layer['s27']([b['s27'], (h, c)])
#             ]
#         )
#         a = layer['a1'](s)
#         y = [
#             layer['t1'](a[0,:,:]),
#             layer['t2'](a[1,:,:]),
#             layer['t3'](a[2,:,:]),
#             layer['t4'](a[3,:,:]),
#             layer['t5'](a[4,:,:]),
#             layer['t6'](a[5,:,:]),
#             layer['t7'](a[6,:,:]),
#             layer['t8'](a[7,:,:]),
#             layer['t9'](a[8,:,:]),
#             layer['t10'](a[9,:,:]),
#             layer['t11'](a[10,:,:]),
#             layer['t12'](a[11,:,:]),
#             layer['t13'](a[12,:,:]),
#             layer['t14'](a[13,:,:]),
#             layer['t15'](a[14,:,:]),
#             layer['t16'](a[15,:,:]),
#             layer['t17'](a[16,:,:]),
#             layer['t18'](a[17,:,:]),            
#             layer['t19'](a[18,:,:]),
#             layer['t20'](a[19,:,:]),
#             layer['t21'](a[20,:,:]),
#             layer['t22'](a[21,:,:]),
#             layer['t23'](a[22,:,:]),
#             layer['t24'](a[23,:,:]),
#             layer['t25'](a[24,:,:]),
#             layer['t26'](a[25,:,:]),
#             layer['t27'](a[26,:,:])
#         ]
        
#         return(y)

#     pass


# m1().status()
# m2().status()
# _, y = m3().status()
# y.shape
# m4(level=1).status()[1].shape

# class x2(nn.Module):

#     def __init__(self):

#         super(x2, self).__init__()
#         layer = dict()
#         layer['e01'] = nn.Embedding(352899, 256)
#         layer['f01'] = nn.Sequential(nn.Linear(256, constant.embedding))
#         self.layer = nn.ModuleDict(layer)
#         return

#     def forward(self, x):

#         v = self.layer['e01'](x.transpose(0,1))
#         y = self.layer['f01'](v).squeeze()
#         return(y)

#     pass

# class xy3(nn.Module):

#     def __init__(self):

#         super(xy3, self).__init__()
#         information = dict()
#         information['name'] = 'product_code'
#         information['number'] = 47226
#         self.information = information
#         layer = dict()
#         layer['e01'] = nn.Embedding(self.information['number'], 256)
#         layer['p01'] = nn.Embedding(2000, 256)
#         layer['f01'] = nn.Sequential(nn.Linear(256, 50), nn.ReLU())
#         layer['f02'] = nn.Sequential(nn.Linear(256, 50), nn.ReLU())
#         layer['f03'] = nn.Sequential(
#             nn.Linear(constant.embedding, self.information['number']), 
#             nn.Softmax(dim=1)
#         )
#         layer['r01'] = nn.LSTM(256, int(constant.embedding / 5), 5)
#         self.layer = nn.ModuleDict(layer)
#         return

#     def forward(self, x='(sequence length, batch size)'):

#         v = self.layer['e01'](x) + self.layer['p01'](x)
#         h, c = self.layer['f01'](v[0,:,:]), self.layer['f02'](v[0,:,:])
#         h = torch.stack(torch.split(h, int(constant.embedding / 5), 1))
#         c = torch.stack(torch.split(c, int(constant.embedding / 5), 1))
#         _,(o,_) = self.layer['r01'](v, (h,c))
#         o = o.permute(1,0,2).flatten(1,-1)
#         y = self.layer['f03'](o)
#         return(y)
    
#     pass

# class xy4(nn.Module):

#     def __init__(self):

#         super(xy4, self).__init__()
#         information = dict()
#         information['name'] = 'prod_name'
#         information['number'] = 45877
#         self.information = information
#         layer = dict()
#         layer['e01'] = nn.Embedding(self.information['number'], 256)
#         layer['p01'] = nn.Embedding(2000, 256)
#         layer['f01'] = nn.Sequential(nn.Linear(256, 50), nn.ReLU())
#         layer['f02'] = nn.Sequential(nn.Linear(256, 50), nn.ReLU())
#         layer['f03'] = nn.Sequential(
#             nn.Linear(constant.embedding, self.information['number']), 
#             nn.Softmax(dim=1)
#         )
#         layer['r01'] = nn.LSTM(256, int(constant.embedding / 5), 5)
#         self.layer = nn.ModuleDict(layer)
#         return

#     def forward(self, x='(sequence length, batch size)'):

#         v = self.layer['e01'](x) + self.layer['p01'](x)
#         h, c = self.layer['f01'](v[0,:,:]), self.layer['f02'](v[0,:,:])
#         h = torch.stack(torch.split(h, int(constant.embedding / 5), 1))
#         c = torch.stack(torch.split(c, int(constant.embedding / 5), 1))
#         _,(o,_) = self.layer['r01'](v, (h,c))
#         o = o.permute(1,0,2).flatten(1,-1)
#         y = self.layer['f03'](o)
#         return(y)
    
#     pass

# class xy5(nn.Module):

#     def __init__(self):

#         super(xy5, self).__init__()
#         information = dict()
#         information['name'] = 'product_type_no'
#         information['number'] = 134
#         self.information = information
#         layer = dict()
#         layer['e01'] = nn.Embedding(self.information['number'], 256)
#         layer['p01'] = nn.Embedding(2000, 256)
#         layer['f01'] = nn.Sequential(nn.Linear(256, 50), nn.ReLU())
#         layer['f02'] = nn.Sequential(nn.Linear(256, 50), nn.ReLU())
#         layer['f03'] = nn.Sequential(
#             nn.Linear(constant.embedding, self.information['number']), 
#             nn.Softmax(dim=1)
#         )
#         layer['r01'] = nn.LSTM(256, int(constant.embedding / 5), 5)
#         self.layer = nn.ModuleDict(layer)
#         return

#     def forward(self, x='(sequence length, batch size)'):

#         v = self.layer['e01'](x) + self.layer['p01'](x)
#         h, c = self.layer['f01'](v[0,:,:]), self.layer['f02'](v[0,:,:])
#         h = torch.stack(torch.split(h, int(constant.embedding / 5), 1))
#         c = torch.stack(torch.split(c, int(constant.embedding / 5), 1))
#         _,(o,_) = self.layer['r01'](v, (h,c))
#         o = o.permute(1,0,2).flatten(1,-1)
#         y = self.layer['f03'](o)
#         return(y)
    
#     pass
# class x4(nn.Module):

#     def __init__(self):

#         super(x4, self).__init__()
#         layer = dict()
#         layer['f01'] = nn.Sequential(nn.Linear(1,10), nn.ReLU())
#         layer['f02'] = nn.Sequential(nn.Linear(10,256*4), nn.ReLU())
#         layer['f03'] = nn.Sequential(nn.Linear(10,256*4), nn.ReLU())
#         layer['r01'] = nn.LSTM(10, 256, 4)        
#         self.layer = nn.ModuleDict(layer)
#         return

#     def forward(self, x):

#         loop = range(len(x))
#         v = []
#         for l in loop:
            
#             v = v + [self.layer['f01'](x[l:l+1,:].permute(1,0))]
#             pass

#         s = torch.stack(v, 0)
#         h = self.layer['f02'](s.sum(0))
#         c = self.layer['f03'](s.sum(0))
#         h = torch.stack(torch.split(h, int(h.shape[1]/4), 1), 0)
#         c = torch.stack(torch.split(c, int(c.shape[1]/4), 1), 0)
#         _,(m,_) = self.layer['r01'](s, (h, c))
#         y = m.permute(1,0,2).flatten(1,-1)
#         return(y)

#     pass

# class x5(nn.Module):

#     def __init__(self):

#         super(x5, self).__init__()
#         layer = dict()
#         layer['e01'] = nn.Embedding(4, 10)
#         layer['p01'] = nn.Embedding(2000, 10)
#         layer['f01'] = nn.Sequential(nn.Linear(10, 10), nn.ReLU())
#         layer['f02'] = nn.Sequential(nn.Linear(10,256*4), nn.ReLU())
#         layer['f03'] = nn.Sequential(nn.Linear(10,256*4), nn.ReLU())
#         layer['r01'] = nn.LSTM(10, 256, 4) 
#         self.layer =nn.ModuleDict(layer)

#     def forward(self, x):

#         p = x.clone()
#         for r in range(len(p)): p[r,:] = r
#         v = self.layer['e01'](x) + self.layer['p01'](p)
#         s = self.layer['f01'](v)
#         h = self.layer['f02'](s.sum(0))
#         c = self.layer['f03'](s.sum(0))
#         h = torch.stack(torch.split(h, int(h.shape[1]/4), 1), 0)
#         c = torch.stack(torch.split(c, int(c.shape[1]/4), 1), 0)
#         _,(m,_) = self.layer['r01'](s, (h, c))
#         y = m.permute(1,0,2).flatten(1,-1)
#         return(y)

#     pass

# class model(nn.Module):

#     def __init__(self):

#         super(model, self).__init__()
#         layer = dict()
#         layer['x1'] = x1()
#         layer['x2'] = x2()
#         layer['x3'] = x3()
#         layer['x4'] = x4()
#         layer['x5'] = x5()
#         layer['a1'] = nn.TransformerEncoder(
#             encoder_layer=nn.TransformerEncoderLayer(1024, 4, 2048), 
#             num_layers=2, 
#             norm=None
#         )
#         layer['f1'] = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
#         layer['f2'] = nn.Sequential(nn.Linear(512, 512), nn.Sigmoid())
#         self.layer = nn.ModuleDict(layer)
#         return

#     def forward(self, x):

#         v = [
#             self.layer['x1'](x[0]),
#             self.layer['x2'](x[1]),
#             self.layer['x3'](x[2]),
#             self.layer['x4'](x[3]),
#             self.layer['x5'](x[4])
#         ]
#         v = torch.stack(v, 0)
#         v = self.layer['a1'](v)
#         v = self.layer['f1'](v).sum(0)
#         v = self.layer['f2'](v)    
#         e = self.layer['x3'].layer['e25'](x[5][:,0:1].permute(1,0)).squeeze()
#         e = e + self.layer['x3'].layer['p25'](x[5][:,1:2].permute(1,0)).squeeze()
#         t = x[5][:,2:3].squeeze()
#         y = (v, e, t)
#         return(y)

#     pass

        