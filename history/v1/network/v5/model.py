
import torch
import torchvision
from torch import nn

class constant:

    version = '4.0.0'
    embedding = {
        'club_member_status':4,
        "fashion_news_frequency":5,
        "postal_code":352899,
        "article_code":105545, 
        # "t_dat_delta":733,
        # 'article_code_delta':4,
    }
    pass

class mask:

    def padding(x="(length, batch)", value=0):

        y = (x==value).transpose(0,1)
        y = y.cuda() if(x.is_cuda) else y.cpu()
        return(y)

    def sequence(x="(length, batch)", recourse=False):

        length = len(x)
        if(not recourse): y = torch.full((length,length), bool(False))
        if(recourse): y = torch.triu(torch.full((length,length), float('-inf')), diagonal=1)
        y = y.cuda() if(x.is_cuda) else y.cpu()
        return(y)

    pass

class position:

    def encode(x='(length, batch)'):

        y = x.clone()
        for i in range(len(y)): y[i,:] = i
        return(y)

    pass

class vector(nn.Module):

    def __init__(self):

        super(vector, self).__init__()
        layer = dict()
        l = 'default(f1)'
        layer[l] = nn.Sequential(
            nn.Linear(3, 64), nn.LeakyReLU(), nn.Dropout(0.2)
        )
        pass

        l = 'club_member_status(e1)'
        layer[l] = nn.Embedding(constant.embedding['club_member_status'], 32)
        pass

        l = 'fashion_news_frequency(e1)'
        layer[l] = nn.Embedding(constant.embedding['fashion_news_frequency'], 32)
        pass

        l = 'postal_code(e1)'
        layer[l] = nn.Embedding(constant.embedding['postal_code'], 128)
        pass

        l = 'default(f2)'
        layer[l] = nn.Sequential(
            nn.Linear(64+32+32+128, 64), 
            nn.Tanh(), 
            nn.Dropout(0.2)
        )
        pass

        self.layer = nn.ModuleDict(layer)
        return

    def forward(self, b='batch'):

        '''
        >>> batch['vector(numeric)'].shape
        torch.Size([7, 3])
        >>> batch['vector(club_member_status)'].shape
        torch.Size([1, 7])
        >>> batch['vector(fashion_news_frequency)'].shape
        torch.Size([1, 7])
        >>> batch['vector(postal_code)'].shape
        torch.Size([1, 7])
        '''
        v = []
        l = 'default(f1)'
        v += [self.layer[l](b["vector(numeric)"])]
        l = 'club_member_status(e1)'
        v += [self.layer[l](b["vector(club_member_status)"]).squeeze(0)]
        l = 'fashion_news_frequency(e1)'
        v += [self.layer[l](b["vector(fashion_news_frequency)"]).squeeze(0)]
        l = 'postal_code(e1)'
        v += [self.layer[l](b["vector(postal_code)"]).squeeze(0)]
        v = torch.cat(v, 1)
        l = 'default(f2)'
        y = self.layer[l](v)
        return(y)

    pass

class sequence(nn.Module):

    def __init__(self, ):

        super(sequence, self).__init__()
        layer = dict()
        pass
    
        encoder = nn.TransformerEncoderLayer(1, 1)
        layer['price(a1)'] = nn.TransformerEncoder(encoder_layer=encoder, num_layers=1, norm=None)
        pass

        layer['article_code(e1)'] = nn.Embedding(constant.embedding['article_code'], 256)
        encoder = nn.TransformerEncoderLayer(256, 8)
        layer['article_code(a1)'] = nn.TransformerEncoder(encoder_layer=encoder, num_layers=4, norm=None)
        pass

        layer['default(f1)'] = nn.Sequential(nn.Linear(1+256, 64), nn.ReLU(), nn.Dropout(0.2))
        encoder = nn.TransformerEncoderLayer(64, 8)
        layer['default(a1)'] = nn.TransformerEncoder(encoder_layer=encoder, num_layers=2, norm=None)
        layer['default(f2)'] = nn.Sequential(nn.Linear(64, 256), nn.ReLU())
        pass

        self.layer = nn.ModuleDict(layer)
        return

    def forward(self, b='batch'):

        '''
        >>> batch['sequence(price)']['history'].shape
        torch.Size([3, 7, 1])
        >>> batch['sequence(article_code)']['history'].shape
        torch.Size([3, 7])
        >>> batch['sequence(t_dat_delta)']['history'].shape
        torch.Size([3, 7])
        >>> batch['sequence(article_code_delta)']['history'].shape
        torch.Size([3, 7])
        '''
        v = {}
        l = "sequence(price)"
        v[l] = self.layer['price(a1)'](b[l]['history'])
        pass

        l = 'sequence(article_code)'
        v[l] = self.layer['article_code(e1)'](b[l]['history'])
        pass
        
        v = torch.cat([i for i in v.values()], 2)
        v = self.layer["default(f1)"](v)
        v = v + self.layer["default(a1)"](v)
        v = self.layer['default(f2)'](v)
        y = v  ##  (length, batch, 256)
        return(y)
    
    pass

class suggestion(nn.Module):

    def __init__(self):

        super(suggestion, self).__init__()
        layer = dict()
        layer['vector'] = vector()
        layer['sequence'] = sequence()
        layer['default(f1)'] = nn.Sequential(nn.Linear(320, 256), nn.ReLU(), nn.Dropout(0.2))
        encoder = nn.TransformerEncoderLayer(256, 2)
        layer['default(a1)'] = nn.TransformerEncoder(encoder_layer=encoder, num_layers=2, norm=None)
        decoder = nn.TransformerDecoderLayer(256, 2)
        layer['default(a2)'] = nn.TransformerDecoder(decoder_layer=decoder, num_layers=2, norm=None)
        layer['default(f2)'] = nn.Sequential(nn.Linear(256, constant.embedding['article_code']), nn.ReLU(), nn.Softmax(2))
        self.layer = nn.ModuleDict(layer)
        return

    def forward(self, b='batch'):

        # cache = dict()
        v = self.layer['vector'](b)
        s = self.layer['sequence'](b)
        pass

        l = range(len(s))
        v = torch.cat([v.unsqueeze(0) for _ in l], 0)
        m = self.layer['default(f1)'](torch.cat([s, v], 2))
        m = self.layer['default(a1)'](m)
        pass

        f = b['sequence(article_code)']['future'][:-1,:]
        f = self.layer['default(a2)'](
            tgt = self.layer['sequence'].layer['article_code(e1)'](f), 
            memory = m, 
            tgt_mask = mask.sequence(f, True), 
            memory_mask = None, 
            tgt_key_padding_mask = mask.padding(f, 0), 
            memory_key_padding_mask = None
        )
        pass

        y = self.layer['default(f2)'](f)
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
